from dataclasses import asdict
from collections import defaultdict
from posixpath import split
import torch
import numpy as np
import pickle

from copy import deepcopy
from typing import List
from trajdata import UnifiedDataset, AgentBatch, AgentType
from trajdata.simulation import SimulationScene
from trajdata.simulation import sim_metrics

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.vis_utils import render_state_trajdata
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from tbsim.policies.common import RolloutAction, Action
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.timer import Timers
from tbsim.utils.trajdata_utils import parse_trajdata_batch, get_drivable_region_map,locate_traffic_lights, is_offroad_by_heading
from tbsim.utils.rollout_logger import RolloutLogger
from tbsim.utils.smoothing_utils import smooth_positions
from tbsim.utils.agent_rel_classify import determine_centerline_relationship,plot_relationships_separately,filter_scenes,save_filtered_data_as_json
from torch.nn.utils.rnn import pad_sequence
from trajdata.data_structures.state import StateArray
agent_types=[AgentType.UNKNOWN,AgentType.VEHICLE,AgentType.PEDESTRIAN,AgentType.BICYCLE,AgentType.MOTORCYCLE]

from tbsim.configs.selected_scene_config import PREDEFINED_SCENE_INIT


class EnvUnifiedSimulation(BaseEnv, BatchedEnv):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            log_data=True,
            renderer=None,
            restart_track_id=True,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = dataset.num_scenes()
        self._num_scenes = num_scenes

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes
        self._current_scene_indices = None

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self._cached_raw_observation = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._log_data = log_data
        self.logger = None
        self.restart_track_id = restart_track_id

    def update_random_seed(self, seed):
        self._npr = np.random.RandomState(seed=seed)
    



    @property
    def current_scene_names(self):
        return deepcopy([scene.scene.name for scene in self._current_scenes])

    @property
    def current_num_agents(self):
        return sum(len(scene.agents) for scene in self._current_scenes)

    def reset_multi_episodes_metrics(self):
        for v in self._metrics.values():
            v.multi_episode_reset()

    @property
    def current_agent_scene_index(self):
        si = []
        for scene_i, scene in zip(self.current_scene_index, self._current_scenes):
            si.extend([scene_i] * len(scene.agents))
        return np.array(si, dtype=np.int64)

    @property
    def current_agent_track_id(self):
        if self.restart_track_id:
            id_by_scene=defaultdict(lambda :0)
            track_id = np.zeros_like(self.current_agent_scene_index)
            for i in range(self.current_agent_scene_index.shape[0]):
                track_id[i] = id_by_scene[self.current_agent_scene_index[i]]
                id_by_scene[self.current_agent_scene_index[i]]+=1
            return track_id
        else:
            return np.arange(self.current_num_agents)

    @property
    def current_scene_index(self):
        return self._current_scene_indices.copy()

    @property
    def current_agent_names(self):
        names = []
        for scene in self._current_scenes:
            names.extend([a.name for a in scene.agents])
        return names

    @property
    def num_instances(self):
        return self._num_scenes

    @property
    def total_num_scenes(self):
        return self._num_total_scenes

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    @property
    def horizon(self):
        return self._env_config.simulation.num_simulation_steps
    
    def get_gt_action(self, obs):
        ego = Action(
            positions=obs["ego"]["target_positions"],
            yaws=obs["ego"]["target_yaws"]
        ) if "ego" in obs else None
        agents = Action(
            positions=obs["agents"]["target_positions"],
            yaws=obs["agents"]["target_yaws"]
        ) if "agents" in obs else None
        return RolloutAction(ego=ego, agents=agents)

    def _disable_offroad_agents(self, scene, disable_parked=True):
        """Filter out parked and offroad agents"""
        obs = scene.get_obs(get_full_fut_traj=disable_parked)
        obs = parse_trajdata_batch(obs)
        drivable_region = get_drivable_region_map(obs["maps"])
        raster_pos = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        valid_agents = []

        if disable_parked:
            # Get edge and traffic light regions
            valid_indices_set = self._filter_parked_agents(obs,raster_pos)
        else:
            valid_indices_set = set(range(len(scene.agents)))

        # Filter out offroad agents
        for i, rpos in enumerate(raster_pos):
            if (i in valid_indices_set and 
                (scene.agents[i].name == "ego" or drivable_region[i, int(rpos[1]), int(rpos[0])].item() > 0)):
                valid_agents.append(scene.agents[i])

        scene.agents = valid_agents


    def _filter_parked_agents(self, obs,raster_pos, moving_speed_th = 0.3, disable_control_on_stationary= "any_speed"):
        ''' Filters out parked or stationary vehicles from the scene based on multiple criteria.
            This method identifies and removes parked vehicles by considering:
            1. Future movement patterns (using ground truth future trajectories)
            2. Proximity to road edges or traffic lights
            3. Lane association (whether vehicle is on a valid lane)
        '''
        has_lane = obs["extras"]["has_lane"]
        edge_region, traffic_light_region = self.get_cross_walk_ped_area(obs["maps"])

        # Define a speed threshold below which an agent is considered stationary
        full_fut_speed = torch.norm(obs["extras"]["full_fut_traj"][..., 3:5], dim=-1)
        full_fut_valid = obs["extras"]["full_fut_valid"]
        # Different stationary detection modes
        if 'any_speed' in disable_control_on_stationary:
            # Check if agent moves at any point in the future
            moving_mask = ((full_fut_speed > moving_speed_th).to(torch.float) * full_fut_valid).sum(dim=-1) > 0
        elif 'current_speed' in disable_control_on_stationary:
            # Only check current speed
            moving_mask = ((full_fut_speed[...,0] > moving_speed_th).to(torch.float) * full_fut_valid[...,0]) > 0
        else:
            moving_mask = torch.ones(*full_fut_valid.shape[:-1], dtype=torch.bool, device=full_fut_valid.device)
            
        stationary = ~moving_mask 
        #also use the lane information to filter out parked agents
        near_edge = []
        near_traffic_light = []

        batch_size = edge_region.shape[0]
        edge_neighborhood_size = 1
        traffic_light_neighborhood_size = 0  # Larger than edge_neighborhood_size
        #TODO, obtain traffic light or edge region from the map directlly not from raster map
        # Process each batch
        for i in range(batch_size):
            x, y = raster_pos[i].long()
            
            # Edge region checks
            x_min_edge, x_max_edge = max(0, x - edge_neighborhood_size), min(edge_region.shape[2] - 1, x + edge_neighborhood_size)
            y_min_edge, y_max_edge = max(0, y - edge_neighborhood_size), min(edge_region.shape[1] - 1, y + edge_neighborhood_size)
            near_edge.append(torch.any(edge_region[i, y_min_edge:y_max_edge + 1, x_min_edge:x_max_edge + 1]))

            # Traffic light region checks
            x_min_tl, x_max_tl = max(0, x - traffic_light_neighborhood_size), min(traffic_light_region.shape[2] - 1, x + traffic_light_neighborhood_size)
            y_min_tl, y_max_tl = max(0, y - traffic_light_neighborhood_size), min(traffic_light_region.shape[1] - 1, y + traffic_light_neighborhood_size)
            near_traffic_light.append(torch.any(traffic_light_region[i, y_min_tl:y_max_tl + 1, x_min_tl:x_max_tl + 1]))

        near_edge = torch.tensor(near_edge)
        near_traffic_light = torch.tensor(near_traffic_light)
        
        # Get centerline heading
        centerline_heading = obs["extras"]["centerline_heading"][:, 50:55]  # Using points, the first point of lane is 50 TODO:cchange to parameter
        
        # Check for off-road based on heading
        # is_offroad = self._is_offroad_by_heading(centerline_heading)
        
        # Update parked/invalid vehicle detection to include heading-based off-road check
        is_parked = (stationary & near_edge & ~near_traffic_light) | ~has_lane #| is_offroad
        
        # Use a mask to get valid agents (those that are not parked or off-road)
        valid_indices = torch.where(~is_parked)[0].tolist()
        
        return valid_indices
    
    @staticmethod
    def get_cross_walk_ped_area(map_tensor):
        # Prepare color tensors
        map_tensor = map_tensor*255
        dark_blue_min_t = torch.tensor([0.0, 0.0, 100.0], requires_grad=False).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        dark_blue_max_t = torch.tensor([50.0, 50.0, 255.0], requires_grad=False).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        mean_pink_rgb = torch.tensor([240, 61, 254], requires_grad=False)
        similar_pink_variance = torch.tensor(50, requires_grad=False)

        # Note: Using `clone().detach()` is redundant when you've just created the tensor and know it doesn't require a grad
        similar_pink_min_t = (mean_pink_rgb - similar_pink_variance).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        similar_pink_max_t = (mean_pink_rgb + similar_pink_variance).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                # Compute the masks using batch operations
        similar_pink_mask_t = torch.all((similar_pink_min_t <= map_tensor) & (map_tensor <= similar_pink_max_t), dim=1)
        dark_blue_mask_t = torch.all((dark_blue_min_t <= map_tensor) & (map_tensor <= dark_blue_max_t), dim=1)
       
        return dark_blue_mask_t, similar_pink_mask_t
                
        
    def add_new_agents(self,agent_data_by_scene):
        for sim_scene,agent_data in agent_data_by_scene.items():
            if sim_scene not in self._current_scenes:
                continue
            if len(agent_data)>0:
                sim_scene.add_new_agents(agent_data)
                sim_scene.initialize_ref_dict()
    def reset(self, scene_indices: List = None, start_frame_index = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        if scene_indices is None:
            # randomly sample a batch of scenes for close-loop rollouts
            all_indices = np.arange(self._num_total_scenes)
            scene_indices = self._npr.choice(
                all_indices, size=(self.num_instances,), replace=False
            )

        scene_info = [self.dataset.get_scene(i) for i in scene_indices]

        self._num_scenes = len(scene_info)
        self._current_scene_indices = scene_indices

        assert (
                np.max(scene_indices) < self._num_total_scenes
                and np.min(scene_indices) >= 0
        )
        if start_frame_index is None:
            start_frame_index = self._env_config.simulation.start_frame_index
        self._current_scenes = []
        filtered_scene_indices = []  # A list to store the indices of scenes we end up keeping

        self.ego_indices = []
        accumulate_count = 0
        scene_index_counter = {}

        for i, si in enumerate(scene_info):
            sim_scene: SimulationScene = SimulationScene(
                env_name=self._env_config.name,
                scene_name=si.name,
                scene=si,
                dataset=self.dataset,
                init_timestep=start_frame_index,
                freeze_agents=True,
                return_dict=True,
                # vectorize_lane=self.dataset.vectorize_lane,
            )
            obs = sim_scene.reset()
            num_agent = len(obs["agent_name"])
        
            # Only add the scene to _current_scenes if there are more than 1 agent
            if num_agent > 1:
                self._disable_offroad_agents(sim_scene)
                if len(sim_scene.agents)>1:
                    self.ego_indices.append(accumulate_count)
                    accumulate_count+=len(sim_scene.agents)
                    self._current_scenes.append(sim_scene)
                    # filtered_scene_indices.append(scene_indices[i])  # Store the index of the scene we are keeping
                     # Check if this scene_index has been encountered before
                    if scene_indices[i] in scene_index_counter:
                        # Increment the counter for this scene_index
                        scene_index_counter[scene_indices[i]] += 1
                    else:
                        # Initialize the counter for this new scene_index
                        scene_index_counter[scene_indices[i]] = 1
                    # Create a new, unique scene_index as an integer
                    unique_scene_index = int(str(scene_indices[i]) + str(scene_index_counter[scene_indices[i]]))

                    # Append the unique_scene_index to the list of new_scene_indices
                    filtered_scene_indices.append(unique_scene_index)

                
        self._current_scene_indices = filtered_scene_indices
        self._num_scenes = len(self._current_scenes)
         
        self._frame_index = 0
        self._cached_observation = None
        self._cached_raw_observation = None
        self._done = False

        obs_keys_to_log = [
            "centroid",
            "yaw",
            "curr_speed",
            "extent",
            "world_from_agent",
            "scene_index",
            "track_id",
            "drivable_map",
            "raster_from_world"
            
        ]
        self.logger = RolloutLogger(obs_keys=obs_keys_to_log)

        for v in self._metrics.values():
            v.reset()

    def render(self, actions_to_take):
        scene_ims = []
        ego_inds = self.ego_indices
        for i in ego_inds:
            im = render_state_trajdata(
                batch=self.get_observation()["agents"],
                batch_idx=i,
                action=actions_to_take
            )
            scene_ims.append(im)
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        agents = Action(
            positions=ac[:, :, :2],
            yaws=ac[:, :, 2:3]
        )

        return RolloutAction(agents=agents)

    def get_info(self):
        map_info = {scene.scene_name: f"{scene.cache.scene.env_name}:{scene.cache.scene.location}" for scene in self._current_scenes}
        #if there is ego and ctrl indices, we modify the names by scnene_ego_egoidx_ctrl_ctrl_idx
        if hasattr(self,"ego_indices") and hasattr(self, "ctrl_indices"):
            scene_count = defaultdict(int)
            scene_index_name = []

            for scene, ego_idx, ctrl_idx in zip(self.current_scene_names, self.ego_index_byscene, self.ctrl_indices):
                base_name = f"{scene}_ego_{ego_idx}_ctrl_{ctrl_idx}"
                suffix = scene_count[base_name]
                updated_name = f"{base_name}_{suffix}" if suffix > 0 else base_name
                scene_index_name.append(updated_name)
                # Increment the count for this base_name
                scene_count[base_name] += 1

        else:
            scene_index_name = self.current_scene_names
        info = dict(scene_index=scene_index_name, map_info=map_info)
        if self._log_data:
            sim_buffer = self.logger.get_serialized_scene_buffer()
            sim_buffer = [sim_buffer[k] for k in self.current_scene_index]
            info["buffer"] = sim_buffer
            # self.logger.get_trajectory()
        return info

    def get_multi_episode_metrics(self):
        metrics = dict()
        for met_name, met in self._metrics.items():
            met_vals = met.get_multi_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            elif met_vals is not None:
                metrics[met_name] = met_vals

        return TensorUtils.detach(metrics)

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """
        metrics = dict()
        # get ADE and FDE from SimulationScene
        metrics["ade"] = np.zeros(self.num_instances)
        metrics["fde"] = np.zeros(self.num_instances)
        for i, scene in enumerate(self._current_scenes):
            mets_per_agent = scene.get_metrics([sim_metrics.ADE(), sim_metrics.FDE()])
            metrics["ade"][i] = np.array(list(mets_per_agent["ade"].values())).mean()
            metrics["fde"][i] = np.array(list(mets_per_agent["fde"].values())).mean()

        # aggregate per-step metrics
        for met_name, met in self._metrics.items():
            met_vals = met.get_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            else:
                metrics[met_name] = met_vals

        for k in metrics:
            assert metrics[k].shape == (self.num_instances,)
        return TensorUtils.detach(metrics)

    def get_observation_by_scene(self):
        obs = self.get_observation()["agents"]
        obs_by_scene = []
        obs_scene_index = self.current_agent_scene_index
        for i in range(self.num_instances):
            obs_by_scene.append(TensorUtils.map_ndarray(obs, lambda x: x[obs_scene_index == i]))
        return obs_by_scene
        
    def get_observation(self,include_ego_obs=None):
        #dummy include_ego_os
        if self._cached_observation is not None:
            return self._cached_observation

        self.timers.tic("get_obs")

        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))
        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_trajdata_batch(agent_obs)
        agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
        agent_obs["scene_index"] = self.current_agent_scene_index
        agent_obs["track_id"] = self.current_agent_track_id
        agent_obs["env_name"] = [self._current_scenes[self.current_scene_index.index(i)].env_name for i in agent_obs["scene_index"]]
        

        # cache observations
        self._cached_observation = dict(agents=agent_obs)
        self.timers.toc("get_obs")

        return self._cached_observation


    def get_observation_skimp(self):
        self.timers.tic("obs_skimp")
        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False, get_map=False))
        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_trajdata_batch(agent_obs)
        agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
        agent_obs["scene_index"] = self.current_agent_scene_index
        agent_obs["track_id"] = self.current_agent_track_id
        self.timers.toc("obs_skimp")
        return dict(agents=agent_obs)

    def _add_per_step_metrics(self, obs):
        for k, v in self._metrics.items():
            v.add_step(obs, self.current_scene_index)

    def _step(self, step_actions: RolloutAction, num_steps_to_take):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")

        obs = self.get_observation()["agents"]   
       
        # record metrics
        # self._add_per_step_metrics(obs)

        action = step_actions.agents.to_dict()
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        self._add_per_step_metrics(obs)
        for action_index in range(num_steps_to_take):
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                return
            # # log state and action
            obs_skimp = self.get_observation_skimp()
            
            if self._log_data:
                action_to_log = RolloutAction(
                    agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                    agents_info=step_actions.agents_info
                )
                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    curr_yaw = obs["yaw"][idx]
                    curr_pos = obs["centroid"][idx]
                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                    if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index]@ world_from_agent + curr_pos
                        next_state[2] =  action["yaws"][idx, action_index, 0] + curr_yaw 

                        ## do smoothing?
                        # prev_xy = obs["agent_hist"][idx][:,:2]
                        # prev_h  = np.arctan2(obs["agent_hist"][idx][:,-2],obs["agent_hist"][idx][:,-1])
                        
                        # #get prev_vel, heading
                        # next_state = smooth_positions(np.hstack((prev_xy,prev_h[:,None])), next_state[None])                        ## do cubic spline and update the states
                        # next_state[:2] = next_state[:2] @ world_from_agent + curr_pos
                        # next_state[2] = next_state[2] + curr_yaw 
                        # if obs["extras"]["exceed_lane"][idx]:
                        #     next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                    else:
                        print("invalid action!")
                    scene_action[agent.name] = next_state
                    idx += 1
                StateArray_action = dict()
                for k,v in scene_action.items():
                    xyzh = np.insert(v,2,0.0)
                    StateArray_action[k] = StateArray.from_array(xyzh,"x,y,z,h")
                    
                scene.step(StateArray_action, return_obs=False)

        self._cached_observation = None

        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take
        print(self.timers)

    def step(self, actions: RolloutAction, num_steps_to_take: int = 1, render=False):
        """
        Step the simulation with control inputs

        Args:
            actions (RolloutAction): action for controlling ego and/or agents
            num_steps_to_take (int): how many env steps to take. Must be less or equal to length of the input actions
            render (bool): whether to render state and actions and return renderings
        """
        actions = actions.to_numpy()
        renderings = []
        if render:
            renderings.append(self.render(actions))
        self._step(step_actions=actions, num_steps_to_take=num_steps_to_take)
        return renderings
    
    def adjust_scene(self, adjust_plan):
        agent_data_by_scene = dict()
        for simscene in self._current_scenes:
            if simscene.scene.name in adjust_plan:
                adjust_plan_i = adjust_plan[simscene.scene.name]
                if adjust_plan_i["remove_existing_neighbors"]["flag"] and not adjust_plan_i["remove_existing_neighbors"]["executed"]:
                    simscene.agents = [agent for agent in simscene.agents if agent.name=="ego"]
                    adjust_plan_i["remove_existing_neighbors"]["executed"]=True
                agent_data=list()
                for agent_data_i in adjust_plan_i["agents"]:
                    if not agent_data_i["executed"]:
                        agent_data.append([agent_data_i["name"],
                                        np.array(agent_data_i["agent_state"]),
                                        agent_data_i["initial_timestep"],
                                        agent_types[agent_data_i["agent_type"]],
                                        np.array(agent_data_i["extent"]),
                                        ])
                        agent_data_i["executed"]=True
                agent_data_by_scene[simscene] = agent_data

        self.add_new_agents(agent_data_by_scene)
    
    def adjust_ego(self):
        self.ego_indices = np.where(np.array(self.batch_ego_indices) == 1)[0]
        self.other_agent_indices = np.where(np.array(self.batch_ego_indices) == 0)[0]
        
    def init_split_indices(self):
        ego_mask = [name=="ego" for name in self.current_agent_names]
        other_mask  = [name!="ego" for name in self.current_agent_names]
        self.ego_indices = np.where(np.array(ego_mask) == 1)[0]
        self.other_agent_indices = np.where(np.array(other_mask) == 1)[0]
        self.batch_ctrl_indices = None
        self.batch_ego_indices = None

        
class EnvSplitUnifiedSimulation(EnvUnifiedSimulation):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            log_data=True,
            split_ego = False,
            renderer=None,
            restart_track_id=True,
            parse_obs = True,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset, with the capability of spliting ego and agent observations

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
            split_ego (bool): if set to True, split ego out as the ego observation
            parse_obs (bool or dict): whether to parse the ego and agent observation or not
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = dataset.num_scenes()
        self._num_scenes = num_scenes
        self.split_ego = split_ego
        self.parse_obs = parse_obs

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes
        self._current_scene_indices = None

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self._cached_raw_observation = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._log_data = log_data
        self.restart_track_id = restart_track_id
        self.logger = None
        
       


    def reset(self, scene_indices: List = None, start_frame_index = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        super(EnvSplitUnifiedSimulation,self).reset(scene_indices,start_frame_index)
        self._cached_raw_observation = None

    def render(self, actions_to_take):
        scene_ims = []
        ego_inds = self.ego_indices
        if self.batch_ctrl_indices is not None:
            ctrl_inds = np.where(np.array(self.batch_ctrl_indices) == 1)[0]
        else:
            ctrl_inds = None  # or any other default value
      
        for i,batch_idx in enumerate(ego_inds):
            im = render_state_trajdata(
                batch=self.get_observation(split_ego=False)["agents"],
                batch_idx=batch_idx,
                ctrl_idx=ctrl_inds[i] if ctrl_inds is not None else None,
                i=i,
                action=actions_to_take
            )
            scene_ims.append(im)
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        if self.split_ego:
            ego_idx = self.ego_indices
            agent_idx = self.other_agent_indices
            ego_action = Action(
                positions=ac[ego_idx, :, :2],
                yaws=ac[ego_idx, :, 2:3]
            )
            agent_action = Action(
                positions=ac[agent_idx, :, :2],
                yaws=ac[agent_idx, :, 2:3]
            )
            return RolloutAction(ego=ego_action,agents=agent_action)
        else:
            agents = Action(
                positions=ac[:, :, :2],
                yaws=ac[:, :, 2:3]
            )

            return RolloutAction(agents=agents)


    def get_observation(self,split_ego=None,return_raw=False, include_ego_obs=False):
        if split_ego is None:
            split_ego = self.split_ego
        if return_raw:
            if self._cached_raw_observation is not None:
                return self._cached_raw_observation
        else:
            if self._cached_observation is not None:
                if split_ego and "ego" in self._cached_observation:
                    return self._cached_observation
                elif not split_ego and "ego" not in self._cached_observation:
                    return self._cached_observation
                else:
                    self._cached_observation = None
                    self._cached_raw_observation = None

        self.timers.tic("get_obs")

        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))
        self._cached_raw_observation = raw_obs
        if return_raw:
            return raw_obs
        if split_ego:
            # obtain index of ego and agents    
            ego_idx = self.ego_indices
            agent_idx = self.other_agent_indices
            # raw_obs is the raw trajdata batch_element object without collation
            ego_obs_raw = [raw_obs[idx] for idx in ego_idx]
            # call the collate function to turn batch_element into trajdata batch object
            ego_obs_collated = self.dataset.get_collate_fn(return_dict=True)(ego_obs_raw)
            agent_obs_raw = [raw_obs[idx] for idx in agent_idx]
            # call the collate function to turn batch_element into trajdata batch object
            if not include_ego_obs:
                if len(agent_obs_raw)>0:
                    agent_obs_collated = self.dataset.get_collate_fn(return_dict=True)(agent_obs_raw)
                else:
                    agent_obs_collated = dict()
                    for k,v in ego_obs_collated.items():
                        if isinstance(v,torch.Tensor):
                            if v.dim()>=1:
                                agent_obs_collated[k] = torch.zeros([0,*v.shape[1:]],dtype=v.dtype,device=v.device)
                            else:
                                agent_obs_collated[k] = torch.zeros(0,device=v.device)
                        elif isinstance(v,list):
                            agent_obs_collated[k] = []
                        else:
                            agent_obs_collated[k] = None
                
                # parse_obs can be True (parse both ego and agent), or False (parse neither), or dictionary that determines whether to parse ego or agent observation
            else:
                agent_obs_collated = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
                agent_obs = parse_trajdata_batch(agent_obs_collated)
                agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                agent_obs["scene_index"] = self.current_agent_scene_index
                agent_obs["track_id"] = self.current_agent_track_id
                agent_obs["ego_idx"] = ego_idx
            if self.parse_obs==True:
                parse_plan = dict(ego=True,agent=True)
            elif self.parse_obs==False:
                parse_plan = dict(ego=False,agent=False)
            elif isinstance(self.parse_obs,dict):
                parse_plan = self.parse_obs
            if parse_plan["ego"]:
                ego_obs = parse_trajdata_batch(ego_obs_collated)
                ego_obs = TensorUtils.to_numpy(ego_obs,ignore_if_unspecified=True)
                ego_obs["scene_index"] = self.current_agent_scene_index[ego_idx]
                ego_obs["track_id"] = self.current_agent_track_id[ego_idx]
                # ego_obs["env_name"] = [self._current_scenes[i].env_name for i in ego_obs["scene_index"]]
            else:
                # put collated observation into AgentBatch object from trajdata
                ego_obs = AgentBatch(**ego_obs_collated)
            if parse_plan["agent"]:
                if not include_ego_obs:
                    agent_obs = parse_trajdata_batch(agent_obs_collated)
                    agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                    if len(agent_idx)>0:
                        agent_obs["scene_index"] = self.current_agent_scene_index[agent_idx]
                        agent_obs["track_id"] = self.current_agent_track_id[agent_idx]
                    else:
                        agent_obs["scene_index"] = []
                        agent_obs["track_id"] = []
                else:
                    pass
                # agent_obs["env_name"] = [self._current_scenes[i].env_name for i in agent_obs["scene_index"]]
            else:
                # put collated observation into AgentBatch object from trajdata
                agent_obs = AgentBatch(**agent_obs_collated)
            self._cached_observation = dict(ego=ego_obs,agents=agent_obs)
        else:
            # if ego is not splitted out, then either parse all observa tion or do not parse any observation.
            assert isinstance(self.parse_obs,bool)
            agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
            if self.parse_obs:
                agent_obs = parse_trajdata_batch(agent_obs)
                agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                agent_obs["scene_index"] = self.current_agent_scene_index
                agent_obs["track_id"] = self.current_agent_track_id
                # agent_obs["env_name"] = [self._current_scenes[i].env_name for i in agent_obs["scene_index"]]
            else:
                agent_obs = AgentBatch(**agent_obs)
            self._cached_observation = dict(agents=agent_obs)

        self.timers.toc("get_obs")
        return self._cached_observation

    
    def combine_action(self,step_actions):
        # combine ego and agent actions
        if step_actions.agents is None:
            return RolloutAction(agents=step_actions.ego,agents_info=step_actions.ego_info)
        ego_action = step_actions.ego.to_dict()
        agent_action = step_actions.agents.to_dict()
        ego_idx = self.ego_indices
        agent_idx = self.other_agent_indices
        min_length = min(ego_action["positions"].shape[1],agent_action["positions"].shape[1])
        combined_positions = np.zeros([len(self.current_agent_names),min_length,2])
        combined_yaws = np.zeros([len(self.current_agent_names),min_length,1])
        combined_positions[ego_idx] = ego_action["positions"][:,:min_length]
        combined_positions[agent_idx] = agent_action["positions"][:,:min_length]
        combined_yaws[ego_idx] = ego_action["yaws"][:,:min_length]
        combined_yaws[agent_idx] = agent_action["yaws"][:,:min_length]
        return RolloutAction(agents=Action(positions=combined_positions,yaws=combined_yaws),agents_info=step_actions.agents_info)


    def combine_obs(self,ego_obs,agent_obs):
        # combining ego and agent observation, not really used.
        ego_idx = self.ego_indices
        agent_idx = self.other_agent_indices
        bs = len(self.current_agent_names)
        combined_obs = dict()
        for k,v in ego_obs.items():
            if k in agent_obs and v is not None:
                combined_v = np.zeros([bs,*v.shape[1:]])
                combined_v[ego_idx]=ego_obs[k]
                combined_v[agent_idx]=agent_obs[k]
                combined_obs[k]=combined_v
        return combined_obs
    def get_observation_skimp(self,split_ego=True):
        self.timers.tic("obs_skimp")
        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False, get_map=True))
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id

        
        self.timers.toc("obs_skimp")
        keys_to_take = ['agent_type', 'image', 'drivable_map', 'target_positions',\
             'target_yaws', 'target_availabilities', 'history_positions', 'history_yaws', 'history_availabilities',\
             'curr_speed', 'centroid', 'yaw', 'type', 'extent', 'raster_from_agent', 'agent_from_raster',\
             'raster_from_world', 'agent_from_world', 'world_from_agent', 'scene_index', 'track_id', "extras"]
        obs_selected = {k:v for k,v in obs.items() if k in keys_to_take}
        if split_ego:
            ego_mask = self.ego_indices
            agents_mask =  self.other_agent_indices
            ego_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[ego_mask])
            agents_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[agents_mask])
            
            return dict(ego=ego_obs,agents=agents_obs)
        else:
            return dict(agents=obs_selected)
    def _add_per_step_metrics(self, obs):

        ego_mask = self.ego_indices
        agents_mask =  self.other_agent_indices
        keys_to_take = ['agent_type', 'image', 'drivable_map', 'target_positions',\
             'target_yaws', 'target_availabilities', 'history_positions', 'history_yaws', 'history_availabilities',\
             'curr_speed', 'centroid', 'yaw', 'type', 'extent', 'raster_from_agent', 'agent_from_raster',\
             'raster_from_world', 'agent_from_world', 'world_from_agent', 'scene_index', 'track_id']
        obs_selected = {k:v for k,v in obs.items() if k in keys_to_take}
        ego_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[ego_mask])
        agents_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[agents_mask])
        ego_map_names = [obs["map_names"][i] for i in self.ego_indices]
        agents_map_names = [obs["map_names"][i] for i in self.other_agent_indices]
        ego_obs["map_names"] = ego_map_names
        agents_obs["map_names"] = agents_map_names
        for k, v in self._metrics.items():
            if k.startswith("ego"):
                v.add_step(ego_obs, self._current_scene_indices)
                if "failure" in k or "collision" in k:
                    v.add_step(ego_obs, self._current_scene_indices, agents_obs)
                ##Add the ego influence on other agent metrics here
            elif k.startswith("agents"):
                v.add_step(obs, self._current_scene_indices)
            elif k.startswith("all"):
                v.add_step(obs, self._current_scene_indices)
            else:
                raise KeyError("Invalid metrics name {}".format(k))

    def _step(self, step_actions: RolloutAction, num_steps_to_take):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")
        self.timers.tic("_step")
        # to bypass all the ego split, collation and parsing, directly get raw obs
        raw_obs = self.get_observation(split_ego=False,return_raw=True)
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        # always parse when stepping
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id
        # obs = {k:v for k,v in obs.items() if not isinstance(v,list)}
        obs["batch_ctrl_indices"] = self.batch_ctrl_indices
        obs["batch_ego_indices"] = self.batch_ego_indices
        # record metrics
        self._add_per_step_metrics(obs)
        if step_actions.has_ego:
            combined_step_actions = self.combine_action(step_actions)
            action = combined_step_actions.agents.to_dict()
        else:
            action = step_actions.agents.to_dict()
        
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        for action_index in range(num_steps_to_take):
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                self._cached_raw_observation = None
                return
           
            # self._add_per_step_metrics(obs_skimp["agents"])
            if self._log_data:
                 # # log state and action
                obs_skimp = self.get_observation_skimp(split_ego=True)
                if step_actions.has_ego:
                    ego_idx = self.ego_indices
                    agent_idx = self.other_agent_indices
                    action_t = TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(dict(positions=action_t["positions"][agent_idx],yaws=action_t["yaws"][agent_idx])) if len(agent_idx)>0 else None,
                        agents_info=step_actions.agents_info if len(agent_idx)>0 else None,
                        ego = Action.from_dict(dict(positions=action_t["positions"][ego_idx],yaws=action_t["yaws"][ego_idx])),
                        ego_info=step_actions.ego_info,
                    )
                else:
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                        agents_info=step_actions.agents_info
                    )

                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    ## This can be optimized as batch in the future
                    curr_yaw = obs["yaw"][idx]
                    curr_pos = obs["centroid"][idx]
                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                    if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                        next_state[2] = curr_yaw + action["yaws"][idx, action_index, 0]
                    else:
                        print("invalid action!")
                    scene_action[agent.name] = next_state
                    idx += 1
                   
                StateArray_action = dict()
                for k,v in scene_action.items():
                    xyzh = np.insert(v,2,0.0)
                    StateArray_action[k] = StateArray.from_array(xyzh,"x,y,z,h")
                scene.step(StateArray_action, return_obs=False)

        self._cached_observation = None
        self._cached_raw_observation = None
        self.timers.toc("_step")
        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take
        print(self.timers)

    def save_relationships_to_file(self):
        scene_relation_dict = {}
        for simscene in self._current_scenes.copy():
            scene_key = simscene.scene_name  # Assuming the scene can be uniquely identified and converted to string
            obs = simscene.get_obs()

            #save index_scene name map
            num_agent = len(obs["agent_name"])
            cl_relationship, interaction = determine_centerline_relationship(obs)
            scene_data = {
                "relationships": cl_relationship,
                "interactions": interaction,
                "ref_polyline_ids": obs["extras"]["ref_polyline_ids"]
            }

            # Add the sub-dictionary to the main dictionary
            scene_relation_dict[scene_key] = scene_data
        with open(f"scene_agent_relation/0213_new_relationships_interactions.pkl", "wb") as f:
            pickle.dump(scene_relation_dict, f)
        
        for relationship in ["intersection","merging"]:
            for interactions in ["equal","car2_behind_car1","car1_behind_car2"]:
                filtered_data = filter_scenes(scene_relation_dict, relationship, interactions)
                save_filtered_data_as_json(filtered_data, relationship, interactions) 
        
        #For sanity check
        filtered_data = filter_scenes(scene_relation_dict, "intersection", "equal")
        return filtered_data
    
    
    def init_ego_and_target_agents(self, init_recipe):
        """Initialize ego and target agents for each scene based on the provided recipe.
        
        This method handles multiple instances of the same scene with different ego and control agent
        combinations. For example, the same scene can be used multiple times with different vehicles
        designated as ego and control agents.

        The method uses two tracking mechanisms:
        1. scene_index_map: Tracks which variant of a scene we're currently processing
        2. scene_occurrence_map: Tracks how many times we've used a specific ego-control combination

        Args:
            init_recipe (dict): Configuration dictionary containing:
                - predefined_scene_init: Scene-specific initialization data
                - n_sample_scene: Number of times to sample each ego-control combination (default: 1)

        Example init_recipe structure:
        {
            "predefined_scene_init": {
                "scene-0001": {
                    "indices": [
                        {"ego_idx": 0, "adv_indices": [1]},  # First variant
                        {"ego_idx": 2, "adv_indices": [3]}   # Second variant
                    ],
                    "ref_polyline_ids": [...],
                    "all_ref_polyline_ids": [...]
                }
            },
        }
        """
        #include ref_lane information in the future, 
        
        self.batch_ctrl_indices = []
        self.batch_ego_indices = []
        self.ego_index_byscene = []
        self.ctrl_indices = []
    
        for simscene in self._current_scenes.copy():
            scene_key = simscene.scene_name  # Assuming the scene can be uniquely identified and converted to string
            scene_index = self.scene_index_map[scene_key]
            
            ## given agent_names obtain the ego and ctrl indices
            
            #load vec_map
            predefined_init_dictionary = init_recipe["predefined_scene_init"]
            if predefined_init_dictionary and scene_key in predefined_init_dictionary:
                obs = simscene.get_obs()
                num_agent = len(obs["agent_name"])
                agent_names = obs["agent_name"]  # Get the list of agent names in the scene

                # Check if we're using name-based format
                if "ego_name" in predefined_init_dictionary[scene_key] and "adv_names" in predefined_init_dictionary[scene_key]:
                    # Get names from configuration
                    ego_name = predefined_init_dictionary[scene_key]["ego_name"]
                    adv_names = predefined_init_dictionary[scene_key]["adv_names"]
                    
                    # Map names to indices
                    try:
                        ego_idx = agent_names.index(ego_name)
                    except ValueError:
                        print(f"Warning: Ego agent '{ego_name}' not found in scene {scene_key}")
                        ego_idx = 0  # Default to first agent
                    
                    # Map adversary names to indices
                    ctrl_target_indices = []
                    for name in adv_names:
                        try:
                            idx = agent_names.index(name)
                            ctrl_target_indices.append(idx)
                        except ValueError:
                            print(f"Warning: Adversary agent '{name}' not found in scene {scene_key}")
                else:
                    # Fall back to the original index-based format
                    ctrl_target_indices = predefined_init_dictionary[scene_key]["adv_indices"]
                    ego_idx = predefined_init_dictionary[scene_key]["ego_idx"]

                binary_ctrl_targets = [0] * num_agent
                for idx in ctrl_target_indices:
                    binary_ctrl_targets[idx] = 1
                binary_ego_idices = [0] * num_agent
                binary_ego_idices[ego_idx] = 1
                
                simscene.ctrl_target_indices = ctrl_target_indices 
                simscene.ego_idx = ego_idx
                self.batch_ctrl_indices.extend(binary_ctrl_targets)
                self.batch_ego_indices.extend(binary_ego_idices)
                self.ctrl_indices.append(ctrl_target_indices)
                self.ego_index_byscene.append(ego_idx) 
                