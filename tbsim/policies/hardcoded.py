import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from tbsim.utils.l5_utils import get_current_states
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.lane_utils as LaneUtils
from tbsim.utils.geometry_utils import calc_distance_map,transform_points_tensor

from tbsim.utils.timer import Timers
from tbsim.policies.common import Action, Plan
from tbsim.policies.base import Policy
#strive related
# from tbsim.policies.strive_planner import  load_lane_graphs, HardcodeNuscPlanner, DEF_CONFIG, PlannerConfig
from tbsim.policies.strive_planner_trajdata import  load_lane_graphs_trajdata, HardcodeNuscPlanner, DEF_CONFIG, PlannerConfig

try:
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.Sampling.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")

import tbsim.utils.planning_utils as PlanUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.timer import Timers
from pathlib import Path
from trajdata import MapAPI

from trajdata.data_structures.state import StateArray,StateTensor


class OptimController(Policy):
    """An optimization-based controller"""

    def __init__(
            self,
            dynamics_type,
            dynamics_kwargs,
            step_time: float,
            optimizer_kwargs=None,
    ):
        self.step_time = step_time
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        if dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=dynamics_kwargs["max_steer"],
                max_yawvel=dynamics_kwargs["max_yawvel"],
                acce_bound=dynamics_kwargs["acce_bound"],
            )
        elif dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=dynamics_kwargs["acce_bound"],
                ddh_bound=dynamics_kwargs["ddh_bound"],
                max_hdot=dynamics_kwargs["max_yawvel"],
                max_speed=dynamics_kwargs["max_speed"],
            )
        else:
            raise NotImplementedError(
                "dynamics type {} is not implemented", dynamics_type
            )

    def eval(self):
        pass

    def get_action(self, obs, plan: Plan, init_u=None, **kwargs) -> Tuple[Action, Dict]:
        target_pos = plan.positions
        target_yaw = plan.yaws
        target_avails = plan.availabilities
        device = target_pos.device
        num_action_steps = target_pos.shape[-2]
        init_x = get_current_states(obs, dyn_type=self.dyn.type())
        if init_u is None:
            init_u = torch.randn(
                *init_x.shape[:-1], num_action_steps, self.dyn.udim
            ).to(device)
        if target_avails is None:
            target_avails = torch.ones(target_pos.shape[:-1]).to(device)
        targets = torch.cat((target_pos, target_yaw), dim=-1)
        assert init_u.shape[-2] == num_action_steps
        predictions, raw_traj, final_u, losses = optimize_trajectories(
            init_u=init_u,
            init_x=init_x,
            target_trajs=targets,
            target_avails=target_avails,
            dynamics_model=self.dyn,
            step_time=self.step_time,
            data_batch=obs,
            **self.optimizer_kwargs
        )
        action = Action(**predictions)
        return action, {}


class GTPolicy(Policy):
    def __init__(self, device):
        super(GTPolicy, self).__init__(device)

    def eval(self):
        pass

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        action = Action(
            positions=TensorUtils.to_torch(obs["target_positions"], device=self.device),
            yaws=TensorUtils.to_torch(obs["target_yaws"], device=self.device),
        )
        return action, {}

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        plan = Plan(
            positions=TensorUtils.to_torch(obs["target_positions"], device=self.device),
            yaws=TensorUtils.to_torch(obs["target_yaws"], device=self.device),
            availabilities=TensorUtils.to_torch(obs["target_availabilities"], self.device),
        )
        
        return plan, {}


class ReplayPolicy(Policy):
    def __init__(self, action_log, device):
        super(ReplayPolicy, self).__init__(device)
        self.action_log = action_log

    def eval(self):
        pass

    def get_action(self, obs, step_index=None, **kwargs) -> Tuple[Action, Dict]:
        assert step_index is not None
        scene_index = TensorUtils.to_numpy(obs["scene_index"]).astype(np.int64).tolist()
        track_id = TensorUtils.to_numpy(obs["track_id"]).astype(np.int64).tolist()
        pos = []
        yaw = []
        for si, ti in zip(scene_index, track_id):
            scene_log = self.action_log[str(si)]
            if ti == -1:  # ego
                pos.append(scene_log["ego_action"]["positions"][step_index, 0])
                yaw.append(scene_log["ego_action"]["yaws"][step_index, 0])
            else:
                scene_track_id = scene_log["agents_obs"]["track_id"][0]
                agent_ind = np.where(ti == scene_track_id)[0][0]
                pos.append(scene_log["agents_action"]["positions"][step_index, agent_ind])
                yaw.append(scene_log["agents_action"]["yaws"][step_index, agent_ind])

        # stack and create the temporal dimension
        pos = np.stack(pos, axis=0)[:, None, :]
        yaw = np.stack(yaw, axis=0)[:, None, :]

        action = Action(
            positions=pos,
            yaws=yaw
        )
        return action, {}


class EC_sampling_controller(Policy):
    def __init__(self,ego_sampler,EC_model,agent_planner=None,device="cpu"):
        self.ego_sampler = ego_sampler
        self.EC_model = EC_model
        self.agent_planner = agent_planner
        self.device = device
        self.timer = Timers()
    
    def eval(self):
        self.EC_model.eval()
        if self.agent_planner is not None:
            self.agent_planner.eval()
    
    def get_action(self,obs,**kwargs)-> Tuple[Action, Dict]:
        assert "agent_obs" in kwargs
        agent_obs = kwargs["agent_obs"]
        #TODO: prediction without GC
        if self.agent_planner is not None:
            agent_plan = self.agent_planner(agent_obs)
            agent_plan = torch.cat((agent_plan["predictions"]["positions"],agent_plan["predictions"]["yaws"]),-1).squeeze(1)
        else:
            agent_plan=None
        self.timer.tic("sampling")
        tf = 5
        T = obs["target_positions"].shape[-2]
        bs = obs["history_positions"].shape[0]
        agent_size = agent_obs["image"].shape[0]
        ego_trajs = list()
        #TODO: paralellize this process
        for i in range(bs):
            vel = obs["curr_speed"][i]
            traj0 = torch.tensor([[0., 0., vel, 0, 0., 0., 0.]]).to(vel.device)
            lanes = TensorUtils.to_numpy(obs["ego_lanes"][i])
            lanes = np.concatenate((lanes[...,:2],np.arctan2(lanes[...,3:],lanes[...,2:3])),-1)
            lanes = np.split(lanes,obs["ego_lanes"].shape[1])
            lanes = [lane[0] for lane in lanes if not (lane==0).all()]
            
            def expand_func(x): return self.ego_sampler.gen_trajectory_batch(
                    x, tf, lanes)
            x0 = TrajTree(traj0, None, 0)
            x0.grow_tree(expand_func, 1)
            leaves = x0.get_all_leaves()
            
            if len(leaves) > 0:
                ego_trajs_i = torch.stack([leaf.total_traj for leaf in leaves], 0)
                ego_trajs_i = ego_trajs_i[...,1:,[0,1,4]]
            else:
                ego_trajs_i = torch.cat((obs["target_positions"][i],obs["target_yaws"][i]),-1).unsqueeze(0)
           
            ego_trajs.append(ego_trajs_i)

        self.timer.toc("sampling")
        self.timer.tic("prediction")
        N = max(ego_trajs_i.shape[0] for ego_trajs_i in ego_trajs)
        cond_traj = torch.zeros([agent_size,N,T,3]).to(obs["speed"].device)
        ego_traj_samples = torch.zeros([bs,N,T,3])
        for i in range(bs):
            ego_traj_samples[i,:ego_trajs[i].shape[0]] = ego_trajs[i]
            agent_idx = torch.where(agent_obs["scene_index"]==obs["scene_index"][i])[0]
            ego_pos_world = GeoUtils.batch_nd_transform_points(ego_trajs[i][...,:2],obs["world_from_agent"][i].unsqueeze(0))
            ego_pos_agent = GeoUtils.batch_nd_transform_points(
                ego_pos_world.tile(agent_idx.shape[0],1,1,1),agent_obs["agent_from_world"][agent_idx,None,None]
                )

            ego_yaw_agent = (ego_trajs[i][...,2:]+obs["yaw"][i]).tile(agent_idx.shape[0],1,1,1)-agent_obs["yaw"][agent_idx].reshape(-1,1,1,1)
            cond_traj[agent_idx,:ego_trajs[i].shape[0]] = torch.cat((ego_pos_agent,ego_yaw_agent),-1)

        EC_pred = self.EC_model.get_EC_pred(agent_obs,cond_traj,agent_plan)
        self.timer.toc("prediction")
        self.timer.tic("planning")
        if "drivable_map" in obs:
            drivable_map = obs["drivable_map"].float()
        else:
            drivable_map = batch_utils().get_drivable_region_map(obs["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        
        opt_traj = list()
        for i in range(bs):
            agent_idx = torch.where(agent_obs["scene_index"]==obs["scene_index"][i])[0]
            N_i = ego_trajs[i].shape[0]

            agent_pos_world = GeoUtils.batch_nd_transform_points(EC_pred["EC_trajectories"][agent_idx,:N_i,...,:2],agent_obs["world_from_agent"][agent_idx,None,None])
            agent_pos_ego = GeoUtils.batch_nd_transform_points(agent_pos_world,obs["agent_from_world"][i].unsqueeze(0))
            agent_yaw_ego =EC_pred["EC_trajectories"][agent_idx,:N_i,...,2:]+agent_obs["yaw"][agent_idx].reshape(-1,1,1,1)-obs["yaw"][i]
            agent_traj = torch.cat((agent_pos_ego,agent_yaw_ego),-1)

            idx = PlanUtils.ego_sample_planning(
                ego_trajs[i].unsqueeze(0),
                agent_traj.transpose(0,1).unsqueeze(0),
                obs["extent"][i:i+1, :2],
                agent_obs["extent"][None,agent_idx,:2],
                agent_obs["type"][None,agent_idx],
                obs["raster_from_world"][i].unsqueeze(0),                
                dis_map[i].unsqueeze(0),
                weights={"collision_weight": 1.0, "lane_weight": 1.0,"likelihood_weight":0.0,"progress_weight":0.0},
            )[0]

            opt_traj.append(ego_trajs[i][idx])
        self.timer.toc("planning")
        print(self.timer)
        opt_traj = torch.stack(opt_traj,0)
        action = Action(positions=opt_traj[...,:2],yaws=opt_traj[...,2:])
        action_info = dict()
        action_info["action_samples"] = {"positions":ego_traj_samples[...,:2],"yaws":ego_traj_samples[...,2:]}
        return action, action_info
        

class HierSplineSamplingPolicy(Policy):
    def __init__(self, device, step_time, predictor,ego_sampler=None, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.device = device
        if ego_sampler is not None:
            self.ego_sampler = ego_sampler
        else:
            self.ego_sampler = self.get_ego_sampler()
        self.step_time = step_time
        self.predictor = predictor
    
    def get_ego_sampler(self):
        ego_sampler = SplinePlanner(self.device)
        return ego_sampler
    def eval(self):
        self.predictor.eval()
    
    def get_action(self, obs_dict, **kwargs):
        bs,horizon = obs_dict["target_positions"].shape[:2]
        horizon = 20
        ego_obs_dict = obs_dict
        # obs_dict = kwargs["agent_obs"]
        # agent_obs_dict = kwargs["agent_obs"]
        agent_preds, _ = self.predictor.get_prediction(obs_dict)
        ego_trajs = list()
        #TODO: paralellize this process
        for i in range(bs):
            vel = obs_dict["curr_speed"][i]

            traj0 = torch.tensor([[0., 0., vel, 0, 0., 0., 0.]]).to(vel.device)
            lanes = TensorUtils.to_numpy(obs_dict["extras"]["centerline_xy"][i])
            if lanes.shape[1] == 2:  # When lanes contain only positions
                # Calculate the differences between consecutive points
                d_lanes = np.diff(lanes, axis=0)
                
                # Compute the arctan2 to get heading
                heading = np.arctan2(d_lanes[:, 1], d_lanes[:, 0]).reshape(-1, 1)
                
                # Adding an extra heading to match the shape, this could be set to any value or interpolated
                heading = np.concatenate([heading, heading[-1:]], axis=0)
                
                # Now concatenate the original lanes (positions) with the new heading
                lanes = np.concatenate((lanes, heading), axis=-1)[None]
                # lanes = np.split(lanes,obs_dict["extras"]["centerline_xy"].shape[1])

            elif lanes.shape[1] > 2:  # The case when lanes have more than 2 columns
                lanes = np.concatenate((lanes[...,:2], np.arctan2(lanes[...,3:], lanes[...,2:3])), axis=-1)


                lanes = np.split(lanes,obs_dict["extras"]["centerline_xy"].shape[1])
                lanes = [lane[0] for lane in lanes if not (lane==0).all()]
            # lanes = None
            def expand_func(x): return self.ego_sampler.gen_trajectory_batch(
                    x, self.step_time*horizon, lanes,N=horizon+1)
            x0 = TrajTree(traj0, None, 0)
            x0.grow_tree(expand_func, 1)
            nodes,_ = TrajTree.get_nodes_by_level(x0,depth=1)
            leaves = nodes[1]
            
            if len(leaves) > 0:
                ego_trajs_i = torch.stack([leaf.total_traj for leaf in leaves], 0)
                ego_trajs_i = ego_trajs_i[...,1:,[0,1,4]]
            else:
                ego_trajs_i = torch.cat((obs_dict["target_positions"][i],obs_dict["target_yaws"][i]),-1).unsqueeze(0)
            ego_trajs.append(ego_trajs_i)
            # plot_trajectories(ego_trajs_i)
        if "drivable_map" in obs_dict:
            drivable_map = obs_dict["drivable_map"].float()
        else:    
            drivable_map = batch_utils().get_drivable_region_map(obs_dict["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        plan = list()
        for i in range(bs):
            
            agent_idx = torch.where(obs_dict["all_other_agents_types"][i]>0)[0]
            agent_extent = torch.max(obs_dict["all_other_agents_history_extents"][i,agent_idx,:,:2],axis=-2)[0]
            agent_traj = torch.cat([agent_preds.positions[i,agent_idx],agent_preds.yaws[i,agent_idx]],-1) #agent_traj of each batch
            idx = PlanUtils.ego_sample_planning(
                ego_trajs[i].unsqueeze(0),
                agent_traj.unsqueeze(0),
                obs_dict["extent"][i:i+1, :2],
                agent_extent.unsqueeze(0),
                obs_dict["all_other_agents_types"][i:i+1,agent_idx],
                obs_dict["raster_from_world"][i].unsqueeze(0),                
                dis_map[i].unsqueeze(0),
                weights={"collision_weight": 10.0, "lane_weight": 1.0, "progress_weight": 0.3,"likelihood_weight": 0.20},
            )[0]
            plan.append(ego_trajs[i][idx])
        plan = torch.stack(plan,0)
        action = Action(positions=plan[...,:2],yaws=plan[...,2:])
        return action, {}

class IDMPlanner(Policy):
    def __init__(self, device,dt=0.1):
        super(IDMPlanner, self).__init__(device)
        # IDM Parameters
        self.a = 1.0  # Maximum acceleration [m/s^2]
        self.b = 3.0  # Comfortable braking deceleration [m/s^2]
        self.delta_t = dt  # timestep
        self.s_0 = 5.0  # Minimum jam distance [m]
        self.T = 1.0  # Safe time headway [s]
        self.v_0 = 10.0  # Desired velocity [m/s or ~80 km/h]
        self.convert_to_tensor(self)
    def eval(self):
        pass

    @staticmethod
    def convert_to_tensor(obj):
        """Convert all float attributes of an object to PyTorch tensors."""
        for attr, value in vars(obj).items():
            if isinstance(value, (float, int)):
                setattr(obj, attr, torch.tensor(value))
    def idm_acceleration(self, ego_velocity, delta_v, s):
        """Calculate the IDM acceleration."""
        s_star = self.s_0 + torch.clamp_min(ego_velocity*self.T + (ego_velocity*delta_v)/(2*torch.sqrt(self.a*self.b)), 0)
        return self.a * (1 - torch.pow(ego_velocity/self.v_0, 4) - torch.pow(s_star/s[:,None], 2))

    def get_action(self, obs_dict, **kwargs):
        ego_position = obs_dict["agent_hist"][...,-1,:2]
        ego_velocity = obs_dict["agent_hist"][...,-1,3:5]
        ego_heading = obs_dict["agent_hist"][...,-1,-2:]
        ego_heading = ego_heading[..., [1, 0]]

        centerline_differences = obs_dict["extras"]["centerline_xy"][:, 1:, :] - obs_dict["extras"]["centerline_xy"][:, :-1, :]
        centerline_headings = torch.atan2(centerline_differences[..., 1], centerline_differences[..., 0])
        centerline_headings = torch.cat((centerline_headings, centerline_headings[:, -1].unsqueeze(1)), dim=1)

        agent_position = obs_dict["neigh_hist"][...,-1,:2]
        agent_velocity = obs_dict["neigh_hist"][...,-1,3:5]
        agent_heading = torch.atan2(obs_dict["neigh_hist"][..., [-2]], obs_dict["neigh_hist"][..., [-1]])[...,-1,:]
        agent_mask = obs_dict["neigh_types"] !=0

        agent_traj = torch.cat([agent_position, agent_heading], dim=-1)  
        centerline_traj = torch.cat([obs_dict["extras"]["centerline_xy"], centerline_headings.unsqueeze(-1)], dim=-1)

        B, A, _ = agent_traj.shape
        agent_traj_reshaped = agent_traj.reshape(-1, 3)
        centerline_repeated = centerline_traj.repeat(1, A, 1, 1).reshape(-1, 150, 3)

        tan_dist, norm_dist, psi = GeoUtils.batch_proj(agent_traj_reshaped, centerline_repeated)
        _, norm_dist_indices = norm_dist.reshape(B, A, -1).min(dim=-1)

        # Filter based on norm_dist range
        NORM_DIST_THRESHOLD = 5.0  
        valid_projection_mask = (norm_dist < NORM_DIST_THRESHOLD) & agent_mask.reshape(B*A,1)
        valid_projection_mask = valid_projection_mask.reshape(B, A, 150)

        if not torch.any(valid_projection_mask):
            # Handle this case.
            pass
        # Use norm_dist_indices to filter valid_projection_mask
        point_valid_mask = torch.gather(valid_projection_mask, 2, norm_dist_indices.unsqueeze(-1))
        point_valid_mask = point_valid_mask.squeeze(-1)  # shape will now be [B, A]

        # Get the point on the centerline
        point_on_centerline = torch.gather(obs_dict["extras"]["centerline_xy"], 1, norm_dist_indices.unsqueeze(-1).expand(-1, -1, 2))

        # Check if that point is in front of the ego vehicle
        diff = point_on_centerline - ego_position[:, None]
        in_front_mask = torch.einsum('bij,bj->bi', diff, ego_heading) > 0

        # Combine the two masks: points with valid projections and those in front
        combined_mask = point_valid_mask & in_front_mask
        # Default acceleration
        accel = self.a * (1 - torch.pow(ego_velocity / self.v_0, 4))
        if torch.any(combined_mask):
            filtered_diff = torch.where(combined_mask.unsqueeze(-1), diff, float('inf'))
            
            # Calculate distances using the filtered differences
            distances_in_front = torch.norm(filtered_diff, dim=-1)
            
            # Identify the closest vehicle in front of the ego vehicle
            min_distance, min_distance_idx = torch.min(distances_in_front, dim=1)
            gathered_velocity = torch.gather(agent_velocity, 1, min_distance_idx[:,None,None].expand(B, 1, 2)).squeeze()
            # Get relative speed of the closest vehicle in front
            relative_speed = gathered_velocity - ego_velocity

             # Calculate IDM acceleration
            idm_accel = self.idm_acceleration(ego_velocity, relative_speed, min_distance)


            # Use IDM acceleration where combined_mask is true, and default elsewhere
            combined_mask_expanded = torch.gather(combined_mask,1,min_distance_idx[:,None]).expand_as(accel)
            accel = torch.where(combined_mask_expanded, idm_accel, accel)
        
        new_position, new_yaw = self.acc_follow_centerline2traj(
            centerline_traj,
            ego_velocity,
            accel,
            self.delta_t,
            horizon = 32
            )
        
        # self.plot_scenario(ego_position,centerline_traj,agent_position,new_position)
        plan = torch.cat([new_position, new_yaw[...,None]], dim=-1)
        action = Action(positions=plan[..., :2], yaws=plan[..., 2:])

        return action, {}
    
    @staticmethod
    def acc_follow_centerline2traj(centerline_traj, current_velocity, accel, delta_t, horizon):
        
        #TODO velocity in two dimension, add bicycle model tracking?
        B = centerline_traj.shape[0]
        
        # Initialization
        positions = torch.zeros([B, horizon, 2], dtype=torch.float32).to(accel.device)
        yaws = torch.zeros([B, horizon], dtype=torch.float32).to(accel.device)
        
        # 1. Find the closest centerline point to the ego car
        distances_to_origin = torch.norm(centerline_traj[:, :, :2], dim=-1)
        _, start_indices = torch.min(distances_to_origin, dim=1)
        
        # Calculate velocities for the entire horizon
        time_stamps = torch.arange(horizon, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(accel.device)
        # velocities = current_velocity.unsqueeze(1) + accel.unsqueeze(1) * delta_t * time_stamps
        velocities = torch.clamp(current_velocity.unsqueeze(1) + accel.unsqueeze(1) * delta_t * time_stamps, min=0)

        # Calculate differences and segment lengths
        diffs = centerline_traj[:, 1:, :2] - centerline_traj[:, :-1, :2]
        segment_lengths = torch.norm(diffs, dim=-1)
        
        # Create a mask with ones from start_idx onwards and zeros before that, for each batch item.
        mask = torch.arange(segment_lengths.shape[1]).to(segment_lengths.device).expand(segment_lengths.shape)
        mask = (mask >= start_indices.unsqueeze(-1)).float()

        # Calculate cumulative lengths for each batch item
        cum_lengths_from_start = mask * torch.cumsum(segment_lengths * mask, dim=1)
        # Expected distance covered by ego vehicle over the horizon
        distances = torch.cumsum(velocities[:, :, 0] * delta_t, dim=1)
        
        # Map these distances onto the centerline
        segment_indices = (cum_lengths_from_start.unsqueeze(1) <= distances.unsqueeze(2)).sum(dim=-1) - 1
        segment_indices = torch.clamp(segment_indices, 0, segment_lengths.shape[1]-1)

        # Create a mask where the segment index is the last one
        last_idx_mask = (segment_indices == segment_lengths.shape[1]-1).unsqueeze(-1)
        
        # Mask to avoid cumsum across the horizon where last point is reached
        cumsum_mask = 1 - torch.cumsum(last_idx_mask, dim=1)
        
        # Retrieve the yaw values of the corresponding segments
        selected_yaws = torch.gather(centerline_traj[:, :-1, 2], 1, segment_indices)
        dx = velocities[:, :, 0] * delta_t * torch.cos(selected_yaws)
        dy = velocities[:, :, 0] * delta_t * torch.sin(selected_yaws)

        # Apply the cumsum mask
        dx *= cumsum_mask.squeeze()
        dy *= cumsum_mask.squeeze()

        # Cumulative sum
        positions[:,:,0] = torch.cumsum(dx, dim=1)
        positions[:,:,1] = torch.cumsum(dy, dim=1)
        yaws = selected_yaws
        # Create masked values for the last centerline point
        last_positions = centerline_traj[:, -1, :2].unsqueeze(1).expand(-1, horizon, -1)
        last_yaws = centerline_traj[:, -1, 2].unsqueeze(1).expand(-1, horizon)

        # Update positions and yaws based on the mask
        positions[last_idx_mask.expand_as(positions)] = last_positions[last_idx_mask.expand_as(last_positions)]
        yaws[last_idx_mask.squeeze(-1)] = last_yaws[last_idx_mask.squeeze(-1)]

        return positions, yaws
            

class StrivePlanner_trajdata(Policy):
    def __init__(self, device,dt=0.1,*args, **kwargs):
        super(StrivePlanner_trajdata, self).__init__(device)
       
        # Load Lane Graphs
        lane_graphs = load_lane_graphs_trajdata('nuscenes')
        # Instantiate Planner
        self.cfg = DEF_CONFIG
        self.planner = HardcodeNuscPlanner(lane_graphs, PlannerConfig(**self.cfg))
                
    def eval(self):
        pass

    @staticmethod
    def convert_to_tensor(obj):
        """Convert all float attributes of an object to PyTorch tensors."""
        for attr, value in vars(obj).items():
            if isinstance(value, (float, int)):
                setattr(obj, attr, torch.tensor(value))
  
    def get_action(self, obs_dict, **kwargs):
        batch_size = obs_dict["agent_hist"].shape[0]
        plan_batch = torch.zeros((batch_size, self.cfg["output_len"], 4)) #x,y,z,h
        
        for i in range(batch_size):
            map_name = obs_dict["map_names"][i].split(':')[1]
            
            hist_len =  self.cfg["hist_len"] 
            fut_len =  self.cfg["fut_len"] 
            num_neighbors = obs_dict["num_neigh"][i]

            agent_hist =  obs_dict["agent_hist"][i]
            ego_world = agent_hist            
            neigh_hist =  obs_dict["neigh_hist"][i]
        
            #mask out padded neighbors 
            neigh_world =  neigh_hist
            neigh_world = neigh_world[:num_neighbors]
            #assert that no neighbors would be all zero in the last 9 dimension at current timestep
            # Extract the values of all agents at hist_len for the last 9 dimensions
            slice_at_hist_len = neigh_world[:, hist_len, :]

            # Check if any row (agent) in the slice is all zeros
            all_zero_rows = (slice_at_hist_len == 0).all(dim=1)

            # If any agent (row) is all zeros, raise a ValueError
            if all_zero_rows.any():
                raise ValueError("Some agents have all zero values in the last 9 dimensions at hist_len, will be overlapping w/ ego.")
            
            if neigh_world.shape[0]>0:
                all_world = torch.vstack((ego_world[None],neigh_world))
            else:
                all_world = ego_world[None]
                
            all_world = StateTensor.from_array(all_world, format="x,y,z,xd,yd,xdd,ydd,s,c").as_format("x,y,c,s,xd,yd")
            world_from_agent = obs_dict["world_from_agent"][i]
            
            
            all_world[...,:2] = transform_points_tensor(all_world[...,:2],world_from_agent)
            #transform the c,s,xd,yd without displacement
            world_from_agent_rotate = self.get_rotation_matrix(world_from_agent)
            
            all_world[...,2:4] = transform_points_tensor(all_world[...,2:4], world_from_agent_rotate)
            all_world[...,4:6] = transform_points_tensor(all_world[...,4:6], world_from_agent_rotate)
            
            all_world = all_world.detach().cpu().numpy()
            x_y_hx_hy = all_world[:,:,:4]
            s = np.linalg.norm(all_world[:,:,4:],axis=2)[:,:,None]
            hdot = s*0 # not used
            x_y_hx_hy_s_hdot = np.dstack((x_y_hx_hy,s,s*0))

            # Initlialize Planner
            init_state = x_y_hx_hy_s_hdot
            veh_att = np.tile(np.array([self.cfg["vehicle_length"],self.cfg["vehicle_width"]]),(len(init_state),1))
            batch_mask = np.zeros(len(init_state))
            ego_idx = 0
            
            self.planner.reset(init_state[:,hist_len], veh_att, batch_mask, 1, [map_name], ego_idx=ego_idx)

            # Rollout planner
            agent_t = np.linspace(self.cfg['dt'], self.cfg["dt"]*fut_len, fut_len)
            
            # # for now use constant velocity model to predict agent_obs
            curr_speed = all_world[1:, hist_len,4:6]
            agent_init_state = init_state[1:,hist_len,:4] #x,y,hx,hy
            
            expanded_curr_speed = curr_speed[:, None, :]
            expanded_agent_init_state = agent_init_state[:, None, :]
            # Calculating the position based on constant velocity model
            pos = expanded_curr_speed * agent_t[None, :, None] + expanded_agent_init_state[..., :2]
            # Copying the heading from the initial state
            heading = np.repeat(expanded_agent_init_state[..., 2:4], repeats=fut_len, axis=1)
            # Stacking the position and heading together to form the agent_obs array
            agent_obs = np.concatenate([pos, heading], axis=-1)
            
            #only do const velocity model for next 10 frames
            # agent_obs[:,10:] = 0.0
            
            agent_ptr = np.array([0,len(agent_obs)])
            planner_t = agent_t
            plan = self.planner.rollout(agent_obs, agent_t, agent_ptr, planner_t)

            #transform back to agent centric
            agent_from_world = obs_dict["agents_from_world_tf"][i].cpu()
            plan = plan.to(device=agent_from_world.device, dtype=agent_from_world.dtype)
            
            plan[...,:2] = transform_points_tensor(plan[...,:2],agent_from_world)
            #transform the c,s,xd,yd without displacement
            agent_from_world_rotate = self.get_rotation_matrix(agent_from_world)
            plan[...,2:4] = transform_points_tensor(plan[...,2:4], agent_from_world_rotate)
            # Interpolation: Add a zero pad at the start and compute averages between consecutive points
            plan_padded = torch.cat((torch.zeros_like(plan[:, :1]), plan), dim=1)  # Shape: (batch, L+1, channels)
            interpolated_plan = (plan_padded[:, :-1] + plan_padded[:, 1:]) / 2  # Shape: (batch, L, channels)
            # Interleave interpolated points and original plan in plan_batch
            plan_batch[i, ::2] = interpolated_plan  # Even indices: interpolated values
            plan_batch[i, 1::2] = plan  # Odd indices: original values
            # plan_batch[i,:plan.shape[1]] = plan
            
        
        new_yaw = torch.atan2(plan_batch[...,3],plan_batch[...,2])
        action = Action(positions=plan_batch[..., :2], yaws=new_yaw[...,None])

        return action, {}
    
    @staticmethod
    def get_rotation_matrix(transform_matrix: torch.Tensor) -> torch.Tensor:
        """
        Create a rotation-only transformation matrix by removing translation.
        
        Args:
            transform_matrix: (3,3) transformation matrix
            
        Returns:
            rotation_matrix: (3,3) transformation matrix with only rotation
        """
        rotation_matrix = transform_matrix.clone()
        # Remove translation
        rotation_matrix[0:2, 2] = 0
        # Reset homogeneous coordinates
        rotation_matrix[2, :] = 0
        rotation_matrix[2, 2] = 1
        return rotation_matrix


