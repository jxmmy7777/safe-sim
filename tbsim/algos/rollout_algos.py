''' Pytorch Lightening module for Closed-loop policy training
'''
import pytorch_lightning as pl
import numpy as np
from typing import OrderedDict
import torch
import importlib
import os
from imageio import get_writer
import json
from collections import Counter
from pprint import pprint

from torch.utils.data import DataLoader,Dataset

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers
from tbsim.utils.env_utils import rollout_episodes
from tbsim.policies.wrappers import RolloutWrapper
from l5kit.simulation.unroll import ClosedLoopSimulator
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.policies.wrappers import Pos2YawWrapper
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvL5Builder, EnvDrivesimBuilder
from tbsim.utils.trajdata_utils import parse_trajdata_batch
from tbsim.utils.geometry_utils import VEH_VEH_collision
from tbsim.configs.selected_scene_config import SCENE_IDX_MAP,SCENE_NAMES,PREDEFINED_SCENE_INIT
from trajdata.simulation import SimulationScene

from tbsim.utils.tensor_utils import map_ndarray
from tbsim.utils.agent_rel_classify import load_predefined_scene_init_from_json
from imageio import get_writer
import pickle

import torch.optim as optim

class RolloutModule(pl.LightningModule):
    """A pytorch-lightning callback function that runs rollouts during training"""
    def __init__(
            self,
            algo_config,
            control_config,
            lts_config,
            exp_config,
            data_to_disk = False,
            save_video=False,
            video_dir=None,
            eval_mode = True,
            import_scene_list_mode="intersection",

    ):
        super(RolloutModule, self).__init__()
        self.algo_config = algo_config
        self.control_config = control_config
        self.lts_config = lts_config
        self._save_video = save_video
        self._video_dir = video_dir
        self._data_to_disk = data_to_disk
        self._eval_cfg = exp_config.clone()
        
       
        self.policy = None 
        self.env = None
        self.eval_mode = eval_mode
        
        
        #input is the list of scene name
        if import_scene_list_mode == "human":
            self.scene_indices = [SCENE_IDX_MAP[scene_name] for scene_name in SCENE_NAMES]
            self._eval_cfg.init_recipe["predefined_scene_init"] = PREDEFINED_SCENE_INIT
        
        elif import_scene_list_mode in ["intersection","merging"]:
            relationship = [import_scene_list_mode]
            interaction_list = ["equal"] #["equal","car2_behind_car1","car1_behind_car2"]
            predefined_scene_init = load_predefined_scene_init_from_json(relationship, interaction_list)
            # TODO multiple scene_indices for same scene
            self._eval_cfg.init_recipe["predefined_scene_init"] = predefined_scene_init
            # Modify this part to handle lists for each scene_name
            self.scene_indices = []
            for scene_name, scene_data in predefined_scene_init.items():
                repetition_count = len(scene_data["indices"])  # Since scene_data is always a list
                self.scene_indices.extend([SCENE_IDX_MAP[scene_name]] * repetition_count)
        elif import_scene_list_mode in ["intersection_1"]:
            relationship = ["intersection"]
            interaction_list = ["equal"] #["equal","car2_behind_car1","car1_behind_car2"]
            predefined_scene_init = load_predefined_scene_init_from_json(relationship, interaction_list)
            self._eval_cfg.init_recipe["predefined_scene_init"] = predefined_scene_init
            self.scene_indices = []
            for scene_name, scene_data in predefined_scene_init.items():
                # Choose only the first index from each list of scenes with the same name
                first_scene_index = SCENE_IDX_MAP[scene_name]
                self.scene_indices.append(first_scene_index)
        else:
            self._eval_cfg.init_recipe["predefined_scene_init"] = {}
            self.scene_indices = self._eval_cfg.eval_scenes
        ## duplicate scene_indices n_sample times 0,0,1,1,2,2
        # duplicate scene_indices n_sample times
        temp_scene_indices = []
        for scene_index in self.scene_indices:
            temp_scene_indices.extend([scene_index] * self.lts_config.n_sample_scene)
        self.scene_indices = temp_scene_indices
        self.n_sample_scene = self.lts_config.n_sample_scene
        self._eval_cfg.init_recipe["n_sample_scene"] = self.lts_config.n_sample_scene

     
        # self.policy.agents_policy.policy.nets["policy"].update_guide_config(self.LTS.cur_control_config)
        # self._eval_cfg.adjustment = self.LTS.cur_simulation_config.sim_config.adjustment
        self.automatic_optimization = False
    
    
    def setup_env_policy(self) -> None:
        print(f'Device on_train_start: {self.device}')
        #self._eval_cfg.num_scenes_per_batch==self.n_sample_scene * self.num
        self.num_scenes = self._eval_cfg.num_scenes_per_batch//self.n_sample_scene
        assert self._eval_cfg.num_scenes_per_batch == self.n_sample_scene *  self.num_scenes
        self.LTS = LTSModule(self.control_config, self.lts_config, 
                             num_scenes=self.num_scenes, 
                             n_sample_scene=self.n_sample_scene, 
                             device=self.device)
        if "sim_config"  in self.LTS.cur_simulation_config:
            self._eval_cfg.adjustment.enabled = True
        self._get_env_and_policy()
        self.update_sim_config()

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return None

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        return metrics

    def configure_optimizers(self):
        # if hasattr(self.policy, 'ego_policy') and hasattr(self.policy.ego_policy, 'policy') and not hasattr(self.policy.ego_policy.policy, 'nets'):
        #     return None
        # optim_params = self.algo_config.optim_params["policy"]
        # return optim.Adam(
        #     params=self.policy.ego_policy.policy.nets["policy"].parameters(),
        #     lr=optim_params["learning_rate"]["initial"],
        #     weight_decay=optim_params["regularization"]["L2"],
        # )
        return None 

       
    def _get_env_and_policy(self):
      
        policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
        composer_class = getattr(policy_composers, self._eval_cfg.eval_class)
        composer = composer_class(self._eval_cfg, self.device, ckpt_root_dir=self._eval_cfg.ckpt_root_dir)
        print("Building composer {}".format(self._eval_cfg.eval_class))
     
        policy, exp_config = composer.get_policy()

        if self._eval_cfg.policy.pos_to_yaw:
            policy = Pos2YawWrapper(
                policy,
                dt=exp_config.algo.step_time if exp_config is not None else 0.1,
                yaw_correction_speed=self._eval_cfg.policy.yaw_correction_speed
            )
        # if eval_cfg.rolling_perturb.enabled:
        #     OU_pert = OrnsteinUhlenbeckPerturbation(theta=eval_cfg.rolling_perturb.OU.theta*np.ones(3),
        #                 sigma=eval_cfg.rolling_perturb.OU.sigma*np.array(eval_cfg.rolling_perturb.OU.scale))
        #     policy = PerturbationWrapper(policy,OU_pert)
       
        if self._eval_cfg.agent_eval_class is not None:
            composer_class = getattr(policy_composers, self._eval_cfg.agent_eval_class)
            composer = composer_class(self._eval_cfg, self.device, ckpt_root_dir=self._eval_cfg.ckpt_root_dir)
            agent_policy, exp_config = composer.get_policy()
            if self._eval_cfg.policy.pos_to_yaw:
                agent_policy = Pos2YawWrapper(
                    agent_policy,
                    dt=exp_config.algo.step_time,
                    yaw_correction_speed=self._eval_cfg.policy.yaw_correction_speed
                )
        else:
            agent_policy = None


        if self._eval_cfg.env in ["nusc","drivesim"]:
            if self._eval_cfg.agent_eval_class is not None:
                rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
            else:
                rollout_policy = RolloutWrapper(agents_policy=policy)
        elif self._eval_cfg.ego_only:
            rollout_policy = RolloutWrapper(ego_policy=policy)
        else:
            if self._eval_cfg.agent_eval_class is not None:
                rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
            else:
                rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

        self.policy = rollout_policy
        # create env
        if self._eval_cfg.env == "nusc":
            if agent_policy is not None:
                split_ego = True
            else:
                split_ego = False
            env_builder = EnvNuscBuilder(eval_config=self._eval_cfg, exp_config=exp_config, device=self.device)
            if "parse_obs" in exp_config.env.data_generation_params:
                parse_obs=exp_config.env.data_generation_params.parse_obs
            else:
                parse_obs=True
            env = env_builder.get_env(split_ego=split_ego,parse_obs=parse_obs)
        elif self._eval_cfg.env == "drivesim":
            if agent_policy is not None:
                split_ego = True
            else:
                split_ego = False
            env_builder = EnvDrivesimBuilder(eval_config=self._eval_cfg, exp_config=exp_config, device=self)
            if "parse_obs" in exp_config.env.data_generation_params:
                parse_obs=exp_config.env.data_generation_params.parse_obs
            else:
                parse_obs=True
            env = env_builder.get_env(split_ego=split_ego,parse_obs=parse_obs)
        elif self._eval_cfg.env == 'l5kit':
            env_builder = EnvL5Builder(eval_config=self._eval_cfg, exp_config=exp_config, device=self)
            env = env_builder.get_env()
        else:
            raise NotImplementedError("{} is not a valid env".format(self._eval_cfg.env))
        self.env = env
#double check this implmentation 
    def _run_rollout(self, batch_idx: int):
        if not hasattr(self.policy, 'ego_policy'):
            self.setup_env_policy()
        scene_i = 0#batch_idx*self._eval_cfg.num_scenes_per_batch
        # eval_scenes = self._eval_cfg.eval_scenes

        result_stats = None

        total_adjust_plan = dict()
        total_trace = dict()
        total_info = dict()
        total_renderings = []

        #modify to the same num as batch_scenes, TODO consider config is num_batch,but we run num_scene_per_batch
        while scene_i < self._eval_cfg.num_scenes_per_batch:
            # scene_indices = eval_scenes[scene_i: scene_i + self._eval_cfg.num_scenes_per_batch]
            scene_indices = self.scene_indices[scene_i: scene_i + self._eval_cfg.num_scenes_per_batch]
            scene_i += self._eval_cfg.num_scenes_per_batch
            stats, info, renderings,adjust_plans,trace = rollout_episodes(
                self.env,
                self.policy,
                num_episodes=self._eval_cfg.num_episode_repeats,
                n_step_action=self._eval_cfg.n_step_action,
                render=self._save_video,
                skip_first_n=self._eval_cfg.skip_first_n,
                scene_indices=scene_indices,
                start_frame_index_each_episode=self._eval_cfg.start_frame_index_each_episode,
                seed_each_episode=self._eval_cfg.seed_each_episode,
                horizon=self._eval_cfg.num_simulation_steps,
                adjust_plan_recipe=self._eval_cfg.adjustment.to_dict() if self._eval_cfg.adjustment.enabled else None,
                init_recipe =self._eval_cfg.init_recipe.to_dict(), #LTS will output ego_init and target_init info
                control_config =self.LTS.cur_control_config,
                device = self.device,
                reset_scene_index_map = True #Since we want to run the same scenes everytime @TODO only set True if we finish all the scenes
            )

            for ei,adjust_plan in enumerate(adjust_plans):
                for k,v in adjust_plan.items():
                    total_adjust_plan["{}_{}".format(k,ei)]=v 
            for ei,trace_i in enumerate(trace):
                for k,v in trace_i.items():
                    total_trace["{}_{}".format(k,ei)]=v

            print(info["scene_index"])
            pprint(stats)

            if result_stats is None:
                result_stats = stats.copy()
                result_stats["scene_index"] = np.array(info["scene_index"])
            else:
                for k in stats:
                    result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
                result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])
        
            # Aggregate info
            for k, v in info.items():
                if k not in total_info:
                    total_info[k] = v
                else:
                    if isinstance(v, list):
                        total_info[k].extend(v)
                    elif isinstance(v, dict):
                        total_info[k].update(v)

            # Store rendering in total_renderings
            if self._save_video:
                total_renderings.extend(renderings)
        # Process and save the renderings
        if self._save_video:
            for ei, episode_rendering in enumerate(total_renderings):
                for i, scene_images in enumerate(episode_rendering):
                    video_dir = os.path.join(self._eval_cfg.results_dir, "videos/")
                    writer = get_writer(os.path.join(
                        video_dir, "ep{}_{}_{}.mp4".format(self.current_epoch,total_info["scene_index"][i], ei)), fps=10)
                    print("video to {}".format(os.path.join(
                        video_dir, "ep{}_{}_{}.mp4".format(self.current_epoch,total_info["scene_index"][i], ei))))
                    for im in scene_images:
                        writer.append_data(im)
                    writer.close()

        # Write data to disk after while loop
        # At the end of the while loop
        if self._data_to_disk:
            # Write result_stats
            with open(os.path.join(self._eval_cfg.results_dir, f"ep{self.current_epoch}_stats.json"), "w+") as fp:
                stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
                json.dump(stats_to_write, fp)
            
            # Write total_info
            # serializable_info = numpy_to_list(total_info)
            # with open(os.path.join(self._eval_cfg.results_dir, "total_info.json"), "w+") as fp:
            #     json.dump(serializable_info, fp)
            
            # # Write total_adjust_plan
            # if len(total_adjust_plan) > 0:
            #     with open(os.path.join(self._eval_cfg.results_dir, "adjust_plan.json"), "w+") as fp:
            #         json.dump(total_adjust_plan, fp)
            
            # # Write total_trace
            # if len(total_trace) > 0:
            #     with open(os.path.join(self._eval_cfg.results_dir, "trace.pkl"), "wb") as fp:
            #         pickle.dump(total_trace, fp)
            
            # Additional disk writing, if any
            self._write_results_to_disk(total_info, total_adjust_plan, total_trace)

        return result_stats
        
    def training_step(self, batch, batch_idx):
        # Add logic to decide whether to run rollout based on global step and other conditions
        if self.eval_mode:
            # dummy tensor to avoid error
            return {"loss":torch.tensor(0.0) , "metrics": None}
        else:
            raise NotImplementedError
            # Run the rollout
            stats = self._run_rollout(self.global_step)
            # Optionally log stats or do something else with the result
            return {"losses": None, "metrics": stats}

    def validation_step(self, batch, batch_idx):
        # Similar logic to training_step, depending on how you want to handle rollouts during validation
        stats = self._run_rollout(batch_idx)
        
        return {"losses": None, "metrics": stats}
    
    def validation_epoch_end(self, outputs) -> None:
        # aggregate metrics
        aggregated_metrics = {}
        for k in outputs[0]["metrics"]:
            # Aggregate by concatenating
             # Check if the numpy array contains strings
            if outputs[0]["metrics"][k].dtype.type is np.str_:
                continue  # Skip string arrays
            if isinstance(outputs[0]["metrics"][k] , np.ndarray):
                m = torch.cat([torch.tensor(o["metrics"][k]) for o in outputs], dim=0)
                aggregated_metrics[k] = m
            self.log("metrics/" + k, m.to(dtype=torch.float32).mean())
            if k in self.LTS.target_metrics:
                self.log("target_metrics/" + k, m.to(dtype=torch.float32).mean())
        #LTS update based on metrics TODO metric for each scene
        logs = self.LTS.update_control_cfg(aggregated_metrics)

        
        # Flatten the dictionary to create meaningful keys
        flat_config = flatten_dict(self.LTS.cur_control_config["guide_config"], parent_key='control')

        # Log each key-value pair
        for k, v in flat_config.items():
            if torch.is_tensor(v) and v.numel() > 1:   # Check if tensor has more than one element
                v = v.mean()     
            self.log("control_sample/{}".format(k),v)
        # Log each mean 
        for k, v in  self.LTS.means.items():
            self.log("control_mean/{}".format(k),v)
        # Log each probs
        # for k, v in  self.LTS.logits.items():
        #     self.log(f'control_probs/{k}', v, histogram=True)
         
        for key, value in logs.items():
            self.log(f"update/{key}", value)
          
        # Update Diffusion and Simulation params
        self.update_sim_config()
        
        return None

    def update_sim_config(self):
        # Update diffusion model config, we need to consider batch dim vs num_scene_toevaluate
        # self.policy.agents_policy.policy.nets["policy"].update_guide_config(self.LTS.cur_control_config) Moved to rollout_episode
        # Update Simulator config (initialization params)
        self._eval_cfg.adjustment.update(self.LTS.cur_simulation_config.sim_config.adjustment)
        # update init recipe in the future
        
    def _write_results_to_disk(self, total_info, total_adjust_plan, total_trace):
        
        experience_hdf5_path = os.path.join(self._eval_cfg.results_dir, f"data_ep{self.current_epoch}.hdf5")
        if self._data_to_disk and "buffer" in total_info:
            self.dump_episode_buffer(
                total_info["buffer"],
                total_info["scene_index"],
                h5_path=experience_hdf5_path
            )
        torch.cuda.empty_cache()
        if len(total_adjust_plan)>0:
            with open(os.path.join(self._eval_cfg.results_dir, "adjust_plan.json"),"w+") as fp:
                json.dump(total_adjust_plan,fp)
                print("adjust plan saved to {}".format(os.path.join(self._eval_cfg.results_dir, "adjust_plan.json")))

        info_except_buffer = {k:v for k,v in total_info.items() if k!="buffer"}
        for k,v in info_except_buffer.items():
            if k not in total_info:
                total_info[k]=v
            else:
                if isinstance(v,list):
                    total_info[k].extend(v)
                elif isinstance(v,dict):
                    total_info[k].update(v)

        with open(os.path.join(self._eval_cfg.results_dir, "sim_info.json"),"w+") as fp:
            json.dump(total_info["map_info"],fp)
            print("sim info saved to {}".format(os.path.join(self._eval_cfg.results_dir, "sim_info.json")))

        # if len(total_trace)>0:
        #     with open(os.path.join(self._eval_cfg.results_dir, "trace.pkl"),"wb") as fp:
        #         pickle.dump(total_trace,fp)
        #         print("trace saved to {}".format(os.path.join(self._eval_cfg.results_dir, "trace.pkl")))

    @staticmethod
    def dump_episode_buffer(buffer, scene_index, h5_path):
        import h5py
        h5_file = h5py.File(h5_path, "a")

        ep_count = Counter()
        for si, scene_buffer in zip(scene_index, buffer):
            # TODO: fix this hack
            # Postfix scene index with episode count (scene may repeat with multiple episodes)
            ep_i = ep_count[si]
            ep_count[si] += 1
            for mk in scene_buffer:
                h5key = "/{}_{}/{}".format(si, ep_i, mk)
                if h5key in h5_file:
                    print(f"{h5key} already exists, overwriting...")
                    del h5_file[h5key]
                h5_file.create_dataset(h5key, data=scene_buffer[mk])
        h5_file.close()
        print("scene {} written to {}".format(scene_index, h5_path))

    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=1)  # Define a dummy dataset

    def val_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=1)  # Define a dummy dataset
    
class DummyDataset(Dataset):
    def __init__(self, num_steps=1):
        self.num_steps = num_steps
    def __len__(self):
        return self.num_steps  # You can set the length to whatever you want

    def __getitem__(self, idx):
        return torch.tensor([0])  # Return a dummy tensor
    
def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items