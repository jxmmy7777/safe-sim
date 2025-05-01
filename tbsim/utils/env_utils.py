from typing import OrderedDict
import numpy as np
import pytorch_lightning as pl
import torch
import importlib
import os
from imageio import get_writer

from tbsim.envs.base import BatchedEnv, BaseEnv
from tbsim.configs.env_configs import get_eval_scenes_and_predefined_init, import_env_configs

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers
from tbsim.evaluation.env_builders import EnvNuscBuilder,EnvNuplanBuilder

from trajdata.simulation import SimulationScene
import random

from collections import defaultdict

def rollout_episodes(
    env,
    policy,
    num_episodes,
    skip_first_n=1,
    n_step_action=1,
    render=False,
    scene_indices=None,
    start_frame_index_each_episode=None,
    device=None,
    obs_to_torch=True,
    adjust_plan_recipe=None,
    init_recipe = None,
    control_config = None,
    horizon=None,
    seed_each_episode=None,
    reset_scene_index_map=False,
    initialize=False,    
):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy (RolloutWrapper): a policy that controls agents in the environment
        num_episodes (int): number of episodes to rollout for
        skip_first_n (int): number of steps to skip at the begining
        n_step_action (int): number of steps to take between querying models
        render (bool): if True, return a sequence of rendered frames
        scene_indices (tuple, list): (Optional) scenes indices to rollout with
        start_frame_index_each_episode (List): (Optional) which frame to start each simulation episode from,
        device: device to cast observation to
        obs_to_torch: whether to cast observation to torch
        adjust_plan_recipe (dict): (Optional) initialization condition, either a fixed plan or a recipe for random generation
        horizon (int): (Optional) override horizon of the simulation
        seed_each_episode (List): (Optional) a list of seeds, one for each episode

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
        renderings (list): A list of rendered frames in the form of np.ndarray, one for each episode
    """
    stats = {}
    info = {}
    renderings = []
    is_batched_env = isinstance(env, BatchedEnv)
    timers = Timers()
    adjust_plans = list()
    if seed_each_episode is not None:
        assert len(seed_each_episode) == num_episodes
    if start_frame_index_each_episode is not None:
        assert len(start_frame_index_each_episode) == num_episodes
        
    ego_policy = policy.unwrap()["Rollout.ego_policy"]
    trace = list()
    for ei in range(num_episodes):
        if start_frame_index_each_episode is not None:
            start_frame_index = start_frame_index_each_episode[ei]
        else:
            start_frame_index = None
        
        env.reset(scene_indices=scene_indices, start_frame_index=start_frame_index)
        if adjust_plan_recipe is not None:
            if "random_init_plan" in adjust_plan_recipe:
                # recipe provided
                if adjust_plan_recipe["random_init_plan"]:
                    adjust_recipe = adjust_plan_recipe
                    raise NotImplementedError("Random initialization is not implemented yet")
                    adjust_plan = random_initial_adjust_plan(env,adjust_recipe)
                    
                else:
                    adjust_plan = None
                    adjust_recipe = None 
            else:
                # explicit plan provided
                adjust_plan = adjust_plan_recipe
        else:
            adjust_plan = None
            adjust_recipe = None
        if adjust_plan is not None:
            env.adjust_scene(adjust_plan)
        #initialize the relations
        # if initialize:
        #     env.save_relationships_to_file()
        if init_recipe["predefined_scene_init"] is not None:
            if not hasattr(env,"scene_index_map") or reset_scene_index_map:
                 #for multiple same scenes and run batch each iteration
                env.scene_index_map = defaultdict(int)
                env.scene_occurrence_map = defaultdict(int)
            env.init_ego_and_target_agents(init_recipe)
            env.adjust_ego() #set ego indices
            
            #update batch_indices
            if control_config is not None:
                control_config.guide_config.batch_ctrl_indices = env.batch_ctrl_indices
                control_config.guide_config.batch_ego_indices = env.batch_ego_indices
                policy.agents_policy.policy.nets["policy"].update_guide_config(control_config,device) #specify device!
        else:
            #if split ego and no
            env.init_split_indices()
       
        if seed_each_episode is not None:
            env.update_random_seed(seed_each_episode[ei])
            np.random.seed(seed_each_episode[ei])
            random.seed(seed_each_episode[ei])
            torch.manual_seed(seed_each_episode[ei])
            torch.cuda.manual_seed(seed_each_episode[ei])
            

        done = env.is_done()
        counter = 0
        step_since_last_update = 0
        frames = list()
        while not done:
            timers.tic("step")
            with timers.timed("obs"):
                obs = env.get_observation(include_ego_obs=True)
            with timers.timed("to_torch"):
                if obs_to_torch:    
                    device = policy.device if device is None else device
                    obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
                else:
                    obs_torch = obs

            with timers.timed("network"):
                action = policy.get_action(obs_torch, step_index=counter)
                
            if counter < skip_first_n:
                # use GT action for the first N steps to warm up environment state (velocity, etc.)
                gt_action = env.get_gt_action(obs) #TODO we should eliminate ego action in agents
                action.ego = gt_action.ego
                gt_action.agents.eliminate_ego_action(obs["agents"]["ego_idx"])
                action.agents = gt_action.agents
                env.step(action, num_steps_to_take=1, render=False)
                counter += 1
                step_since_last_update+=1
            else:
                with timers.timed("env_step"):
                    ims = env.step(
                        action, num_steps_to_take=n_step_action, render=render
                    )  # List of [num_scene, h, w, 3]
                if render:
                    frames.extend(ims)
                counter += n_step_action
                step_since_last_update += n_step_action
            timers.toc("step")
            # print(timers)

            done = env.is_done()
            
            if horizon is not None and counter >= horizon:
                break
        metrics = env.get_metrics()
        if hasattr(ego_policy,"savetrace") and ego_policy.savetrace:
            trace.append(ego_policy.trace.copy())
            

        for k, v in metrics.items():
            if k not in stats:
                stats[k] = []
            if is_batched_env:  # concatenate by scene
                stats[k] = np.concatenate([stats[k], v], axis=0)
            else:
                stats[k].append(v)

        env_info = env.get_info()
        for k, v in env_info.items():
            if k not in info:
                if isinstance(v,dict):
                    info[k] = dict()
                else:
                    info[k] = list()

            if is_batched_env:
                if isinstance(v,dict):
                    info[k].update(v)
                else:
                    info[k].extend(v)
            else:
                info[k].append(v)
        del env_info
        if hasattr(ego_policy,"reset"):
            ego_policy.reset()
        if render:
            frames = np.stack(frames)
            if is_batched_env:
                # [step, scene] -> [scene, step]
                frames = frames.transpose((1, 0, 2, 3, 4))
            renderings.append(frames)
        if adjust_plan is not None:
            adjust_plans.append(adjust_plan)

    multi_episodes_metrics = env.get_multi_episode_metrics()
    stats.update(multi_episodes_metrics)
    env.reset_multi_episodes_metrics()

    return stats, info, renderings, adjust_plans, trace

def build_environment_and_scenes(eval_cfg, exp_config, device, agent_policy=None, import_scene_list_mode="None"):
    """
    Build environment and get scene selection in one go.
    
    Returns:
        env: The environment object
        eval_scenes: List of scene indices to evaluate
        predefined_init: Dictionary of scene initialization parameters
    """
    # 1. Import environment-specific configs
    TRAIN_SCENE_IDX_MAP, SCENE_IDX_MAP, SCENE_NAMES, \
    PREDEFINED_SCENE_INIT, PREDEFINED_SCENE_ALL_INIT = import_env_configs(eval_cfg.env)

    # 2. Build environment based on type
    split_ego = agent_policy is not None
    parse_obs = exp_config.env.data_generation_params.get("parse_obs", True)

    if eval_cfg.env == "nusc":
        env_builder = EnvNuscBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    elif eval_cfg.env == "nuplan":
        env_builder = EnvNuplanBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    else:
        raise ValueError(f"Unknown environment: {eval_cfg.env}")

    env = env_builder.get_env(split_ego=split_ego, parse_obs=parse_obs)

    # 3. Get scene selection based on mode
    eval_scenes, predefined_init = get_eval_scenes_and_predefined_init(
        import_scene_list_mode,
        eval_cfg,
        SCENE_IDX_MAP,
        SCENE_NAMES,
        PREDEFINED_SCENE_INIT,
        PREDEFINED_SCENE_ALL_INIT,
        TRAIN_SCENE_IDX_MAP
    )

    return env, eval_scenes, predefined_init
