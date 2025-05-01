"""
A script for evaluating closed-loop simulation, merging policy/env initialization
with guidance configuration, parameter parsing, and result directory naming.
"""

import argparse
import importlib
import json
import os
import pickle
import random
from collections import Counter
from pprint import pprint

import numpy as np
import torch
import yaml
from imageio import get_writer

from tbsim.configs.eval_config import EvaluationConfig

from tbsim.policies.wrappers import  Pos2YawWrapper, RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.env_utils import rollout_episodes, build_environment_and_scenes
from tbsim.utils.tensor_utils import map_ndarray

from tbsim.configs.guidance_config import GuidanceConfig
from tbsim.configs.base import Dict

torch.set_float32_matmul_precision("medium")

# ------------------------------------------------------------------
# Utility functions for parsing incoming guidance parameters
# ------------------------------------------------------------------

def parse_value(value_str):
    """Parse string value into appropriate Python type."""
    value_str = value_str.strip()
    
    if value_str.lower() == 'none':
        return None
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
        
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            elements = [parse_value(e.strip()) for e in value_str[1:-1].split(',')]
            return elements
        except:
            pass
            
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            import ast
            return ast.literal_eval(value_str)
        except:
            pass
    
    try:
        return float(value_str) if '.' in value_str else int(value_str)
    except:
        return value_str

def parse_guidance_params(param_strings):
    """
    Parse guidance parameters from a single string of the form:
      guidance_name,param_name,value;guidance_name2,param_name2,value2
    """
    if not param_strings:
        return {}
        
    params_dict = {}
    param_groups = param_strings.split(';')
    
    for param_str in param_groups:
        try:
            parts = param_str.split(',', 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid parameter format: {param_str}")
                
            guidance_name, param_name, value_str = parts
            value = parse_value(value_str)
            
            if guidance_name not in params_dict:
                params_dict[guidance_name] = {}
            params_dict[guidance_name][param_name] = value
            
        except Exception as e:
            raise ValueError(
                f"Error parsing parameter: {param_str}\n"
                f"Error: {str(e)}\n"
                "Expected format: guidance_name,param_name,value"
            )
    
    return params_dict

def build_single_policy(eval_cfg, device, exp_config=None):
    """Build a single policy for all agents."""
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
    policy, exp_config = composer.get_policy()
    
    if eval_cfg.policy.pos_to_yaw:
        policy = Pos2YawWrapper(
            policy,
            dt=exp_config.algo.step_time if exp_config is not None else 0.1,
            yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
        )
    return policy, exp_config

def build_dual_policies(eval_cfg, device, modify_cfg):
    """Build separate policies for ego and agents."""
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")

    # Build ego policy
    policy, _ = build_single_policy(eval_cfg, device)
    
    # Build agent policy
    composer_class = getattr(policy_composers, eval_cfg.agent_eval_class)
    composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
    agent_policy, exp_config = composer.get_policy(modify_config=modify_cfg)
    
    if eval_cfg.policy.pos_to_yaw:
        agent_policy = Pos2YawWrapper(
            agent_policy,
            dt=exp_config.algo.step_time if exp_config is not None else 0.1,
            yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
        )
    
    return policy, agent_policy, exp_config
# ------------------------------------------------------------------
# The primary function that 1)intilize the scene and 2)runs simulation
# ------------------------------------------------------------------
def run_adv_simulation(eval_cfg,
                       data_to_disk,
                       render_to_video,
                       import_scene_list_mode="None",
                       visualize_diffusion=False,
                       guidance_params=None):
    """
    Run adversarial simulation with configurable parameters, plus
    environment and policy building in the same style you previously used.
    """
    ## set env config 

    # ------------------ Reproducibility settings ------------------
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------ Choose batch type based on environment ------------------
    if eval_cfg.env in ["nusc", "drivesim", "nuplan"]:
        set_global_batch_type("trajdata")
    elif eval_cfg.env == 'l5kit':
        set_global_batch_type("l5kit")

    print(eval_cfg)

    # ------------------ Guidance configuration ------------------
    guide_config = GuidanceConfig()
    guide_config.set_guidance_fn(eval_cfg.guidance_fn)

    # Default parameters:
    guide_config.update_params({
        'inner_lr': 0.2,
        'scale_grad_by_std': False,
        'inner_beta': 0.5,
        'multiple_guidance_strategy': "weight_guide",
        'grad_wrt': "clean_guide",
        'sample_mode': "ddpm",
        'sample_step': 1
    })

    # If user specified overrides via --guidance_params:
    if guidance_params:
        parsed_params = parse_guidance_params(guidance_params)
        for guidance_name, params in parsed_params.items():
            if guidance_name == 'params':
                guide_config.update_params(params)
            elif guidance_name == 'combine_loss':
                guide_config.update_combine_loss(params)
            else:
                guide_config.update_config(guidance_name, params)

    # Wrap into `modify_cfg`
    modify_cfg = Dict()
    modify_cfg.guide_config = Dict(guide_config.to_dict())

    # ------------------ Build the result directory name ------------------
    # result_dir_name = create_result_dir_name(guidance_params)
    eval_cfg.results_dir = os.path.join(eval_cfg.results_dir)
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    # Optionally store the final guidance config & params
    config_dir = os.path.join(eval_cfg.results_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    if guidance_params:
        with open(os.path.join(config_dir, "guidance_params.json"), "w") as f:
            json.dump(parse_guidance_params(guidance_params), f, indent=2)
    with open(os.path.join(config_dir, "full_guidance_config.json"), "w") as f:
        json.dump(guide_config.to_dict(), f, indent=2)

    # ------------------ Prepare device ------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------ Build the policy/policies and environment ------------------    
    # single policy for all agents:
    if eval_cfg.agent_eval_class is None:
        policy, exp_config = build_single_policy(eval_cfg, device)
        agent_policy = None
    # dual policies for ego and agents:
    else:
        policy, agent_policy, exp_config = build_dual_policies(eval_cfg, device, modify_cfg)
    if eval_cfg.env in ["nusc","drivesim","nuplan"]:
        if eval_cfg.agent_eval_class is not None:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
        else:
            rollout_policy = RolloutWrapper(agents_policy=policy)
    elif eval_cfg.ego_only:
        rollout_policy = RolloutWrapper(ego_policy=policy)
    else:
        if eval_cfg.agent_eval_class is not None:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
        else:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

   
    # Prepare the scene selection
    env, eval_scenes, predefined_init = build_environment_and_scenes(eval_cfg, exp_config, device, agent_policy,import_scene_list_mode)
    # 3) Store that predefined init in the config so your environment/rollout can see it
    eval_cfg.init_recipe["predefined_scene_init"] = predefined_init
    
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]

    # We will store stats, etc.
    result_stats = None
    scene_i = 0

    # Prepare for data collection
    total_adjust_plan = {}
    total_trace = {}
    total_info = {}

    # -------------- Actual simulation rollout loop --------------
    while scene_i < min(eval_cfg.num_scenes_to_evaluate, len(eval_scenes)):
        scene_indices = eval_scenes[scene_i : scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch

        stats, info, renderings, adjust_plans, trace = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=eval_cfg.num_episode_repeats,
            n_step_action=eval_cfg.n_step_action,
            render=render_to_video,
            skip_first_n=eval_cfg.skip_first_n,
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch,
            start_frame_index_each_episode=eval_cfg.start_frame_index_each_episode,
            seed_each_episode=eval_cfg.seed_each_episode,
            horizon=eval_cfg.num_simulation_steps,
            adjust_plan_recipe=eval_cfg.adjustment.to_dict() if eval_cfg.adjustment.enabled else None,
            init_recipe=eval_cfg.init_recipe,
            control_config=modify_cfg,
            device=device,
        )

        # Merge stats
        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            for k in stats:
                result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate(
                [result_stats["scene_index"], np.array(info["scene_index"])]
            )

        # Collect adjustments and traces
        for ei, adjust_plan in enumerate(adjust_plans):
            for k, v in adjust_plan.items():
                total_adjust_plan[f"{k}_{ei}"] = v
        for ei, trace_i in enumerate(trace):
            for k, v in trace_i.items():
                total_trace[f"{k}_{ei}"] = v

        # Print the stats for this batch
        print(info["scene_index"])
        pprint(stats)

        # Save stats to disk
        stats_filepath = os.path.join(eval_cfg.results_dir, "stats.json")
        stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
        with open(stats_filepath, "w") as fp:
            json.dump(stats_to_write, fp)

        # Save videos if requested
        if render_to_video:
            video_dir = os.path.join(eval_cfg.results_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            for ei, episode_rendering in enumerate(renderings):
                for i, scene_images in enumerate(episode_rendering):
                    outname = f"{info['scene_index'][i]}_{ei}.mp4"
                    writer = get_writer(os.path.join(video_dir, outname), fps=10)
                    print(f"Video -> {os.path.join(video_dir, outname)}")
                    for im in scene_images:
                        writer.append_data(im)
                    writer.close()

        # Possibly save data to disk
        if data_to_disk and "buffer" in info:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                h5_path=os.path.join(eval_cfg.results_dir, "data.hdf5")
            )

        # Save partial logs
        if total_adjust_plan:
            with open(os.path.join(eval_cfg.results_dir, "adjust_plan.json"), "w") as fp:
                json.dump(total_adjust_plan, fp)
                print("Saved adjust_plan.json")

        info_except_buffer = {k: v for k, v in info.items() if k != "buffer"}
        for k, v in info_except_buffer.items():
            if k not in total_info:
                total_info[k] = v
            else:
                if isinstance(v, list):
                    total_info[k].extend(v)
                elif isinstance(v, dict):
                    total_info[k].update(v)

        with open(os.path.join(eval_cfg.results_dir, "sim_info.json"), "w") as fp:
            json.dump(total_info, fp)
            print("Saved sim_info.json")

        if total_trace:
            with open(os.path.join(eval_cfg.results_dir, "trace.pkl"), "wb") as fp:
                pickle.dump(total_trace, fp)
                print("Saved trace.pkl")

        torch.cuda.empty_cache()

    return result_stats, total_info, None, total_adjust_plan, total_trace


def dump_episode_buffer(buffer, scene_index, h5_path):
    """
    Example method to dump data from each scene into an HDF5 file.
    """
    import h5py
    h5_file = h5py.File(h5_path, "a")
    ep_count = Counter()
    for si, scene_buffer in zip(scene_index, buffer):
        ep_i = ep_count[si]
        ep_count[si] += 1
        for mk in scene_buffer:
            h5key = f"/{si}_{ep_i}/{mk}"
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print(f"scene {scene_index} written to {h5_path}")


# ------------------------------------------------------------------
# CLI argument parser setup
# ------------------------------------------------------------------

def add_simulation_args(parser):
    """Add simulation-related arguments (already done in your code)."""
    parser.add_argument("--config_file", type=str, default=None,
                       help="A json file containing evaluation configs")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="local rank for torch.distributed")
    parser.add_argument("--env", type=str, required=True,
                       choices=["nusc", "drivesim", "nuplan", "l5kit"],
                       help="Environment to use")
    parser.add_argument("--eval_class", type=str, default=None,
                       help="Optionally specify the evaluation class through argparse")
    parser.add_argument("--agent_eval_class", type=str, default=None,
                       help="Optionally specify the evaluation class for agents if it's different from ego")
    parser.add_argument("--ckpt_root_dir", type=str, default=None,
                       help="Root directory to look for training run directories")
    parser.add_argument("--policy_ckpt_dir", type=str, default=None,
                       help="Directory to look for saved checkpoints")
    parser.add_argument("--policy_ckpt_key", type=str, default=None,
                       help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Root directory of the dataset")
    parser.add_argument("--num_scenes_per_batch", type=int, default=None,
                       help="Number of scenes to run concurrently (to accelerate eval)")
    parser.add_argument("--results_root_dir", type=str, required=True,
                       help="Root directory for results")
    parser.add_argument("--render", action="store_true",
                       help="Whether to render simulation to video")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--guidance_fn_weights", type=lambda s: list(map(float, s.split('_'))), default=None)
    parser.add_argument("--load-cache", action="store_true",
                       help="whether to evaluate the model")
    parser.add_argument("--data_filter_fn", type=str, choices=["has_lane"], default="")
    parser.add_argument("--scene_select_mode", type=str, default="intersection",
                       choices=["human", "ttc", "partial_diffusion", "no_collision"])
    parser.add_argument("--visualize-diffusion", action="store_true",
                       help="whether to evaluate the model")
    parser.add_argument("--sim-steps", type=int, default=100)
    parser.add_argument("--num_scenes_to_evaluate", type=int, default=100)
    parser.add_argument("--skip_first_n", type=int, default=0)
    parser.add_argument("--n_step_action", type=int, default=5)
    parser.add_argument("--guidance_fn", type=lambda s: s.split('_'),
                       default=["offroad"])
    parser.add_argument("--guidance_params", type=str,
                       help="""
                       Guidance parameters in format: guidance_name,param_name,value;guidance_name2,param_name2,value2
                       
                       Examples:
                       - Basic parameters:
                         params,inner_lr,0.2;params,scale_grad,False
                       - Lists and weights:
                         combine_loss,weights,[0.5,0.5,0.0,0.0]
                       - Dictionaries:
                         causecollision,adv_term_weight,{"distance":1.0,"speed_penalty":0.0}
                       - Multiple parameters:
                         params,inner_lr,0.3;collision,radius,2.0;speed,desired_speed,2.5
                       """)
    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )
    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )


def main():
    parser = argparse.ArgumentParser(description="Run adversarial simulation")
    add_simulation_args(parser)
    args = parser.parse_args()

    # Load base config
    cfg = EvaluationConfig()
    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    # Update config with command line arguments
    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.ckpt_root_dir is not None:
        cfg.ckpt_root_dir = args.ckpt_root_dir

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch
        
    # Set simulation related parameters
    cfg.nusc.num_simulation_steps = args.sim_steps
    cfg.num_scenes_to_evaluate = args.num_scenes_to_evaluate
    cfg.nusc.skip_first_n = args.skip_first_n
    cfg.nusc.n_step_action = args.n_step_action

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.agent_eval_class is not None:
        cfg.agent_eval_class = args.agent_eval_class

    if args.seed is not None:
        cfg.seed = args.seed
        
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None
        

    # Set guidance parameters
    cfg.guidance = args.guidance
    cfg.guidance_fn = args.guidance_fn

    # Set cache parameters
    assert not args.load_cache
    cfg.train.load_cache = args.load_cache
    cfg.train.data_filter = args.data_filter_fn

    # Update environment sub-config
    for k in cfg["nusc"]:  # hardcoded for now! copy env-specific config to the global-level
        cfg[k] = cfg["nusc"][k]

    # Remove env keys if needed
    cfg.pop("nusc")
    cfg.pop("drivesim")
    cfg.pop("l5kit")

    # Load checkpoint YAMLs if specified
    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)

    # Lock config
    cfg.lock()

    # Run simulation
    stats, info, renderings, adjust_plans, trace = run_adv_simulation(
        eval_cfg=cfg,
        data_to_disk=True,
        render_to_video=args.render,
        import_scene_list_mode=args.scene_select_mode,
        visualize_diffusion=args.visualize_diffusion,
        guidance_params=args.guidance_params
    )

    print("Simulation completed successfully!")
    print(f"Results saved to: {cfg.results_dir}")


if __name__ == "__main__":
    main()