import pandas as pd
import json
import argparse

import pandas as pd
import json
import argparse
import numpy as np
import os
from pprint import pprint
import torch
import h5py
# from trajdata.simulation.sim_stats import calc_stats
import tbsim.utils.tensor_utils as TensorUtils
import pathlib
from pyemd import emd

from torch import Tensor
from typing import Dict

def calc_stats(
    positions: Tensor, heading: Tensor, dt: float, bins: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Calculate scene statistics for a simulated scene.

    Args:
        positions (Tensor): N x T x 2 tensor of agent positions (in world coordinates).
        heading (Tensor): N x T x 1 tensor of agent headings (in world coordinates).
        dt (float): The data's delta timestep.
        bins (Dict[str, Tensor]): A mapping from statistic name to a Tensor of bin edges.

    Returns:
        Dict[str, Tensor]: A mapping of value names to histograms.
    """

    velocity: Tensor = (
        torch.diff(
            positions,
            dim=1,
            prepend=positions[:, [0]] - (positions[:, [1]] - positions[:, [0]]),
        )
        / dt
    )
    velocity_norm: Tensor = torch.linalg.vector_norm(velocity, dim=-1)

    accel: Tensor = (
        torch.diff(
            velocity,
            dim=1,
            prepend=velocity[:, [0]] - (velocity[:, [1]] - velocity[:, [0]]),
        )
        / dt
    )
    accel_norm: Tensor = torch.linalg.vector_norm(accel, dim=-1)

    lon_acc: Tensor = accel_norm * torch.cos(heading.squeeze(-1))
    lat_acc: Tensor = accel_norm * torch.sin(heading.squeeze(-1))

    jerk: Tensor = (
        torch.diff(
            accel_norm,
            dim=1,
            prepend=accel_norm[:, [0]] - (accel_norm[:, [1]] - accel_norm[:, [0]]),
        )
        / dt
    )

    return {
        "velocity": torch.histogram(velocity_norm, bins["velocity"]),
        "lon_accel": torch.histogram(lon_acc, bins["lon_accel"]),
        "lat_accel": torch.histogram(lat_acc, bins["lat_accel"]),
        "jerk": torch.histogram(jerk, bins["jerk"]),
    }
    
def interpolate_nan(array, axis=1):
    # Define a function to replace NaN with the nearest neighbor
    def fill_nan(vector):
        valid_mask = ~np.isnan(vector)
        valid_data = vector[valid_mask]
        valid_index = np.nonzero(valid_mask)[0]
        
        # If the vector is entirely NaN, return it as is
        if len(valid_data) == 0:
            return vector

        # Fill NaN with the nearest valid entries
        interpolated = np.interp(np.arange(len(vector)), valid_index, valid_data)
        return interpolated

    # Apply the function along the specified axis
    return np.apply_along_axis(fill_nan, axis, array)



def extract_unique_parts(result_names):
    split_names = [name.split('_') for name in result_names]
    common_parts = set(split_names[0])
    for parts in split_names[1:]:
        common_parts.intersection_update(parts)
    unique_parts_list = []
    for parts in split_names:
        unique_parts = [part for part in parts if part not in common_parts]
        unique_parts_list.append('_'.join(unique_parts))
    return unique_parts_list

    
def calc_hist_distance(hist1, hist2, bin_edges):
    hist1 = np.asarray(hist1, dtype=float)
    hist2 = np.asarray(hist2, dtype=float)

    if hist1.ndim != 1 or hist2.ndim != 1:
        raise ValueError("hist1 and hist2 should be 1D arrays")

    # Normalization
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)
    
    bins = np.array(bin_edges)
    bins_dist = np.abs(bins[:, None] - bins[None, :])

    hist_dist = emd(hist1.copy(), hist2.copy(), bins_dist)

    # Calculate Jensen-Shannon Divergence
    hist_dist_jsd = js_divergence(hist1, hist2)
    return hist_dist,hist_dist_jsd

def js_divergence(hist1, hist2):
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate M
    M = 0.5 * (hist1 + hist2)

    # Calculate the Kullback-Leibler divergences
    kl1 = np.sum(hist1 * np.log((hist1 / M) + 1e-9))
    kl2 = np.sum(hist2 * np.log((hist2 / M) + 1e-9))

    # Calculate the Jensen-Shannon Divergence
    jsd = 0.5 * (kl1 + kl2)

    return jsd
def compute_and_save_stats(h5_path, mode ="tbsim", scene_keys_list = []):
    """Compute histogram statistics for a run"""
    
    h5f = h5py.File(h5_path, "r")
    bins = {
        "velocity": torch.linspace(0, 30, 21),
        "lon_accel": torch.linspace(0, 10, 21),
        "lat_accel": torch.linspace(0, 10, 21),
        "jerk": torch.linspace(0, 20, 21),
        "curvature": torch.linspace(-1,1,21)
    }
    sim_stats = dict()

    ticks = None
    dt = 0.1
    for i, scene_index in enumerate(h5f.keys()):
        scene_name = scene_index[:10]
        if len(scene_keys_list) > 0 and scene_name not in scene_keys_list:
            continue
        if i % 10 == 0:
            print(i)
        scene_data = h5f[scene_index]
        sim_pos = scene_data["centroid"]
        sim_yaw = np.array(scene_data["yaw"])[..., None]
        # sim_speed = scene_data["curr_speed"][:][:, None]
        if "dt" in scene_data:
            dt = scene_data["dt"][()]
        if mode =="tbsim":
            dt  = 0.5
            sim_pos = sim_pos[:,::5]
            sim_yaw = sim_yaw[:,::5]
            mask = np.array(scene_data["exceed_lane"])[...,None]
            mask = mask[:,::5] #to bool mask
            mask = mask.astype(bool)
             
            # Replace entries in sim_pos and sim_yaw with NaN where the mask is True
            sim_pos= np.where(mask, np.nan, sim_pos)  # Replace True values with np.nan
            sim_yaw[mask] = np.nan
        # !!neglect ego in the first index!
        sim_pos = sim_pos[1:]
        sim_yaw = sim_yaw[1:]
      
        
        sim = calc_stats(positions=torch.Tensor(sim_pos), heading=torch.Tensor(sim_yaw), dt=dt, bins=bins)

        for k in sim:
            if k not in sim_stats:
                sim_stats[k] = sim[k].hist.long()
            else:
                sim_stats[k] += sim[k].hist.long()

        if ticks is None:
            ticks = dict()
            for k in sim:
                ticks[k] = sim[k].bin_edges

    for k in sim_stats:
        sim_stats[k] = TensorUtils.to_numpy(sim_stats[k] / len(h5f.keys())).tolist()
    for k in ticks:
        ticks[k] = TensorUtils.to_numpy(ticks[k]).tolist()

    results_path = pathlib.Path(h5_path).parent.resolve()
    output_file = os.path.join(results_path, "hist_stats.json")
    json.dump({"stats": sim_stats, "ticks": ticks}, open(output_file, "w+"), indent=4)
    print("results dumped to {}".format(output_file))
    # Call the helper function to plot and save histograms
    # load GT histograms
    # real_histogram_file: str = "/net/acadia3a/data/wjchang/simulation_result/0918_replay_100GroundTruth/hist_stats.json"
    real_histogram_file: str = "/net/ca-home1/home/mai/wjchang/traffic-behavior-simulation/exp/GT_hist_stats_subsample.json"

    with open(real_histogram_file, 'r') as f:
        loaded_histograms = json.load(f)
    # --------calculate distribution metrics
    # hist_dist = calc_hist_distance(sim_stats["velocity"], loaded_histograms["stats"]["velocity"], ticks["velocity"])
    
    attributes = ['velocity', 'lon_accel', 'lat_accel', 'jerk']

    # Single dictionary to hold all the calculated distances
    all_distances = {}

    # Loop over each attribute to calculate the distances
    for attr in attributes:

        dist_was,dist_jsd = calc_hist_distance(sim_stats[attr], loaded_histograms["stats"][attr], ticks[attr])
        all_distances[f'was_{attr}'] = dist_was
        all_distances[f'js_{attr}'] = dist_jsd
        
    all_distances["realism"] =  np.mean([all_distances["was_lon_accel"], all_distances["was_lat_accel"], all_distances["was_jerk"]])
    print(all_distances)
    return all_distances
# rest of your functions remain unchanged
def compute_nuplan_metrics(h5_path, scene_keys_list = []):
    from strive_planner.nuplan_metrics import NuScenesMapEnv, compute_metrics
    from tbsim.configs.selected_scene_config import SCENE_TO_MAP_DICT
    from collections import defaultdict

    map_env = NuScenesMapEnv('/net/ca-home1/home/mai/wjchang/traffic-behavior-simulation/strive_planner/data',
                            bounds=[-17.0, -38.5, 60.0, 38.5],
                            L=224,
                            W=224,
                            layers=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                            device=torch.device('cpu'),
                            flip_singapore=False,
                            load_lanegraph=False,
                            lanegraph_res_meters=1.0,
                            pix_per_m=2 )
    
    #convert to strive? format:
    
    #input: map_name, ego_obs,agent_obs(x,y,hx,hy), extents ,attack index
    h5f = h5py.File(h5_path, "r")
     # Dictionary to store sum of metrics for computing mean later
    all_metrics = []

    #convert h5py to strive format
    for i, scene_index in enumerate(h5f.keys()):
        scene_name = scene_index[:10]
        if len(scene_keys_list) > 0 and scene_name not in scene_keys_list:
            continue
        if i % 10 == 0:
            print(i)
        scene_data = h5f[scene_index]
        sim_pos = np.array(scene_data["centroid"])
        sim_yaw = np.array(scene_data["yaw"])[...,None]
        extent  = np.array(scene_data["extent"])[...,:2]
        # sim_speed = scene_data["curr_speed"][:][:, None]
        if "dt" in scene_data:
            dt = scene_data["dt"][()]
        else:
            dt = 0.5
            sim_pos = sim_pos[:,::5]
            sim_yaw = sim_yaw[:,::5]
            mask = np.array(scene_data["exceed_lane"])[...,None]
            mask = mask[:,::5] #to bool mask
            mask = mask.astype(bool)
            # Replace entries in sim_pos and sim_yaw with NaN where the mask is True
            sim_pos= np.where(mask, np.nan, sim_pos)  # Replace True values with np.nan
            sim_yaw[mask] = np.nan
 
        
        scene_name = scene_index[:10]
        map_name = SCENE_TO_MAP_DICT[scene_name]
        map_idx = map_env.map_list.index(map_name)
       
        # convert sim_pos and sim_yaw to x,y,hx,hy
        fut_adv = np.concatenate((sim_pos,np.cos(sim_yaw),np.sin(sim_yaw)),axis = -1) # fut_adv is all vehicle's states
        # fut_adv = np.vstack([plan[None]]+[agent_obs[i][None] for i in range(len(agent_obs)) if not np.any(np.isnan(agent_obs[i]))])
        # if there is nan in fut_adv
        if np.any(np.isnan(fut_adv)):
            print("warning, Nan values, do interpolation")
            fut_adv = interpolate_nan(fut_adv)
        
        veh_att = extent
        veh_att = veh_att[:len(fut_adv)]
        if "ego" in scene_index:
            atk_agt_idx = extract_ctrl_index(scene_index)
            print(scene_index,atk_agt_idx)
            veh_att = veh_att[:,0] #no need to have T dimension for tbsim
        elif "atk_id" in scene_data:
            # if "singapore" in map_name
            atk_agt_idx = np.array(scene_data["atk_id"]).item()
            
        else:
            raise ValueError("should contain attacker")
        metrics, freq_metrics_cnt, freq_metrics_total, seq_metrics = compute_metrics(torch.from_numpy(veh_att), torch.from_numpy(fut_adv), atk_agt_idx, dt, map_env, map_idx, all_adv = True)

        for k,v in seq_metrics.items():
            print(k,v)
        if len(seq_metrics) == 0:
            continue
        all_metrics.append(np.array(list(seq_metrics.values())))

        # Close the HDF5 file
    h5f.close()
    
    # Convert list of arrays to a 2D NumPy array and compute mean across columns
    all_metrics_np = np.vstack(all_metrics)
    mean_metrics = np.nanmean(all_metrics_np, axis=0)
    
    # Create a dictionary mapping metric names to their means
    metric_names = list(seq_metrics.keys())
    mean_metrics_dict = dict(zip(metric_names, mean_metrics))
    mean_metrics_dict["num_scene"] = len(all_metrics)
    return mean_metrics_dict

def extract_ctrl_index(filename: str) -> int:
    import re
    # Pattern to find the control index in the file name
    ego_pattern = re.compile(r'_ego_(\d+)_')
    ctrl_pattern = re.compile(r'_ctrl_\[(\d+)\]_')
    
    # Search for matches in the filename
    ego_match = ego_pattern.search(filename)
    ctrl_match = ctrl_pattern.search(filename)
    
    if not (ego_match and ctrl_match):
        raise ValueError("Filename does not contain the required ego or ctrl indices.")

    # Extract the numerical indices
    ego_index = int(ego_match.group(1))
    ctrl_index = int(ctrl_match.group(1))

    # Adjust the ctrl_index only if it's less than the ego index
    adjusted_index = ctrl_index if ctrl_index > ego_index else ctrl_index + 1
    
    return adjusted_index