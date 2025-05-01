from logging import raiseExceptions
from signal import raise_signal
import torch
import torch.nn.functional as F
import numpy as np

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig
from trajdata.data_structures.state import StateTensor,StateArray
from typing import Dict

# Define the keywords needed for guidance
GUIDANCE_KEYWORDS = [
    "world_from_agent",
    "curr_speed",
    "yaw",
    "agent_fut_extent",
    "all_other_agents_types",
    "scene_index",
    "scene_ids",
    "extras",
    "centroid",
    "raster_from_agent"
]

def extract_data_batch_for_guidance(data_batch, curr_states=None):
    """Extract and reshape relevant data for guidance calculations"""
    data_batch_for_guidance = {}
    guidance_keywords_used = GUIDANCE_KEYWORDS 
    for k in guidance_keywords_used:
        if k == 'extras':
            if 'extras' in data_batch:
                data_batch_for_guidance['extras'] = {}
                for j in data_batch['extras']:
                    B, M = data_batch['extras'][j].shape[:2]
                    data_batch_for_guidance['extras'][j] = data_batch['extras'][j].reshape(
                        B*M, *data_batch['extras'][j].shape[2:]
                    )
        elif k == 'scene_index' and k not in data_batch and 'scene_ids' in data_batch:
            data_batch_for_guidance[k] = torch.tensor([
                int(scene_id.split("_")[-1]) for scene_id in data_batch["scene_ids"]
            ])
            B, A = data_batch["history_positions"].shape[:2]
            data_batch_for_guidance[k] = data_batch_for_guidance[k].unsqueeze(1).expand(B, A).reshape(B*A)
        else:
            if k == "raster_from_agent":
                data_batch_for_guidance[k] = data_batch[k]
            elif data_batch.get(k) is not None:
                B, M = data_batch[k].shape[:2]
                data_batch_for_guidance[k] = data_batch[k].reshape(B*M, *data_batch[k].shape[2:])
    
    if curr_states is not None:
        data_batch_for_guidance["curr_states"] = curr_states

    return data_batch_for_guidance

def repeat_for_samples(data_batch_for_guidance: Dict[str, torch.Tensor], num_samples: int) -> Dict[str, torch.Tensor]:
    """Repeat the guidance data for multiple samples"""
    repeated_data = {}
    
    for k, v in data_batch_for_guidance.items():
        if isinstance(v, torch.Tensor):
            if k == "raster_from_agent":
                repeated_data[k] = v
            else:
                # Repeat and reshape based on tensor dimensions
                repeated_data[k] = v.unsqueeze(1).repeat(1, num_samples, *([1] * (len(v.shape)-1))).reshape(
                    -1, *v.shape[1:]
                )
        elif k == "extras":
            repeated_data[k] = {}
            for j, val in v.items():
                if isinstance(val, torch.Tensor):
                    repeated_data[k][j] = val.unsqueeze(1).repeat(
                        1, num_samples, *([1] * (len(val.shape)-1))
                    ).reshape(-1, *val.shape[1:])
                else:
                    repeated_data[k][j] = val
        else:
            repeated_data[k] = v
            
    return repeated_data

def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    if isinstance(state,StateTensor):
        pos = state.position.as_tensor()
        yaw = state.heading
        speed = state.as_format("v_lon")[...,0].as_tensor()
    elif isinstance(state,StateArray):
        pos = state.position.as_ndarray()
        yaw = state.heading
        speed = state.as_format("v_lon")[...,0].as_ndarray()
    else:
        if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
            # state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
            yaw = state[...,6:7]
            speed = torch.norm(state[..., 2:4], dim=-1)
        elif state.shape[-1] == 8:
            yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
            speed = torch.norm(state[..., 2:4], dim=-1)
        else:
            assert state.shape[-1] == 9 #x,y,z,vx,vy,ax,ay,s,c
            yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
            speed = torch.norm(state[..., 3:5], dim=-1)
            # state = StateTensor.from_array(state,"x,y,z,xd,yd,xdd,ydd,s,c")
            # pos = state.position.as_tensor()
            # yaw = state.heading
            # speed = state.as_format("v_lon")[...,0].as_tensor()
        pos = state[..., :2]
        
       
    mask = torch.bitwise_not(torch.cat([pos,speed[...,None],yaw],-1).isnan().any(-1))
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def rasterize_agents_scene(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    
    b, a, t, _ = agent_hist_pos.shape
    _, _, _, h, w = maps.shape
    maps = maps.clone()
    agent_hist_pos = TensorUtils.unsqueeze_expand_at(agent_hist_pos,a,1)
    agent_mask_tiled = TensorUtils.unsqueeze_expand_at(agent_mask,a,1)*TensorUtils.unsqueeze_expand_at(agent_mask,a,2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos.reshape(b*a,-1,2), raster_from_agent.reshape(b*a,3,3)).reshape(b,a,a,t,2)
    raster_hist_pos = raster_hist_pos * agent_mask_tiled.unsqueeze(-1)  # Set invalid positions to 0.0 Will correct below
    
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels [B, A, A, T, 2]
    raster_hist_pos = raster_hist_pos.transpose(2,3)
    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, A, T, A]
    hist_image = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    
    ego_mask = torch.zeros_like(raster_hist_pos_flat,dtype=torch.bool)
    ego_mask[:,range(a),:,range(a)]=1
    agent_mask = torch.logical_not(ego_mask)


    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*agent_mask, src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*ego_mask, src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[..., 0] = 0  # correct the 0th index from invalid positions
    hist_image[..., -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, a, t, h, w)

    maps = torch.cat((hist_image, maps), dim=2)  # treat time as extra channels
    return maps


def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
        cat=True,
        filter=None,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels

    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, T, A]

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
    hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, t, h, w)
    if filter=="0.5-1-0.5":
        kernel = torch.tensor([[0.5, 0.5, 0.5],
                    [0.5, 1., 0.5],
                    [0.5, 0.5, 0.5]]).to(hist_image.device)

        kernel = kernel.view(1, 1, 3, 3).repeat(t, t, 1, 1)
        hist_image = F.conv2d(hist_image, kernel,padding=1)
    if cat:
        maps = maps.clone()
        maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
        return maps
    else:
        return hist_image

def rasterize_agents_rec(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
        cat=True,
        ego_neg = False,
        parallel_raster=False,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    with torch.no_grad():
        b, a, t, _ = agent_hist_pos.shape
        _, _, h, w = maps.shape
        
        coord_tensor = torch.cat((torch.arange(w).view(w,1,1).repeat_interleave(h,1),
                                torch.arange(h).view(1,h,1).repeat_interleave(w,0),),-1).to(maps.device)

        agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
        raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
        

        raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below

        raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
        
        raster_hist_pos_yx = torch.cat((raster_hist_pos[...,1:],raster_hist_pos[...,0:1]),-1)

        if parallel_raster:
        # vectorized version, uses much more memory
            coord_tensor_tiled = coord_tensor.view(1,1,1,h,w,-1).repeat(b,t,a,1,1,1)
            dyx = raster_hist_pos_yx[...,None,None,:]-coord_tensor_tiled
            cos_yaw = torch.cos(-agent_hist_yaw)
            sin_yaw = torch.sin(-agent_hist_yaw)

            rotM = torch.stack(
                [
                    torch.stack([cos_yaw, sin_yaw], dim=-1),
                    torch.stack([-sin_yaw, cos_yaw], dim=-1),
                ],dim=-2,
            )
            rotM = rotM.transpose(1,2)
            rel_yx = torch.matmul(rotM.unsqueeze(-3).repeat(1,1,1,h,w,1,1),dyx.unsqueeze(-1)).squeeze(-1)
            agent_extent_yx = torch.cat((agent_extent[...,1:2],agent_extent[...,0:1]),-1)
            extent_tiled = agent_extent_yx[:,None,:,None,None]
            
            flag = (torch.abs(rel_yx)<extent_tiled).all(-1).type(torch.int)

            agent_mask_tiled = agent_mask.transpose(1,2).type(torch.int)
            if ego_neg:
                # flip the value for ego
                agent_mask_tiled[:,:,0] = -agent_mask_tiled[:,:,0]
            hist_img = flag*agent_mask_tiled.view(b,t,a,1,1)
            
            if hist_img.shape[2]>1:
            # aggregate along the agent dimension
                hist_img = hist_img[:,:,0] + hist_img[:,:,1:].max(2)[0]*(hist_img[:,:,0]==0)
            else:
                hist_img = hist_img.squeeze(2)
        else:

        # loop through all agents, slow but memory efficient
            coord_tensor_tiled = coord_tensor.view(1,1,h,w,-1).repeat(b,t,1,1,1)
            agent_extent_yx = torch.cat((agent_extent[...,1:2],agent_extent[...,0:1]),-1)
            hist_img_ego = torch.zeros([b,t,h,w]).to(maps.device)
            hist_img_nb = torch.zeros([b,t,h,w]).to(maps.device)
            for i in range(raster_hist_pos_yx.shape[-2]):
                dyx = raster_hist_pos_yx[...,i,None,None,:]-coord_tensor_tiled
                yaw_i = agent_hist_yaw[:,i]
                cos_yaw = torch.cos(-yaw_i)
                sin_yaw = torch.sin(-yaw_i)

                rotM = torch.stack(
                    [
                        torch.stack([cos_yaw, sin_yaw], dim=-1),
                        torch.stack([-sin_yaw, cos_yaw], dim=-1),
                    ],dim=-2,
                )
                
                rel_yx = torch.matmul(rotM.unsqueeze(-3).repeat(1,1,h,w,1,1),dyx.unsqueeze(-1)).squeeze(-1)
                extent_tiled = agent_extent_yx[:,None,i,None,None]
                
                flag = (torch.abs(rel_yx)<extent_tiled).all(-1).type(torch.int)
                if i==0:
                    if ego_neg:
                        hist_img_ego = -flag*agent_mask[:,0,:,None,None]
                    else:
                        hist_img_ego = flag*agent_mask[:,0,:,None,None]
                else:
                    hist_img_nb = torch.maximum(hist_img_nb,agent_mask[:,0,:,None,None]*flag)
                
            if a>1:
                hist_img = hist_img_ego + hist_img_nb*(hist_img_ego==0)
            else:
                hist_img = hist_img_ego

        if cat:
            maps = maps.clone()
            maps = torch.cat((hist_img, maps), dim=1)  # treat time as extra channels
            return maps
        else:
            return hist_img



def get_drivable_region_map(maps):
    if isinstance(maps, torch.Tensor):
        drivable = torch.amax(maps[..., -3:, :, :], dim=-3).bool()
    else:
        drivable = np.amax(maps[..., -3:, :, :], axis=-3).astype(bool)
    return drivable



def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch["agent_hist"].shape[1]
    fut_len = batch["agent_fut"].shape[1]
    b, a, neigh_len, _ = batch["neigh_hist"].shape
    empty_neighbor = a == 0
    if empty_neighbor:
        batch["neigh_hist"] = torch.ones(b, 1, hist_len, batch["neigh_hist"].shape[-1]) * torch.nan
        batch["neigh_fut"] = torch.ones(b, 1, fut_len, batch["neigh_fut"].shape[-1]) * torch.nan
        batch["neigh_types"] = torch.zeros(b, 1)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, batch["neigh_hist_extents"].shape[-1])
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, batch["neigh_hist_extents"].shape[-1])
    elif neigh_len < hist_len:
        # Create the padding tensor with NaNs and set the device
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch["neigh_hist"].shape[-1], device=batch["neigh_hist"].device) * torch.nan
        # Concatenate tensors
        batch["neigh_hist"] = torch.cat((hist_pad, batch["neigh_hist"]), dim=2)
        # Create the padding tensor with zeros and set the device
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch["neigh_hist_extents"].shape[-1], device=batch["neigh_hist_extents"].device)* torch.nan
        # Concatenate tensors
        batch["neigh_hist_extents"] = torch.cat((hist_pad, batch["neigh_hist_extents"]), dim=2)

def parse_scene_centric(batch: dict, rasterize_mode:str):
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])

    curr_pos = hist_pos[:,:,-1]
    curr_yaw = hist_yaw[:,:,-1]
    assert isinstance(batch["centered_agent_state"],StateTensor) or isinstance(batch["centered_agent_state"],StateArray)


    curr_speed = hist_speed[..., -1]
    centered_state = batch["centered_agent_state"]
    centered_yaw = centered_state.heading[...,0]
    centered_pos = centered_state.position
    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type[agent_type < 0] = 0
    agent_type[agent_type == 1] = 3
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.


    centered_world_from_agent = torch.inverse(batch["centered_agent_from_world_tf"])



    # map-related
    if batch["maps"] is not None:
        map_res = batch["maps_resolution"][0,0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        
        centered_raster_from_agent = torch.Tensor([
            [map_res, 0, 0.25 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).to(centered_state.device)
        b,a = curr_yaw.shape[:2]
        centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
        
        agents_from_center = (GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros(b*a,2,device=curr_yaw.device))
                                @GeoUtils.transform_matrices(torch.zeros(b*a,device=curr_yaw.device),-curr_pos.reshape(-1,2))).reshape(*curr_yaw.shape[:2],3,3)
        center_from_agents = GeoUtils.transform_matrices(curr_yaw.flatten(),curr_pos.reshape(-1,2)).reshape(*curr_yaw.shape[:2],3,3)
        raster_from_center = centered_raster_from_agent @ agents_from_center
        center_from_raster = center_from_agents @ centered_agent_from_raster

        raster_from_world = batch["rasters_from_world_tf"]
        world_from_raster,_ = torch.linalg.inv_ex(raster_from_world)
        raster_from_world[torch.isnan(raster_from_world)] = 0.
        world_from_raster[torch.isnan(world_from_raster)] = 0.
        
        if rasterize_mode=="none":
            maps = batch["maps"]
        elif rasterize_mode=="point":
            maps = rasterize_agents_scene(
                batch["maps"],
                hist_pos,
                hist_yaw,
                None,
                hist_mask,
                raster_from_center,
                map_res
            )
        elif rasterize_mode=="square":
            #TODO: add the square rasterization function for scene-centric data
            raise NotImplementedError
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0


    d = dict(
        image=maps,
        map_names = batch["map_names"],
        drivable_map=drivable_map,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=centered_pos,
        yaw=centered_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=centered_raster_from_agent,
        agent_from_raster=centered_agent_from_raster,
        raster_from_center=raster_from_center,
        center_from_raster=center_from_raster,
        agents_from_center = agents_from_center,
        center_from_agents = center_from_agents,
        raster_from_world=raster_from_world,
        agent_from_world=batch["centered_agent_from_world_tf"],
        world_from_agent=centered_world_from_agent,
    )

    return d 

def parse_node_centric(batch: dict,rasterize_mode:str):
    maybe_pad_neighbor(batch)
    fut_pos, fut_yaw, fut_speed, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]
    assert isinstance(curr_state,StateTensor) or isinstance(curr_state,StateArray)
    curr_yaw = curr_state.heading[...,0]
    curr_pos = curr_state.position

    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type[agent_type < 0] = 0
    agent_type[agent_type == 1] = 3
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

    neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = trajdata2posyawspeed(batch["neigh_hist"])
    neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = trajdata2posyawspeed(batch["neigh_fut"])
    neigh_curr_speed = neigh_hist_speed[..., -1]
    neigh_types = batch["neigh_types"]
    # convert nuscenes types to l5kit types
    neigh_types[neigh_types < 0] = 0
    neigh_types[neigh_types == 1] = 3
    # mask out invalid extents
    neigh_hist_extents = batch["neigh_hist_extents"]
    neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

    world_from_agents = torch.inverse(batch["agents_from_world_tf"])


    # map-related
    if batch["maps"] is not None and batch["maps"].nelement() > 0:
        map_res = batch["maps_resolution"][0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        raster_from_agent = torch.Tensor([
            [map_res, 0, 0.25 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).to(curr_state.device)
        agent_from_raster = torch.inverse(raster_from_agent)
        raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=batch["maps"].shape[0], dim=0)
        agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=batch["maps"].shape[0], dim=0)
        raster_from_world = torch.bmm(raster_from_agent, batch["agents_from_world_tf"])
        all_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos.to(curr_state.device)), dim=1)
        all_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw.to(curr_state.device)), dim=1)

        all_extents = torch.cat((batch["agent_hist_extent"].unsqueeze(1),batch["neigh_hist_extents"].to(curr_state.device)),1).max(dim=2)[0][...,:2]
        all_hist_mask = torch.cat((hist_mask[:, None], neigh_hist_mask.to(curr_state.device)), dim=1)
        if rasterize_mode=="none":
            maps = batch["maps"]
        elif rasterize_mode=="point":
                maps = rasterize_agents(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_extents,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
        elif rasterize_mode=="square":
            maps = rasterize_agents_rec(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_extents,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
        else:
            raise Exception("unknown rasterization mode")
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0
    filter_lane:bool = True if "has_lane" in batch["extras"] else False
    d = dict(
        maps= batch["maps"],
        image=maps,
        drivable_map=drivable_map,
        map_names = batch["map_names"],
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_speeds=fut_speed,
        target_availabilities=batch["extras"]["has_lane"][:,None]*fut_mask if filter_lane else fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=curr_pos,
        yaw=curr_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=raster_from_agent,
        agent_from_raster=agent_from_raster,
        raster_from_world=raster_from_world,
        agent_from_world=batch["agents_from_world_tf"],
        world_from_agent=world_from_agents,
        all_other_agents_history_positions=neigh_hist_pos,
        all_other_agents_history_yaws=neigh_hist_yaw,
        all_other_agents_history_availability=neigh_hist_mask,
        all_other_agents_history_availabilities=neigh_hist_mask,  # dump hack to agree with l5kit's typo ...
        all_other_agents_curr_speed=neigh_curr_speed,
        all_other_agents_future_positions=neigh_fut_pos,
        all_other_agents_future_yaws=neigh_fut_yaw,
        all_other_agents_future_availability=neigh_fut_mask,
        all_other_agents_types=neigh_types,
        all_other_agents_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
        all_other_agents_history_extents=neigh_hist_extents * extent_scale,
    )
    if "agent_lanes" in batch:
        d["ego_lanes"] = batch["agent_lanes"]
    
    return d

@torch.no_grad()
def parse_trajdata_batch(batch: dict,rasterize_mode="point"):
    
    if "num_agents" in batch:
        # scene centric
        d = parse_scene_centric(batch,rasterize_mode)
        
    else:
        # agent centric
        d = parse_node_centric(batch,rasterize_mode)

    batch = dict(batch)
    batch.update(d)
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            batch[k]=v.nan_to_num(0)
    batch.pop("agent_name", None)
    batch.pop("robot_fut", None)
    return batch


def get_modality_shapes(cfg: ExperimentConfig,rasterize_mode:str="point"):
    h = cfg.env.rasterizer.raster_size
    if rasterize_mode=="none":
        return dict(static=(3,h,h),dynamic=(0,h,h),image=(3,h,h))
    else:
        num_channels = (cfg.algo.history_num_frames + 1) + 3
        return dict(static=(3,h,h),dynamic=(cfg.algo.history_num_frames + 1,h,h),image=(num_channels, h, h))
        

def locate_traffic_lights(maps_tensor, tolerance=20):
    """
    Locate traffic lights on a rasterized map with a given tolerance using PyTorch operations.
    
    Parameters:
    - maps_tensor: A 4D PyTorch tensor (Batch, 3, Height, Width) representing the RGB rasterized map
    - tolerance: A value representing the allowed deviation from the exact RGB values
    
    Returns:
    - A dictionary with traffic light statuses as keys and tensors of traffic light positions as values
    """
    
    traffic_light_colors = {
        # "GREEN": torch.tensor([0, 200, 0]),
        "RED": torch.tensor([200, 0, 0]),
        # "UNKNOWN": torch.tensor([150, 150, 0])
    }

    traffic_light_positions = {}

    for status, color in traffic_light_colors.items():
        lower_bounds = (maps_tensor >= color[:, None, None] - tolerance)
        upper_bounds = (maps_tensor <= color[:, None, None] + tolerance)
        
        # Create a boolean mask where each RGB channel is within the tolerance range
        mask = lower_bounds & upper_bounds
        
        # Combine the RGB channel information
        combined_mask = mask.all(dim=1)
        
        # Store in the dictionary
        traffic_light_positions[status] = combined_mask

    return traffic_light_positions

def enlarge_batch_samples(arr: torch.Tensor, batch_size: int, num_samples: int = None) -> torch.Tensor:
    """
    Enlarge array shape based on batch size and number of samples.
    If num_samples is None, return original array.
    
    Args:
        arr: Input tensor to be enlarged
        batch_size: Expected batch size
        num_samples: Number of samples to repeat (optional)
    
    Returns:
        Enlarged tensor if num_samples is provided, otherwise original tensor
    """
    if num_samples is None or arr.shape[0] == batch_size * num_samples:
        return arr
        
    if arr.shape[0] != batch_size:
        raise ValueError(f"Array batch size {arr.shape[0]} doesn't match expected batch size {batch_size}")
    
    # Create the expanded shape based on batch_size and sample
    expanded_shape = [batch_size * num_samples] + list(arr.shape[1:])
    
    # Enlarge array
    enlarged_arr = arr.unsqueeze(1).repeat(1, num_samples, *([1] * len(arr.shape[1:]))).reshape(expanded_shape)
    
    return enlarged_arr
def enlarge_batch_shape(target_arr, other_arr):
    ''' if the target_arr is batch_size*sample, expand other_arr with sample, and then reshape
    '''
    if target_arr.shape[0] ==other_arr.shape[0]:
        return other_arr
    batch_size = other_arr.shape[0]
    sample = target_arr.shape[0] // batch_size

    # Get the shape dimensions of other_arr
    other_shape = other_arr.shape

    # Create the expanded shape based on batch_size and sample
    expanded_shape = [batch_size * sample] + list(other_shape[1:])

    # Enlarge other_arr
    enlarged_other_arr = other_arr.unsqueeze(1).repeat(1, sample, *([1] * len(other_shape[1:]))).reshape(expanded_shape)

    return enlarged_other_arr
def calculate_heading(centerline_xy):
    # Calculate the difference between consecutive points
    diff = centerline_xy[:, 1:, :] - centerline_xy[:, :-1, :]

    # Extract x and y differences
    diff_x = diff[..., 0]
    diff_y = diff[..., 1]

    # Calculate heading using atan2
    heading = torch.atan2(diff_y, diff_x)

    # Pad the heading tensor with zeros to match the original shape
    heading = torch.cat([heading,heading[:,-1:]], dim=1)

    return heading

def extend_ego_plan(ego_plan, target_length, dt=0.1):
    """
    Extend the ego plan to match the target length using the last observed velocity.

    :param ego_plan: The original ego plan trajectory.
    :param target_length: The target number of timesteps.
    :return: The extended ego plan.
    """
    # Calculate the last velocity from the ego plan
    last_velocity = (ego_plan[:, -1] - ego_plan[:, -2]) / dt

    # Number of additional points needed
    num_additional_points = target_length - ego_plan.shape[1]

    # Create a sequence of deltas to add
    deltas = (
        last_velocity.unsqueeze(1)
        * dt
        * torch.arange(1, num_additional_points + 1, device=ego_plan.device).unsqueeze(-1)
    )

    # Extend the ego plan
    extended_plan = torch.cat([ego_plan, ego_plan[:, -1:] + deltas.cumsum(dim=1)], dim=1)

    return extended_plan




#reference from https://github.com/NVlabs/CTG/blob/f916c008c3ecf2360bfa050639606eaab7c207f5/tbsim/utils/trajdata_utils.py
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from typing import Union

def get_full_fut_traj(element: Union[AgentBatchElement, SceneBatchElement]):
    """Get mask for moving agents.
    """
    fut_sec = 20
    dt = 0.1
    T = int(fut_sec / dt)
    if isinstance(element, AgentBatchElement):
        # (T, 8), (T)
        fut_traj, _ = element.get_agent_future(element.agent_info, (fut_sec, fut_sec))
        t, k = fut_traj.shape
        if T > t:
            pad = np.zeros((T-t, k))
            fut_traj = np.concatenate([fut_traj, pad], axis=0)
        else:
            fut_traj = fut_traj[:T]
    elif isinstance(element, SceneBatchElement):
        # (M, T, 8), (M, T)
        fut_traj, _, _ = element.get_agents_future((fut_sec, fut_sec), element.nearby_agents)
        fut_traj_list = []
        for arr in fut_traj:
            t, k = arr.shape
            if T > t:
                pad = np.zeros((T-t, k))
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:T]
            fut_traj_list.append(arr)
        fut_traj = np.stack(fut_traj_list, axis=0)
    else:
        raise ValueError(f"Unknown element type: {type(element)}")
    fut_traj = torch.as_tensor(fut_traj, dtype=torch.float)
    return fut_traj
def get_full_fut_valid(element: Union[AgentBatchElement, SceneBatchElement]):
    """Get mask for moving agents.
    """
    fut_sec = 20.0
    dt = 0.1
    T = int(fut_sec / dt)
    if isinstance(element, AgentBatchElement):
        # (T, 3)
        _, fut_valid = element.get_agent_future(element.agent_info, (fut_sec, fut_sec))
        t, k = fut_valid.shape
        if T > t:
            pad = np.zeros((T-t, k))
            fut_valid = np.concatenate([fut_valid, pad], axis=0)
        else:
            fut_valid = fut_valid[:T]
        # (T, 3) -> (T)
        fut_valid = fut_valid[...,0]
    elif isinstance(element, SceneBatchElement):
        # (M, T)
        _, fut_valid, _ = element.get_agents_future((fut_sec, fut_sec), element.nearby_agents)
        fut_valid_list = []
        for arr in fut_valid:
            t, k = arr.shape
            if T > t:
                pad = np.zeros((T-t, k))
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:T]
            fut_valid_list.append(arr)
        fut_valid = np.stack(fut_valid_list, axis=0)
        # (M, T, 3) -> (M, T)
        fut_valid = fut_valid[...,0]
    else:
        raise ValueError(f"Unknown element type: {type(element)}")
    fut_valid = torch.as_tensor(fut_valid, dtype=torch.float)
    return fut_valid

def get_stationary_mask(data_batch, disable_control_on_stationary, moving_speed_th = 0.3):
    '''
    This function is called when disable_control_on_stationary is not False.
    '''
    # (B, (M), T, 8)
    full_fut_speed = data_batch['extras']['full_fut_traj'][...,2]
    # (B, (M), T)
    full_fut_valid = data_batch['extras']['full_fut_valid']
    # mask out those stationary all the time in GT
    if 'any_speed' in disable_control_on_stationary:
        # (B, (M), T), (B, (M), T) -> (B, (M))
        moving_mask = ((full_fut_speed > moving_speed_th).to(torch.float) * full_fut_valid).sum(dim=-1) > 0
    # mask out those stationary at the first timestep in GT
    elif 'current_speed' in disable_control_on_stationary:
        # technically we are using one timestep from the current timestep
        # (B, (M), T), (B, (M), T) -> (B, (M))
        moving_mask = ((full_fut_speed[...,0] > moving_speed_th).to(torch.float) * full_fut_valid[...,0]) > 0
    else:
        moving_mask = torch.ones(*full_fut_valid.shape[:-1], dtype=torch.bool, device=full_fut_valid.device)
    stationary_mask = ~moving_mask
    # print('0 stationary_mask', stationary_mask)
    # mask out those not on lane (in parking lot)
    if 'on_lane' in disable_control_on_stationary:
        map_max_dist = 2.0
        # assumer vec map in agent-centric coordinates
        lane_points = data_batch['extras']['centerline_xy'].detach().clone()
        lane_points = torch.where(torch.isnan(lane_points), torch.tensor(10e8, dtype=lane_points.dtype, device=lane_points.device), lane_points)
        
        # (B, (M), S_seg, S_point, 3) -> (B, (M), S_seg, S_point)
        dist_to_lane = torch.norm(lane_points[...,:2], dim=-1)
        # (B, (M), S_seg, S_point) -> (B, (M), S_seg*S_point)
        # (B, (M), S_seg*S_point) -> (B, (M))
        parking_mask = dist_to_lane.min(dim=-1)[0] > map_max_dist
        #should also see the headings are aligned in that mimum part, if its not algined 
        centerline_heading = data_batch['extras']['init_centerline_heading'] #TODO: avoid explicct numbers here
        # parking_mask = parking_mask  | is_offroad_by_heading(centerline_heading,) | ~data_batch['extras']['has_lane']
        # print("dist_to_lane.min(dim=-1)[0]", dist_to_lane.min(dim=-1)[0])
        # print('parking_mask', parking_mask)
        stationary_mask = stationary_mask | parking_mask
    # This mode is in order to test the influence of fixing the center vehicle on other vehicles
    if 'center' in disable_control_on_stationary:
        # (B, M)
        if len(stationary_mask.shape) == 2:
            stationary_mask[:,0] = True
        else: # (B)
            stationary_mask[0] = True
    # print('1 stationary_mask', stationary_mask)
    return stationary_mask

def is_offroad_by_heading(centerline_heading, threshold_deg=45):
    """
    Determines if agents are off-road based on their centerline heading using
    trigonometric similarity.
    
    This method uses cosine similarity between the centerline heading and the
    reference direction (0 degrees) to determine if a vehicle is off-road.
    This approach naturally handles the periodic nature of angles without
    needing explicit handling of discontinuities.
    
    Args:
        centerline_heading (torch.Tensor): Centerline headings in radians [batch, num_points]
        threshold_deg (float): Maximum allowed angular deviation in degrees (default: 45)
    
    Returns:
        torch.Tensor: Boolean tensor indicating which agents are off-road [batch]
    """
    # Convert threshold to radians and compute its cosine
    threshold_rad = np.deg2rad(threshold_deg)
    threshold_cos = np.cos(threshold_rad)
    cos_similarity = torch.cos(centerline_heading)
    mean_cos_similarity = torch.mean(cos_similarity, dim=1)

    is_offroad = mean_cos_similarity < threshold_cos
    
    return is_offroad