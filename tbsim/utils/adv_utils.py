# Generating proposals for adversarial cars

import torch
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.agent_rel_classify import find_conflict_point

import numpy as np
from typing import List, Tuple

def generate_collision_paths(
    data_batch: dict,
    ego_indices: torch.Tensor,
    adv_indices: torch.Tensor, 
    desired_delta_s: torch.Tensor,
    normal_offsets: float,
    ref_idx: int = 0,
    max_interaction_dist: float = 50.0,
    samples: int = 1,
    T: int = 32,
    dt: float = 0.1
) -> dict:
    """
    Generate collision paths for adversarial vehicles in closed-loop scenarios.
    For each adversarial vehicle within interaction distance of ego, generates potential
    collision trajectories by targeting specific distance differences at conflict points.

    Args:
        data_batch: Dictionary containing scene data with keys:
            - centroid: Vehicle positions (B, 2)
            - extent: Vehicle dimensions (B, 2) 
            - yaw: Vehicle headings (B,)
            - curr_speed: Vehicle speeds (B,)
            - ego_plan: Ego vehicle trajectory plan
            - world_from_agent/agent_from_world: Coordinate transforms
            - extras: Additional data including centerlines
        ego_indices: One-hot tensor indicating ego vehicle indices
        adv_indices: One-hot tensor indicating adversarial vehicle indices
        desired_delta_s: Target distance differences at conflict points
        normal_offsets: Lateral offset from centerline
        ref_idx: Index of reference path to use (if multiple available)
        max_interaction_dist: Maximum distance to consider vehicle interactions
        samples: Number of trajectory samples to generate
        T: Number of timesteps in trajectory horizon
        dt: Time step duration

    Returns:
        dict: Adversarial proposals containing:
            - adv_proposals: Raw control actions (num_adv, samples, T, 2)
            - adv_proposals_states: Vehicle states (num_adv, samples, T, 4)
            - adv_idx: Indices of adversarial vehicles
            - padded_adv_collision_actions: Padded actions for batch processing
            - padded_adv_proposals_pos: Padded positions for batch processing
            - centerline: Updated centerlines
    """
    device = data_batch["centroid"].device
    B = len(ego_indices)
    
    # Convert one-hot indices to regular indices
    ego_indices = get_nonzero_idx_np(ego_indices)
    adv_indices = get_nonzero_idx_np(adv_indices)
    
    # Initialize output tensors
    adv_proposals_actions = torch.zeros(len(adv_indices), samples, T, 2).to(device)
    adv_proposals_states = torch.zeros(len(adv_indices), samples, T, 4).to(device)
    
    # Get coordinate transforms
    world_from_agent = data_batch["world_from_agent"]
    agent_from_world = data_batch["agent_from_world"]

    for i, (ego_idx, adv_idx) in enumerate(zip(ego_indices, adv_indices)):
        # Get ego and adversary states
        ego_centroid = data_batch["centroid"][ego_idx]
        adv_centroid = data_batch["centroid"][adv_idx]
        
        ego_extent = data_batch["extent"][ego_idx][:2]
        adv_extent = data_batch["extent"][adv_idx][:2]
        
        ego_yaw = data_batch["yaw"][ego_idx]
        adv_yaw = data_batch["yaw"][adv_idx]
        ego_speed = data_batch["curr_speed"][ego_idx]
        adv_speed = data_batch["curr_speed"][adv_idx]
        
        # Get and transform trajectories
        ego_plan = data_batch["ego_plan"][i]  # in ego centric
        world_ego_plan = transform_points_tensor(ego_plan.to(world_from_agent.device), 
                                               world_from_agent[ego_idx])

        # Get and transform centerlines
        ego_centerline = data_batch["extras"]["centerline_world_xy"][ego_idx]
        
        #select adv centerlines
        # Process reference paths for adversary
        num_poss_refs = int(data_batch["extras"]["num_poss_refs"][adv_idx].item())
            
        if num_poss_refs > 1 and ref_idx is not None:
            adv_all_pos_centerlines = data_batch["extras"]["all_poss_refs"][adv_idx][:num_poss_refs]
            ref_idx = min(ref_idx, num_poss_refs-1)
            adv_centerline_agent = adv_all_pos_centerlines[ref_idx]
            # data_batch["extras"]["centerline_xy"][adv_idx] = adv_centerline_agent
            # adv_all_pos_centerlines_world = transform_points_tensor(adv_all_pos_centerlines, world_from_agent[adv_idx])
            adv_centerline = transform_points_tensor(adv_centerline_agent, world_from_agent[adv_idx])
        
        else:
            adv_centerline = data_batch["extras"]["centerline_world_xy"][adv_idx]
            adv_centerline_agent = transform_points_tensor(
                data_batch["extras"]["centerline_xy"][adv_idx],
                agent_from_world[adv_idx]
            )

    
        # Find conflict points
        conflict_point, conflict_idx_ego, conflict_idx_adv = find_conflict_point(
            ego_centerline, adv_centerline
        )
        
        # Check if conflict point is ahead
        closest_idx_ego = find_current_lane_idx(ego_centerline, ego_centroid)
        closest_idx_adv = find_current_lane_idx(ego_centerline, adv_centroid)
        is_ego_before_conflict = closest_idx_ego < conflict_idx_ego
        is_adv_before_conflict = closest_idx_adv < conflict_idx_adv

        # Calculate distances to conflict point
        if (is_ego_before_conflict or is_adv_before_conflict) and conflict_point is not None:
            ego_fut2conflict = torch.norm(ego_centroid - conflict_point, dim=-1)
            ctrl_fut2conflict = torch.norm(adv_centroid - conflict_point, dim=-1)
        else:
            ego_fut2conflict = torch.zeros(1, device=device)
            ctrl_fut2conflict = torch.zeros(1, device=device)

        # Generate proposals if within interaction distance
        if torch.norm(ego_centroid - adv_centroid) < max_interaction_dist:
            adv_collision_state, adv_collision_actions = generate_conflict_pt_proposals(
                adv_centroid, adv_speed, adv_yaw,
                adv_centerline_agent,  # Plan in local frame
                ego_speed, world_ego_plan, conflict_point,
                delta_s=torch.abs(ego_fut2conflict - ctrl_fut2conflict),
                sample_num=50,  # TODO: use samples parameter
                world_from_agent_adv=world_from_agent[adv_idx],
                target_deltas=desired_delta_s,
                normal_offsets=normal_offsets,
                dt=dt, T=T
            )
            
            adv_proposals_actions[i] = adv_collision_actions[None]
            adv_proposals_states[i] = adv_collision_state[None]
            
            #Debug Visualization
    
    # Generate collision paths
    return  {
        "adv_proposals_dict": {
            "adv_proposals": adv_proposals_actions,
            "adv_proposals_states": adv_proposals_states,
            "adv_idx": adv_indices,
            "padded_adv_collision_actions": create_padded_tensor(B, adv_indices, adv_proposals_actions),
            "padded_adv_proposals_pos": create_padded_tensor(B, adv_indices, adv_proposals_states[...,:2])
        },
        # "centerline": data_batch["extras"]["centerline_xy"]
    }
    
    
def get_nonzero_idx_np(my_array):
    return np.nonzero(my_array)[0]

def create_padded_tensor(batch_size, adv_idx, adv_proposals):
    """
    Create a tensor for a given batch size, where all elements are zero-padded
    except for the indices specified by adv_idx, which are filled with adv_proposals.

    :param batch_size: int, the size of the batch.
    :param adv_idx: Tensor, the indices in the batch to fill with adv_proposals.
    :param adv_proposals: Tensor, the values to insert at indices specified by adv_idx.
    :return: Tensor, the resulting padded tensor.
    """
    # Determine the shape for the padded tensor
    padded_shape = (batch_size, *adv_proposals.shape[1:])
    # Initialize the tensor with zeros
    padded_tensor = torch.zeros(padded_shape, dtype=adv_proposals.dtype, device=adv_proposals.device)
    
    # Update the tensor with adv_proposals at adv_idx
    padded_tensor[adv_idx] = adv_proposals

    return padded_tensor
def actions_states_from_positions(position, adv_speed, dt=0.1):
    position = torch.tensor(position, dtype=torch.float32, device=adv_speed.device)
    adv_speed = torch.tensor([adv_speed], device=position.device)  # Ensure adv_speed is a tensor on the same device
    
    # Calculate speeds based on position differences and dt
    distances = torch.sqrt(torch.sum(torch.diff(position, dim=0) ** 2, dim=1))
    speeds = torch.cat((adv_speed, distances / dt))
    
    # Calculate accelerations
    accelerations = torch.diff(speeds) / dt
    accelerations = torch.cat((torch.tensor([0.0], device=position.device), accelerations))  # Assume initial acceleration is 0
    
    # Calculate heading angles
    dx = torch.diff(position[:, 0])
    dy = torch.diff(position[:, 1])
    thetas = torch.atan2(dy, dx)
    thetas = torch.cat((torch.tensor([0.0], device=position.device), thetas))  # Assume initial heading is 0
    
    # Calculate turning rates
    turning_rates = torch.diff(thetas) / dt
    turning_rates = torch.cat((torch.tensor([0.0], device=position.device), turning_rates))  # Assume initial turning rate is 0
    
    # Combine accelerations and turning rates as actions
    actions = torch.stack((accelerations, turning_rates), dim=1)
    
    # Formulate states [(x, y, v, theta)]
    states = torch.cat((position, speeds.unsqueeze(1), thetas.unsqueeze(1)), dim=1)
    
    return actions, states


def find_current_lane_idx(centerline, current_pos):
    """
    Find the index of the closest point on the centerline to the current position.
    """
    distances = torch.norm(centerline - current_pos.unsqueeze(0), dim=-1)
    return torch.argmin(distances)

def generate_straight_line_proposal(start_pos, start_speed, start_yaw, end_pos, end_speed, T=3.2):
    """
    Generate a straight-line trajectory proposal using PyTorch, aiming to reach a specified end state.

    :param start_pos: Starting position [x, y] as a tensor.
    :param start_speed: Starting speed in m/s as a scalar.
    :param start_yaw: Starting yaw in radians as a scalar.
    :param end_pos: End position [x, y] as a tensor.
    :param end_speed: Desired speed at end position in m/s as a scalar.
    :param T: Time in seconds to reach the end position.
    :return: State (x, y, v, theta) and actions (acc, turning_rate) for the proposal.
    """
    device = start_pos.device
    
    # Calculate direction to target and distance
    direction_vector = end_pos - start_pos
    direction_to_target = torch.atan2(direction_vector[1], direction_vector[0])
    
    # Initial turning needed to face the target
    initial_turning = direction_to_target - start_yaw
    initial_turning = (initial_turning + torch.pi) % (2 * torch.pi) - torch.pi
    
    # Calculate the required acceleration to reach the end speed from the start speed in time T
    required_acceleration = (end_speed - start_speed) / T
    
    # Time steps for the trajectory
    timesteps = torch.linspace(0, T, int(T * 10) , device=device)
    
    # Calculate trajectory points
    speeds = start_speed + required_acceleration * timesteps
    distances = start_speed * timesteps + 0.5 * required_acceleration * timesteps**2
    
    # Assume straight line, so no additional turning is required after the initial alignment
    turning_rate = torch.tensor(0, dtype=torch.float32, device=device)
    
    trajectory_x = start_pos[0] + distances * torch.cos(direction_to_target)
    trajectory_y = start_pos[1] + distances * torch.sin(direction_to_target)
    trajectory_theta = torch.full_like(trajectory_x, direction_to_target)  # Constant yaw across the trajectory
    
    # Compiling the state and action for each timestep
    states = torch.stack((trajectory_x, trajectory_y, speeds, trajectory_theta), dim=1)
    actions = torch.stack((torch.full_like(timesteps, required_acceleration), torch.full_like(timesteps, turning_rate)), dim=1)
    
    actions[0, 1 ] = initial_turning / 0.1
    return states, actions

# ---------------------IDM----------------------

def generate_conflict_pt_proposals(
    adv_pos, 
    adv_speed, 
    adv_yaw, 
    adv_centerline,
    ego_speed,
    ego_plan_world,
    conflict_pt,
    world_from_agent_adv,
    delta_s=0.0,
    sample_num=5,
    target_deltas=None,
    normal_offsets=0.0,
    dt=0.1,
    T=32,
):
    """
    Generate trajectory proposals for an adversarial vehicle to maintain a specific distance 
    from the ego vehicle at a conflict point.

    Args:
        adv_pos: Current position of adversarial vehicle (agent frame)
        adv_speed: Current speed of adversarial vehicle
        adv_yaw: Current yaw of adversarial vehicle
        adv_centerline: Centerline for adversarial vehicle (agent frame)
        ego_speed: Current speed of ego vehicle
        ego_plan_world: Planned trajectory of ego vehicle (world frame)
        conflict_pt: Point where vehicles might conflict (world frame)
        world_from_agent_adv: Transform from agent to world coordinates
        T: Time horizon for planning
        delta_s: Target distance difference
        sample_num: Number of trajectory samples to generate
        target_deltas: Target distance differences to maintain
        normal_offsets: Lateral offset from centerline

    Returns:
        tuple: (best_state, best_actions)
            - best_state: Selected trajectory state (pos, vel, heading)
            - best_actions: Control actions for selected trajectory
    """
    # 1. Setup IDM parameters for trajectory generation
    idm_params = {
        'v_0': torch.linspace(2.0, 10.0, sample_num, device=adv_pos.device),  # Desired velocities
        'a': 1.0,    # Maximum acceleration [m/s^2]
        'b': 3.0,    # Comfortable braking deceleration [m/s^2]
        's_0': 2,    # Minimum distance [m]
        'T': 1.0,    # Safe time headway [s]
        'delta': 4,  # Acceleration exponent
        'dt':dt,    # Time step
        "horizon":T
    }

    # 2. Generate trajectory proposals using IDM
    state, actions = idm_get_proposals(
        adv_pos, adv_speed, adv_yaw, adv_centerline,
        ego_speed, ego_plan_world, conflict_pt,
        delta_s, idm_params, normal_offsets
    )  # Returns: state (num_samples,T,4), actions (num_samples,T,2)

    # 3. Calculate distances to conflict point
    # Transform adversarial trajectories to world frame
    adv_plan_world = transform_points_tensor(state[...,:2], world_from_agent_adv)
    ego_plan_world = ego_plan_world.to(world_from_agent_adv.device)

    # Calculate distances for both vehicles
    adv2_conflict_pt = torch.norm(adv_plan_world - conflict_pt, dim=-1)  # (num_samples, T)
    ego2_conflict_pt = torch.norm(ego_plan_world - conflict_pt, dim=-1)  # (T,)

    # 4. Create masks for post-conflict points
    adv2_conflict_mask = batch_mask_distances_after_passing(adv2_conflict_pt)
    ego2_conflict_mask = batch_mask_distances_after_passing(ego2_conflict_pt[None])
    combined_mask = torch.logical_or(adv2_conflict_mask, ego2_conflict_mask)

    # 5. Calculate and evaluate distance differences
    # Compute difference in distances to conflict point
    adv_delta_s = adv2_conflict_pt - ego2_conflict_pt
    
    # Mask out points after either vehicle passes conflict point
    masked_adv_delta_s = torch.where(
        ~combined_mask,
        torch.tensor(float('nan'), device=adv_pos.device),
        adv_delta_s
    )

    # 6. Select best trajectory
    # Calculate mean deviation from target distance
    target_delta_s = torch.abs(masked_adv_delta_s - target_deltas).nanmean(dim=1)
    
    # Sort trajectories by deviation (ascending)
    _, idx = torch.sort(target_delta_s, dim=0)
    
    # Return best trajectory (lowest deviation)
    best_state = state[idx][0:1]
    best_actions = actions[idx][0:1]
    
    return best_state, best_actions

def idm_get_proposals(
        adv_pos: torch.Tensor,
        adv_speed: torch.Tensor,
        adv_yaw: torch.Tensor,
        adv_centerline: torch.Tensor,
        ego_speed: torch.Tensor,
        ego_plan: torch.Tensor,
        conflict_pt: torch.Tensor,
        delta_s: torch.Tensor,
        idm_params: dict,
        normal_offsets: float,
        distance_margin: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate trajectory proposals for an adversarial vehicle using IDM (Intelligent Driver Model).
    
    Args:
        adv_pos: Adversarial vehicle position (x,y)
        adv_speed: Adversarial vehicle speed
        adv_yaw: Adversarial vehicle yaw
        adv_centerline: Centerline points for adversarial vehicle
        ego_speed: Ego vehicle speed
        ego_plan: Ego vehicle planned trajectory
        conflict_pt: Point of potential conflict
        delta_s: Distance difference to maintain
        idm_params: IDM parameters dictionary
        normal_offsets: Lateral offset from centerline
        distance_margin: idm distance margin to encourage collision avoidance
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - states: Trajectory states (B,T,4) containing [x,y,v,yaw]
            - actions: Control actions (B,T,2) containing [acceleration, turning_rate]
    """
    # 1. Generate acceleration proposals
    relative_speed = adv_speed - ego_speed
    distance = delta_s + distance_margin
    
    # Get reference IDM acceleration and sample around it
    idm_acc = idm_acceleration(adv_speed, relative_speed, distance, idm_params)
    num_samples = len(idm_acc)
    acc_proposals = torch.linspace(-5.0, 10.0, num_samples, device=adv_pos.device)
    
    # Ensure accelerations don't lead to negative speeds within horizon
    min_acc = -adv_speed / (idm_params["horizon"]*idm_params["dt"])
    acc_proposals = torch.clamp(acc_proposals, min=min_acc)

    # 2. Generate trajectories using accelerations
    positions, speeds, yaws = acc_follow_centerline2traj_simple(
        adv_centerline,
        adv_speed,
        acc_proposals,
        idm_params["dt"],
        horizon=idm_params["horizon"],
        normal_offsets=normal_offsets
    )

    # 3. Process trajectories and compute control actions
    # Add zero position at start for computing initial differences
    padded_positions = torch.cat([
        torch.zeros_like(positions[:,:1]),
        positions
    ], dim=1)
    
    # Smooth initial trajectory points for better continuity from vehicle to lanes
    for i in range(1, 5):
        padded_positions[:, i] = (padded_positions[:, 0] + padded_positions[:, 5]) * (i / 5)
    
    # Calculate turning rates from smoothed position differences
    position_diff = padded_positions[:, 1:] - padded_positions[:, :-1]
    yaws = torch.atan2(position_diff[..., 1], position_diff[..., 0])
    yaws = torch.cat([torch.zeros_like(yaws[:,:1]), yaws], dim=1)
    turning_rates = (yaws[:, 1:] - yaws[:, :-1]) / idm_params["dt"]
    
    # 4. Combine into final states and actions
    acc_broadcast = torch.ones_like(turning_rates) * acc_proposals.unsqueeze(-1)
    actions = torch.stack([acc_broadcast, turning_rates], dim=-1)
    states = torch.cat([
        padded_positions[:, 1:],  # Skip first position (zero padding)
        speeds,
        yaws[..., 1:, None]  # Skip first yaw (zero padding)
    ], dim=-1)

    return states, actions
   
#purely differential will not work, since it is not a continuous trajectory
def plan_turning_rate(current_position, current_yaw, new_position, dt):
    # Calculate direction vectors towards the target positions
    B = new_position.shape[0]
    current_yaw = torch.tensor([current_yaw], device=current_position.device)
    
    # cat the curren_pos and future_pos
    current_position = current_position.unsqueeze(0).repeat(B, 1, 1)
    new_position = torch.cat([current_position, new_position], dim=1)
    
    
    
    position_diff = new_position[:, 1:] - new_position[:, :-1]
    # Calculate the desired yaw angles from direction vectors
    
    theta_desired = torch.atan2(position_diff[...,1], position_diff[...,0])# Shape: B, steps
    
    # Calculate yaw differences between consecutive steps
    theta_diff = torch.diff(theta_desired, dim=1, prepend=current_yaw.unsqueeze(1).repeat(B,1))  # Prepend current_yaw for the initial step
    
    # Normalize yaw differences to the range [-π, π]
    theta_diff = torch.atan2(torch.sin(theta_diff), torch.cos(theta_diff))
    
    # Calculate turning rate as the difference in desired yaw angles over time
    turning_rate = theta_diff / dt  # Shape: B, steps
    
    return turning_rate
import numpy as np
from scipy.interpolate import CubicSpline

def fit_trajectory_and_calculate_turning_rate(waypoints, speed, dt = 0.1):
    """
    Fits a trajectory to given waypoints and calculates the turning rate at each point.

    Parameters:
    - waypoints: A numpy array of shape (T+1, 2), where N is the number of waypoints.
    - Speed: A numpy array of velocities at each waypoint of shape (T,).
    - dt: Time step between waypoints.

    Returns:
    - turning_rates: A numpy array of turning rates at each waypoint of shape (T,).
    """
    waypoints = waypoints.detach().cpu().numpy()
    speed = speed.detach().cpu().numpy().squeeze()
    
    # Split waypoints into x and y coordinates
        # Split waypoints into x and y coordinates
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    
    # Fit cubic splines to x and y coordinates
    spline_x = CubicSpline(np.linspace(0, (len(x)-1)*dt, len(x)), x, bc_type='natural')
    spline_y = CubicSpline(np.linspace(0, (len(y)-1)*dt, len(y)), y, bc_type='natural')
    
    # Calculate first and second derivatives of the splines
    dx = spline_x.derivative()(np.linspace(0, (len(x)-1)*dt, len(x)))
    dy = spline_y.derivative()(np.linspace(0, (len(y)-1)*dt, len(y)))
    ddx = spline_x.derivative(2)(np.linspace(0, (len(x)-1)*dt, len(x)))
    ddy = spline_y.derivative(2)(np.linspace(0, (len(y)-1)*dt, len(y)))
    
    # Calculate curvature
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # Calculate turning rate, omega = velocity * curvature
    turning_rates = speed * curvature
    
    return turning_rates[:-1]

def idm_acceleration(ego_velocity, delta_v, s,  idm_params):
    """Calculate the IDM acceleration using parameters from a dictionary."""
    s_star = idm_params['s_0'] + ego_velocity * idm_params['T'] + (ego_velocity * delta_v) / (2 * (idm_params['a'] * idm_params['b'])**0.5)
    return idm_params['a'] * (1 - torch.pow(ego_velocity / idm_params['v_0'], idm_params['delta']) - torch.pow(s_star / s, 2))

def acc_follow_centerline2traj(centerline_traj, current_speed, accel, delta_t, horizon, normal_offsets):
    '''
    centerline_traj: (1, N, 3) tensor
    current_speed: (1, 1) tensor
    accel: (B,) tensor
    normal_offsets: scalar
    '''
    B = accel.shape[0]
    centerline_traj = centerline_traj.unsqueeze(0).repeat(B, 1, 1)
    current_speed = current_speed.unsqueeze(0).repeat(B, 1)
   
    
    centerline_differences = centerline_traj[:, 1:, :2] - centerline_traj[:, :-1, :2]
    centerline_headings = torch.atan2(centerline_differences[..., 1], centerline_differences[..., 0])
    centerline_headings = torch.cat((centerline_headings, centerline_headings[:, -1].unsqueeze(1)), dim=1)

    normal_vectors_x = torch.cos(centerline_headings + torch.pi / 2)
    normal_vectors_y = torch.sin(centerline_headings + torch.pi / 2)
    # Shape of normal_vectors will be (B, N, 2)
    normal_vectors = torch.stack([normal_vectors_x, normal_vectors_y], dim=-1)
    

    centerline_traj = torch.cat([centerline_traj, centerline_headings.unsqueeze(-1)], dim=-1)
    
    # Initialization
    positions = torch.zeros([B, horizon, 2], dtype=torch.float32).to(accel.device)
    yaws = torch.zeros([B, horizon], dtype=torch.float32).to(accel.device)
    
    # 1. Find the closest centerline point to the ego car
    distances_to_origin = torch.norm(centerline_traj[:, :, :2], dim=-1)
    _, start_indices = torch.min(distances_to_origin, dim=1)
    
    # Calculate velocities for the entire horizon
    time_stamps = torch.arange(horizon, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(accel.device).repeat(B, 1, 1)
    # velocities = current_speed.unsqueeze(1) + accel.unsqueeze(1) * delta_t * time_stamps
    speed = torch.clamp(current_speed.unsqueeze(1) + accel.unsqueeze(1).unsqueeze(2) * delta_t * time_stamps, min=0)

    # Calculate differences and segment lengths
    diffs = centerline_traj[:, 1:, :2] - centerline_traj[:, :-1, :2]
    segment_lengths = torch.norm(diffs, dim=-1)
    
    # Create a mask with ones from start_idx onwards and zeros before that, for each batch item.
    mask = torch.arange(segment_lengths.shape[1]).to(segment_lengths.device).expand(segment_lengths.shape)
    mask = (mask >= start_indices.unsqueeze(-1)).float()

    # Calculate cumulative lengths for each batch item
    cum_lengths_from_start = mask * torch.cumsum(segment_lengths * mask, dim=1)
    # Expected distance covered by ego vehicle over the horizon
    distances = torch.cumsum(speed[:, :, 0] * delta_t, dim=1)
    
    # Map these distances onto the centerline
    segment_indices = (cum_lengths_from_start.unsqueeze(1) <= distances.unsqueeze(2)).sum(dim=-1) - 1
    segment_indices = torch.clamp(segment_indices, 0, segment_lengths.shape[1]-1)

    # Create a mask where the segment index is the last one
    last_idx_mask = (segment_indices == segment_lengths.shape[1]-1).unsqueeze(-1)
    
    # Mask to avoid cumsum across the horizon where last point is reached
    cumsum_mask = 1 - torch.cumsum(last_idx_mask, dim=1)
    
    # Retrieve the yaw values of the corresponding segments
    selected_yaws = torch.gather(centerline_traj[:, :-1, 2], 1, segment_indices)
    dx = speed[:, :, 0] * delta_t * torch.cos(selected_yaws)
    dy = speed[:, :, 0] * delta_t * torch.sin(selected_yaws)

    # Apply the cumsum mask
    dx *= cumsum_mask.squeeze()
    dy *= cumsum_mask.squeeze()

    # Cumulative sum
    positions[:,:,0] = torch.cumsum(dx, dim=1)
    positions[:,:,1] = torch.cumsum(dy, dim=1)
    # ----------------------normal offsets -----------------------

    # Assuming normal_offsets is a scalar, e.g., torch.tensor([offset_value])
    normal_offsets = torch.tensor([normal_offsets], device=accel.device, dtype=torch.float32)

    # Since normal_offsets is a scalar, we can directly use it in arithmetic operations with broadcasting
    # Ensure normal_vectors are correctly shaped for operation

    # Preparing segment_indices for gathering with expected shape adjustment
    segment_indices_expanded = segment_indices.unsqueeze(-1).expand(-1, -1, 2)

    # Correct selection of normals; assuming shape correction for segment_indices and normal_vectors as needed
    selected_normals = torch.gather(normal_vectors, 1, segment_indices_expanded)

    # Multiply selected normal vectors by the normal offset
    # Broadcasting of scalar normal_offsets to the shape of selected_normals during multiplication
    offset_vectors = selected_normals * normal_offsets
    
    positions += offset_vectors
    
    yaws = selected_yaws
    # Create masked values for the last centerline point
    last_positions = centerline_traj[:, -1, :2].unsqueeze(1).expand(-1, horizon, -1)
    last_yaws = centerline_traj[:, -1, 2].unsqueeze(1).expand(-1, horizon)

    # Update positions and yaws based on the mask
    positions[last_idx_mask.expand_as(positions)] = last_positions[last_idx_mask.expand_as(last_positions)]
    yaws[last_idx_mask.squeeze(-1)] = last_yaws[last_idx_mask.squeeze(-1)]

    
    return positions, speed, yaws
def acc_follow_centerline2traj_simple(centerline_traj, current_speed, accel, delta_t, horizon, normal_offsets):
    '''
    centerline_traj: (1, N, 2) tensor
    current_speed: (1, 1) tensor
    accel: (B,) tensor
    normal_offsets: scalar
    #NOTE that this is not planned for batch anymore, its assumed that all centerline_traj are same acrros batch
    '''
    B = accel.shape[0]
    centerline_traj = centerline_traj.unsqueeze(0).repeat(B, 1, 1)
    current_speed = current_speed.unsqueeze(0).repeat(B, 1)
    
    centerline_differences = centerline_traj[:, 1:, :2] - centerline_traj[:, :-1, :2]
    centerline_headings = torch.atan2(centerline_differences[..., 1], centerline_differences[..., 0])
    centerline_headings = torch.cat((centerline_headings, centerline_headings[:, -1].unsqueeze(1)), dim=1)

    normal_vectors_x = torch.cos(centerline_headings + torch.pi / 2)
    normal_vectors_y = torch.sin(centerline_headings + torch.pi / 2)
    # Shape of normal_vectors will be (B, N, 2)
    normal_vectors = torch.stack([normal_vectors_x, normal_vectors_y], dim=-1)
    
    centerline_traj = torch.cat([centerline_traj, centerline_headings.unsqueeze(-1)], dim=-1)
    
    # Initialization
    positions = torch.zeros([B, horizon, 2], dtype=torch.float32).to(accel.device)
    yaws = torch.zeros([B, horizon], dtype=torch.float32).to(accel.device)
    
    # 1. Find the closest centerline point to the ego car
    distances_to_origin = torch.norm(centerline_traj[:, :, :2], dim=-1)
    _, start_indices = torch.min(distances_to_origin, dim=1)
    
    #start from the centerline close to curr point
    # centerline_traj = centerline_traj[:,start_indices[0]:]
    # Calculate velocities for the entire horizon
    time_stamps = torch.arange(horizon, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(accel.device).repeat(B, 1, 1)
    # velocities = current_speed.unsqueeze(1) + accel.unsqueeze(1) * delta_t * time_stamps
    speed = torch.clamp(current_speed.unsqueeze(1) + accel.unsqueeze(1).unsqueeze(2) * delta_t * time_stamps, min=0)

    # Calculate differences and segment lengths
    diffs = centerline_traj[:, 1:, :2] - centerline_traj[:, :-1, :2]
    segment_lengths = torch.norm(diffs, dim=-1)
    
    # Create a mask with ones from start_idx onwards and zeros before that, for each batch item.
    mask = torch.arange(segment_lengths.shape[1]).to(segment_lengths.device).expand(segment_lengths.shape)
    mask = (mask >= start_indices.unsqueeze(-1)).float()

    # Calculate cumulative lengths for each batch item
    cum_lengths_from_start = mask * torch.cumsum(segment_lengths * mask, dim=1)
    # Expected distance covered by ego vehicle over the horizon
    distances = torch.cumsum(speed[:, :, 0] * delta_t, dim=1)
    
    # Map these distances onto the centerline
    segment_indices = (cum_lengths_from_start.unsqueeze(1) <= distances.unsqueeze(2)).sum(dim=-1) - 1
    segment_indices = torch.clamp(segment_indices, 0, segment_lengths.shape[1]-1)

    # Create a mask where the segment index is the last one
    last_idx_mask = (segment_indices == segment_lengths.shape[1]-1).unsqueeze(-1)
    
    # Mask to avoid cumsum across the horizon where last point is reached
    cumsum_mask = 1 - torch.cumsum(last_idx_mask, dim=1)
    
    # Retrieve the yaw values of the corresponding segments
    selected_yaws = torch.gather(centerline_traj[:, :-1, 2], 1, segment_indices)
    dx = speed[:, :, 0] * delta_t * torch.cos(selected_yaws)
    dy = speed[:, :, 0] * delta_t * torch.sin(selected_yaws)

    # Apply the cumsum mask
    dx *= cumsum_mask.squeeze() 
    dy *= cumsum_mask.squeeze() 

    # Cumulative sum ( this is delta pos along the centerline)
    positions[:,:,0] = torch.cumsum(dx, dim=1)
    positions[:,:,1] = torch.cumsum(dy, dim=1)
    #Plus the current point should be the actual position
    positions += centerline_traj[:, start_indices[0], :2].unsqueeze(1)
    # ----------------------normal offsets -----------------------

    # Assuming normal_offsets is a scalar, e.g., torch.tensor([offset_value])
    normal_offsets = torch.tensor([normal_offsets], device=accel.device, dtype=torch.float32)

    # Since normal_offsets is a scalar, we can directly use it in arithmetic operations with broadcasting
    # Ensure normal_vectors are correctly shaped for operation

    # Preparing segment_indices for gathering with expected shape adjustment
    segment_indices_expanded = segment_indices.unsqueeze(-1).expand(-1, -1, 2)

    # Correct selection of normals; assuming shape correction for segment_indices and normal_vectors as needed
    selected_normals = torch.gather(normal_vectors, 1, segment_indices_expanded)

    # Multiply selected normal vectors by the normal offset
    # Broadcasting of scalar normal_offsets to the shape of selected_normals during multiplication
    offset_vectors = selected_normals * normal_offsets
    
    positions += offset_vectors
    
    yaws = selected_yaws
    # Create masked values for the last centerline point
    #if it pass the last point, we just fix it there
    last_positions = centerline_traj[:, -1, :2].unsqueeze(1).expand(-1, horizon, -1)
    last_yaws = centerline_traj[:, -1, 2].unsqueeze(1).expand(-1, horizon)

    # Update positions and yaws based on the mask
    positions[last_idx_mask.expand_as(positions)] = last_positions[last_idx_mask.expand_as(last_positions)]
    yaws[last_idx_mask.squeeze(-1)] = last_yaws[last_idx_mask.squeeze(-1)]

    
    return positions, speed, yaws

def batch_mask_distances_after_passing(distances):
    """
    Masks distances to zero after the vehicle has passed the conflict point in a batch operation,
    indicated by the distance first decreasing and then increasing.

    Args:
    - distances: A tensor of shape (num_samples, T) representing the distance to the conflict
      point over time for each sample.

    Returns:
    - mask
    """
    # Calculate differences between consecutive distances to find where increase begins
    diffs = distances[:, 1:] - distances[:, :-1]
    # Identify where distance starts increasing after initially decreasing
    signs_change = torch.cat((torch.zeros((distances.shape[0], 1), dtype=torch.bool, device=distances.device), diffs < 0), dim=1) != torch.cat((diffs < 0, torch.zeros((distances.shape[0], 1), dtype=torch.bool, device=distances.device)), dim=1)
    increase_starts = signs_change[:,:-1] & (diffs >= 0)

    # Generate a cumulative sum to propagate the mask after the increase starts
    mask = torch.cumsum(increase_starts, dim=1) == 0
    # Append a column of False to the mask to ensure correct alignment and masking
    mask = torch.cat((mask, torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=distances.device)), dim=1)

    return mask
