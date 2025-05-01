from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.agent_rel_classify import find_conflict_point
from tbsim.utils.trajdata_utils import calculate_heading, enlarge_batch_shape,enlarge_batch_samples
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.planning_utils import (
    get_collision_dis,
    get_drivable_area_loss,
    get_lane_loss_simple,
)

GUIDANCE_REGISTRY = {}

def register_guidance(name):
    def decorator(cls):
        GUIDANCE_REGISTRY[name] = cls
        # Add registered_name as a class attribute
        cls.name = name
        return cls
    return decorator

def guidance_factory(name, config):
    cls = GUIDANCE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown guidance type: {name}")
    return cls(**config)

from tbsim.configs.guidance_config import GuidanceConfig

def initialize_loss_calculator(guide_config=None):
    if guide_config is None:
        raise ValueError("Guide configuration must be provided.")

    if isinstance(guide_config, Dict):
        # Convert old-style config to new GuidanceConfig
        guidance_config = GuidanceConfig()
        
        # Set guidance functions
        guidance_config.set_guidance_fn(guide_config.guidance_fn)
        
        # Update optimization parameters
        if hasattr(guide_config, "params"):
            guidance_config.update_params(guide_config.params)
            
        # Update combined loss configuration
        if hasattr(guide_config, "combineloss_config"):
            guidance_config.update_combine_loss(guide_config.combineloss_config)
            
        # Update individual guidance configurations
        if hasattr(guide_config, "guidance_configs"):
            for name in guide_config.guidance_fn:
                if name in guide_config.guidance_configs:
                    guidance_config.update_config(name, guide_config.guidance_configs[name])
    else:
        guidance_config = guide_config

    calculators = []
    for loss_name in guidance_config.guidance_fn:
        calculator = guidance_factory(loss_name, guidance_config.get_config(loss_name))
        calculators.append(calculator)

    if len(calculators) > 1:
        return CombinedLossCalculator(
            calculators,
            guidance_config.combineloss_config["weights"],
            guidance_config.combineloss_config["ctrl_weights"],
            guidance_config.combineloss_config
        )
    else:
        return calculators[0]
    
class Guidance(ABC):
    def __init__(self, loss_timesteps=None, filter_timesteps=None, loss_scale=1.0) -> None:
        """loss_timesteps: the horizon that we calculate the loss
        filter_timesteps: the horizon we use to filter trajs
        """
        self.loss_timesteps = loss_timesteps
        self.filter_timesteps = filter_timesteps
        self.loss_scale = loss_scale
        self.device = None

    @abstractmethod
    def calculate_loss(self, action, state, data_batch_for_guidance):
        """Calculate loss with guidance data
        
        Args:
            action: Action trajectories
            state: State trajectories
            data_batch_for_guidance: Dictionary containing guidance-related data
        """
        raise NotImplementedError

    def calculate_grad(self, action, state, a_t, grad_wrt, data_batch_for_guidance):
        """Calculate gradient with guidance data
        
        Args:
            action: cleaned traj or noisy_traj
            state: resulting traj
            a_t: noised traj input tau_k
            grad_wrt: either "clean_guide" or "noisy_guide"
            data_batch_for_guidance: Dictionary containing guidance-related data
        """
        # Notice that we should guide it through the reward instead of loss
        loss = self.calculate_loss(action, state, data_batch_for_guidance)
        if grad_wrt == "clean_guide":
            grad = torch.autograd.grad(-loss.sum(), a_t, retain_graph=True)[0]
        elif grad_wrt == "noisy_guide":
            grad = torch.autograd.grad(-loss.sum(), action, retain_graph=True)[0]
        else:
            raise NotImplementedError(f"Unknown grad_wrt: {grad_wrt}")

        assert torch.isfinite(grad).all()
        state.detach()  # Disable gradient computation
        action.detach()
        return grad

    def filter(self, action, state, data_batch_for_guidance):
        """Filter trajectories based on loss"""
        loss = self.calculate_loss(action, state, data_batch_for_guidance)
        if self.filter_timesteps is not None:
            loss = loss[:, :self.filter_timesteps]
        batch_size = data_batch_for_guidance["batch_size"]
        loss = loss.sum(dim=-1).reshape(batch_size, -1)
        armin_indices = torch.argmin(loss, dim=1)
        filtered_a_trajs = torch.gather(
            action.view((batch_size, -1, *action.shape[1:])),
            1,
            armin_indices[..., None, None, None].expand((-1, -1, *action.shape[1:])),
        )
        return filtered_a_trajs

    def to_device(self, device):
        self.device = device

class CombinedLossCalculator(Guidance):
    def __init__(self, loss_calculators, weights, ctrl_weights, args):
        super().__init__()
        assert len(weights) == len(loss_calculators),"Number of weights does not match number of loss calculators"
        self.loss_calculator_dict = {calc.name: calc for calc in loss_calculators}
        
        self.weights = torch.tensor(weights if weights is not None else [1.0] * len(loss_calculators))
        self.ctrl_weights = torch.tensor(ctrl_weights if ctrl_weights is not None else self.weights.clone())
        
        
        # Filter setup
        assert args["filter_criterion"] in ["combined", "individual", "weight_combined", "heirachical"]
        self.filter_criterion = args["filter_criterion"]
        self.filter_target = args["filter_target"]
        
        # Control target setup
        self.ctrl_filter_target = args["ctrl_filter_target"] if "ctrl_filter_target" in args else None
        self.ctrl_filter_criterion = args.get("ctrl_filter_criterion")
        
        if self.filter_criterion == "weight_combined":
            self.filter_weights = {type(calc).__name__: w for calc, w in zip(loss_calculators, args.filter_weights)}
        self.batch_weights = None

    def calculate_loss(self, action, state, data_batch_for_guidance):
        total_loss = torch.zeros(action.shape[:2]).to(action.device)
        
        for idx, (loss_name, calculator) in enumerate(self.loss_calculator_dict.items()):
            loss_term = calculator.calculate_loss(action, state, data_batch_for_guidance) #loss BN,T
            print(f"{loss_name}: {loss_term.mean()}")
            if self.batch_weights is None:
                w = self.weights[idx]
            else:
                w = self.batch_weights[:, idx:idx+1]
                w = enlarge_batch_samples(
                    w, data_batch_for_guidance["batch_size"], data_batch_for_guidance["num_samples"]) #B,num_guidance ->BN,num_guidance
            total_loss = total_loss + w * loss_term
            
        return total_loss

    def filter(self, action, state, data_batch_for_guidance):
        if self.filter_criterion == "combined":
            return self._filter_combined(action, state, data_batch_for_guidance)
        elif self.filter_criterion == "weight_combined":
            return self._filter_weight_combined(action, state, data_batch_for_guidance)
        else:
            assert self.filter_target in self.loss_calculator_dict
            filtered_a_trajs = self.loss_calculator_dict[self.filter_target].filter(
                action, state, data_batch_for_guidance
            )
            
            if hasattr(self, "control_mask") and self.ctrl_filter_target is not None:
                control_mask = self.control_mask
                if self.ctrl_filter_criterion == "individual":
                    filtered_all_ctrl = self.loss_calculator_dict[self.ctrl_filter_target].filter(
                        action, state, data_batch_for_guidance
                    )
                    filtered_a_trajs[control_mask] = filtered_all_ctrl[control_mask]
                    
            return filtered_a_trajs

    def _filter_combined(self, action, state, data_batch_for_guidance):
        loss = self.calculate_loss(action, state, data_batch_for_guidance)
        batch_size = data_batch_for_guidance["batch_size"]
        loss = loss.sum(dim=-1).reshape(batch_size, -1)
        armin_indices = torch.argmin(loss, dim=1)
        return torch.gather(
            action.view((batch_size, -1, *action.shape[1:])),
            1,
            armin_indices[..., None, None, None].expand((-1, -1, *action.shape[1:])),
        )

    def _filter_weight_combined(self, action, state, data_batch_for_guidance):
        loss = torch.zeros(action.shape[:2]).to(action.device)
        for loss_name, calculator in self.loss_calculator_dict.items():
            w = self.filter_weights[loss_name]
            loss_term = calculator.calculate_loss(action, state, data_batch_for_guidance)
            loss = loss + w * loss_term
        return self._apply_filter(loss, action, data_batch_for_guidance["batch_size"])

    def to_device(self, device):
        for calculator in self.loss_calculator_dict.values():
            calculator.to_device(device)
        self.device = device
        self.weights = self.weights.to(device)
        self.ctrl_weights = self.ctrl_weights.to(device)

    def update_params(self, params):
        for calculator in self.loss_calculator_dict.values():
            calculator.update_params(params)
        self.batch_size = params["batch_size"]

    def update_config(self, config):
        # update loss weights if each agent is different
        # weight is shape (batch_size,loss_term) loss: (batch*sample,T)
        for calculator in self.loss_calculator_dict.values():
            calculator.update_config(config)

        # Setup the dynamic weights based on control indices
        self.batch_weights = self.weights.repeat(len(config["batch_ctrl_indices"]), 1)

        self.control_mask = torch.tensor(config["batch_ctrl_indices"], dtype=torch.bool, device=self.device)
        self.batch_weights[self.control_mask] = self.ctrl_weights


    def _apply_filter(self, loss, action, batch_size):
        # Apply the filter based on the filter criterion
        if self.filter_criterion == "combined":
            loss = loss.sum(dim=-1).reshape(batch_size, -1)
            armin_indices = torch.argmin(loss, dim=1)
            return torch.gather(
                action.view((batch_size, -1, *action.shape[1:])),
                1,
                armin_indices[..., None, None, None].expand((-1, -1, *action.shape[1:])),
            )
        elif self.filter_criterion == "weight_combined":
            loss = loss.sum(dim=-1).reshape(batch_size, -1)
            armin_indices = torch.argmin(loss, dim=1)
            return torch.gather(
                action.view((batch_size, -1, *action.shape[1:])),
                1,
                armin_indices[..., None, None, None].expand((-1, -1, *action.shape[1:])),
            )
        else:
            raise ValueError(f"Unknown filter criterion: {self.filter_criterion}")

@register_guidance("speed")
class ConstantSpeedLossCalculator(Guidance):
    def __init__(self, desired_speed=20, loss_timesteps=None, filter_timesteps=None):
        super().__init__(loss_timesteps, filter_timesteps)

        self.desired_speeds = None  # Initialized to None as it will be set using initialize_speeds
        self.default_desired_speed = desired_speed
        self.mode = "hard"

    def initialize_guide(self, batch_size, device):
        self.desired_speeds = torch.full(
            (batch_size, 1), self.default_desired_speed, dtype=torch.float32, device=device
        )

    def calculate_loss(self, action, state):
        if self.desired_speeds is None or self.desired_speeds.shape[0] != action.shape[0]:
            self.initialize_guide(action.shape[0], device=action.device)
            self.desired_speeds = self.desired_speeds.to(action.device)
        # Calculate the loss based on the trFajectories(x,y,)
        # loss = torch.linalg.norm(state[...,-2] - self.desired_speed, dim =-1)
        # Important
        self.desired_speeds = enlarge_batch_shape(state, self.desired_speeds)
        if self.mode == "soft":
            speed_diff = torch.abs(state[..., -2] - self.desired_speeds)
            # Apply a threshold of ±1 for the desired speed
            speed_tolerance = 1.0
            loss = torch.relu(speed_diff - speed_tolerance)
        else:
            loss = torch.abs(state[..., -2] - self.desired_speeds)
        # Return the calculated loss
        return loss

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass

    def update_config(self, config: Dict[str, Any]) -> None:

        self.initialize_guide(len(config["batch_ctrl_indices"]), device=self.device)

        control_mask = torch.tensor(config["batch_ctrl_indices"], dtype=torch.bool, device=self.device)

        self.desired_speeds[control_mask] = config.speed_config["desired_speed"]

@register_guidance("route")
class RouteLossCalculator(Guidance):
    def __init__(
        self, 
        lane_margin=1.0, 
        nonlinear_factor=5.0, 
        loss_timesteps=None, 
        filter_timesteps=None, 
        loss_scale=1.0,
        max_penalty_value=100.0,  # Added parameter
        tangential_reward_scale=0.0,  # Added parameter
        min_points_for_direction=10  # Added parameter for direction calculation
    ):
        super().__init__(loss_timesteps, filter_timesteps, loss_scale)
        # Initialize parameters
        self.lane_margin = lane_margin
        self.non_linear_margin = lane_margin + 0.5
        self.nonlinear_factor = nonlinear_factor
        self.loss_scale = loss_scale
        
        # Move magic numbers to attributes
        self.max_penalty_value = max_penalty_value
        self.tangential_reward_scale = tangential_reward_scale
        self.min_points_for_direction = min_points_for_direction

    def update_config(self, config: Dict[str, Any]):
        pass
        
    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass
        
    def calculate_loss(self, action, state, params):
        # Extract parameters from params dictionary
        lane = params.get("centerline")  # Assumed to be already correctly shaped
        lane_avail = params.get("lane_avail")
        
        # Calculate the loss based on the trajectories (x, y, yaw)
        ego_trajectories = torch.cat([state[..., :2], state[..., 3:4]], dim=-1)
        Ne, T = ego_trajectories.shape[:2]
        cost = torch.zeros((Ne, T), device=ego_trajectories.device)

        # Compute the normal distance to the lane
        norm_dist = GeoUtils.batch_get_distance(
            ego_trajectories.reshape(-1, 3),
            lane[:, None].repeat_interleave(T, 1).reshape(Ne * T, -1, 3),
        )
        norm_dist_min, norm_dist_indices = norm_dist.min(1)
        margin_to_lane = (norm_dist_min.reshape(Ne, T) - self.lane_margin).clamp(min=0)

        # Apply nonlinear penalty
        nonlinear_penalty = torch.where(
            margin_to_lane >= self.non_linear_margin, 
            self.nonlinear_factor * (margin_to_lane - self.non_linear_margin), 
            margin_to_lane
        )

        # Clip the nonlinear penalty using class attribute
        nonlinear_penalty = torch.clamp(nonlinear_penalty, min=0, max=self.max_penalty_value)

        # Compute tangential reward using class attribute
        norm_dist_indices = norm_dist_indices.reshape(Ne, T)
        distances = torch.linalg.norm(lane[:, 1:, :2] - lane[:, :-1, :2], dim=-1)
        mask = torch.arange(distances.size(1), device=distances.device).expand(Ne, -1)
        mask = (mask >= norm_dist_indices[:, 0:1]) & (mask < norm_dist_indices[:, -1:])
        total_tangential_distance = torch.sum(distances * mask, dim=-1)
        tangential_reward = total_tangential_distance * self.tangential_reward_scale

        cost += (nonlinear_penalty - tangential_reward[:, None] / T) * lane_avail[:, None]

        if self.loss_timesteps is not None:
            cost[:, self.loss_timesteps:] = 0.0
        return cost * self.loss_scale

@register_guidance("collision")
class CollisionLossCalculator(Guidance):
    def __init__(
        self,
        raw_types=None,
        radius=5,
        mode="gaussian",
        sigma=1,
        heading_weight=None,
        buff_dist=None,
        prediction_mode="multi_agent",
        loss_timesteps=None,
        filter_timesteps=None,
        adv_mode=False,
        loss_scale=1.0, 
        **kwargs #not using this but dummy for inheritance
    ):
        super().__init__(loss_timesteps, filter_timesteps, loss_scale)
        """ Ensure that each agent are not colliding with each other
        args:
            1)adv_mode: if True, the collision loss between ego and adv will be set to zero
        """
        assert prediction_mode in [
            "constant_velocity",
            "multi_agent",
        ], "prediction_mode must be 'constant_velocity' or 'multi_agent'"

        self.mode = mode
        self.adv_mode = adv_mode
        self.prediction_mode = prediction_mode

        if mode == "disk" or mode == "indicator":
            assert radius is not None, "For disk collision, radius must be provided"
            self.radius = radius
        elif mode == "bounding_box" or mode == "multiple_circle":
            self.buff_dist = buff_dist
            self.raw_types = raw_types
        elif mode == "gaussian":
            assert sigma is not None, "For gaussian collision, sigma must be provided"
            assert heading_weight is not None, "For gaussian collision, heading_weight must be provided"
            assert buff_dist is not None, "For gaussian collision, buff_dist must be provided"

            self.sigma = sigma
            self.heading_weight = heading_weight
            self.sigma = sigma
            self.heading_weight = heading_weight
            self.buff_dist = buff_dist
        elif mode == "indicator":
            self.buff_dist = buff_dist
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.scene_mask = None
        self.world_from_agent = None
        self.agent_speed = None
        self.agent_yaw = None
        self.extents = None
        self.raw_types = None

        # adv mode, we need to mask the loss
        self.ctrl_indices = None
        self.ego_indices = None
        self.ego_mask = None
        self.ctrl_mask = None

    def update_config(self, config: Dict[str, Any]):
        # Update ego_indices and ctrl_indices
        if self.adv_mode:
            self.ctrl_indices = config.get("batch_ctrl_indices", self.ctrl_indices)
            self.ego_indices = config.get("batch_ego_indices", self.ego_indices)
            # Create corresponding masks
            self.ego_mask = torch.tensor(self.ego_indices, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            self.ctrl_mask = torch.tensor(self.ctrl_indices, device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Update the parameters based on the provided dictionary.

        :param params: Dictionary containing the new values for attributes.
        """
        pass
    def calculate_loss(self, action, state, data_batch_for_guidance):
        """Calculate collision loss based on the provided parameters and mode.
        
        Args:
            action: Action trajectories
            state: State trajectories
            data_batch_for_guidance: Dictionary containing guidance data including:
                - world_from_agent: World transformation matrix
                - curr_speed: Current speed
                - yaw: Current yaw
                - ego_extents: Vehicle extents
                - raw_types: Agent types
                - scene_ids: Scene identifiers
                - batch_size: Batch size
        """
        # Get parameters from guidance data
        world_from_agent = data_batch_for_guidance["world_from_agent"]
        agent_speed = data_batch_for_guidance["curr_speed"]
        agent_yaw = data_batch_for_guidance["yaw"]
        extents = data_batch_for_guidance["ego_extents"]
        raw_types = data_batch_for_guidance.get("raw_types")
        BN = data_batch_for_guidance["BN"] #this batchsize is BN
        batch_size = data_batch_for_guidance["batch_size"]
        # Create scene mask
        scene_ids = data_batch_for_guidance["scene_ids"]
        scene_mask = create_scene_mask(scene_ids).to(agent_yaw.device)
        
        # Get agent position from world transform
        agent_pos = None #not using for now

        # Transform state to world coordinates
        world_xy_fut = transform_points_tensor(state[..., :2], world_from_agent)        
        # Reshape to samples B,N
        action = action.reshape((batch_size, -1, *action.shape[1:]))
        world_xy_fut = world_xy_fut.reshape((batch_size, -1, *world_xy_fut.shape[1:]))
        world_yaw = state[..., -1] + agent_yaw.unsqueeze(-1)
        world_yaw = world_yaw.reshape((batch_size, -1, *world_yaw.shape[1:]))[..., None]

        # Calculate pairwise distances based on prediction mode
        if self.prediction_mode == "constant_velocity":
            T = world_xy_fut.shape[2]
            speed = agent_speed.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, T, -1)
            yaw = agent_yaw.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, T, -1)
            
            dt = data_batch_for_guidance["dt"]
            time_steps = torch.arange(1, T + 1).float().to(agent_speed.device) * dt
            
            delta_pos = speed * time_steps.unsqueeze(0).unsqueeze(-1) * torch.cat(
                (torch.cos(yaw), torch.sin(yaw)), dim=-1
            )
            
            predicted_pos = agent_pos.unsqueeze(1).unsqueeze(2) + torch.cumsum(delta_pos, dim=-2)
            distances, dist_vec = pairwise_distances(world_xy_fut, predicted_pos)
        else:  # multi_agent mode
            distances, dist_vec = pairwise_distances(world_xy_fut, world_xy_fut)

        if self.mode == "gaussian":
            cos_vec = torch.cos(world_yaw)
            sin_vec = torch.sin(world_yaw)
            long_vec = torch.cat([cos_vec, sin_vec], dim=-1)
            
            heading_dist_vec, non_heading_dist_vec = self.batch_vector_projection(dist_vec, long_vec)
            heading_distances = torch.linalg.norm(heading_dist_vec, dim=-1) - extents[:, None, None, 0:1] / 2
            non_heading_distances = torch.linalg.norm(non_heading_dist_vec, dim=-1) - extents[:, None, None, 1:2] / 2
            
            heading_distances_normalized = heading_distances / (self.sigma * self.heading_weight)
            non_heading_distances_normalized = non_heading_distances / self.sigma
            buff_dist_term = self.buff_dist**2 if self.buff_dist < 0 else -self.buff_dist**2
            
            collision_loss = torch.exp(
                -0.5 * (heading_distances_normalized**2 + non_heading_distances_normalized**2 + buff_dist_term)
            ) #B,B,N,T
        elif self.mode == "multiple_circle":
            agent_trajectories = torch.cat([world_xy_fut, world_yaw], dim=-1)
            collision_loss = self.get_circle_approximation_collision_loss(
                agent_trajectories, extents, buffer_dist=self.buff_dist
            ) * 25

        # Apply masks and reshape
        identity_mask = torch.eye(collision_loss.shape[0]).unsqueeze(2).unsqueeze(-1).to(collision_loss.device)
        collision_loss = collision_loss * (1 - identity_mask)
        collision_loss = collision_loss.masked_fill(~scene_mask[:, :, None, None], 0.0)
        
        if self.adv_mode:
            #Set ego-adv car pair to zero to encrouange colllisions
            assert (self.ego_mask is not None and self.ctrl_mask is not None), "Ego and ctrl masks are not provided"
            # collision_loss = collision_loss * (-2 * (self.ego_mask * self.ctrl_mask).to(action.device) + 1)
            collision_loss = collision_loss * (1 - (self.ego_mask * self.ctrl_mask).to(action.device))
        collision_loss = collision_loss.sum(dim=1).reshape(-1, collision_loss.shape[-1])

        if self.loss_timesteps is not None:
            collision_loss[:, self.loss_timesteps:] = 0.0
            
        return collision_loss * self.loss_scale
  
    @staticmethod
    def batch_vector_projection(vec, onto_vec):
        """Computes the vector projection and orthogonal component in batch form.
        
        Args:
            vec (torch.Tensor): Input vectors of shape (B,B,N,T,2)
            onto_vec (torch.Tensor): Projection vectors of shape (B,N,1,T,2)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (projection, orthogonal_comp) both of shape (B,B,N,T,2)
        """
        # TODO figure out why this will cause NAN error

        # If vec and onto_vec have different numbers of dimensions, add a dimension to onto_vec
        if len(vec.shape) != len(onto_vec.shape):
            onto_vec = onto_vec.unsqueeze(1)

        dot_product = (vec * onto_vec).sum(dim=-1, keepdims=True)
        norm_onto_vec_sq = (onto_vec * onto_vec).sum(dim=-1, keepdims=True)
        projection = dot_product / norm_onto_vec_sq * onto_vec

        # Subtract the projection from the original vector to get the orthogonal component
        orthogonal_comp = vec - projection

        # assert torch.isfinite(projection).all()

        return projection, orthogonal_comp

    # -------------------------------------- circle approximation------------------------------------------
    @staticmethod
    def get_circle_approximation_collision_loss(agent_trajectories, agent_extents, num_circles=3, buffer_dist=2.0):
        B, N, T, _ = agent_trajectories.shape
        # Compute vehicle attributes (length and width)
        veh_lengths = agent_extents[..., 0:1]  # [B, 1]
        veh_widths = agent_extents[..., 1:]  # [B, 1]

        # Compute radius of the discs for each vehicle assuming length > width
        veh_radius = veh_widths / 2.0  # [B, 1]

        # Create centroids of circles for each vehicle
        cent_min = -(veh_lengths / 2.0) + veh_radius  # [B, 1]
        cent_max = (veh_lengths / 2.0) - veh_radius  # [B, 1]

        # Compute linspace for each agent
        linspace = torch.linspace(0, 1, num_circles, device=agent_trajectories.device)  # [num_circles]
        # Compute circle centers
        cent_x = (1 - linspace) * cent_min + linspace * cent_max  # [B, num_circles]
        cent_x = cent_x.unsqueeze(1).unsqueeze(2).expand(-1, N, T, -1)  # [B, N, T, num_circles]

        # Create dummy states for centroids with y=0 so can transform later
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=-1)  # [B, N, T, num_circles, 2]

        # Get yaw angle and create rotation matrix
        yaw_angle = agent_trajectories[..., 2]  # [B, N, T]
        rotation_matrix = torch.stack(
            [
                torch.cos(yaw_angle),
                -torch.sin(yaw_angle),
                torch.zeros_like(yaw_angle),
                torch.sin(yaw_angle),
                torch.cos(yaw_angle),
                torch.zeros_like(yaw_angle),
                torch.zeros_like(yaw_angle),
                torch.zeros_like(yaw_angle),
                torch.ones_like(yaw_angle),
            ],
            dim=-1,
        )
        rotation_matrix = rotation_matrix.view(B, N, T, 1, 3, 3)  # [B, N, T, 3, 3]

        # Apply rotation to centroids
        centroids_transformed = transform_points_tensor(
            centroids.reshape(-1, num_circles, 2), rotation_matrix.reshape(-1, 3, 3)
        )
        centroids_transformed = centroids_transformed.view(
            B, N, T, num_circles, 2
        )  # reshape back to the original dimensions
        # shift the centroid
        centroids_transformed = centroids_transformed + agent_trajectories[..., :2].unsqueeze(-2)
        # Compute penalty distances
        # reference:https://github.com/nv-tlabs/STRIVE/blob/main/src/losses/traffic_model.py#L166
        penalty_dists = veh_radius + veh_radius.transpose(0, 1) + buffer_dist  # [B, B]

        # Compute pairwise distances across all samples
        cur_cent1 = centroids_transformed.unsqueeze(0).expand(B, B, N, T, num_circles, 2)
        cur_cent2 = centroids_transformed.unsqueeze(1).expand(B, B, N, T, num_circles, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2)  # [B, B, N, T, num_circles, num_circles]
        min_pair_dists = torch.min(pair_dists.view(B, B, N, T, -1), dim=-1)[0]  # [B, B, N, T]

        is_colliding_mask = min_pair_dists <= penalty_dists.unsqueeze(-1).unsqueeze(-1)  # [B, B, N, T]

        # Diagonals are self collisions so ignore them
        identity = torch.eye(B, device=agent_trajectories.device).unsqueeze(2).unsqueeze(2)  # [B, B, 1, 1]
        cur_valid_mask = torch.logical_not(identity).expand_as(is_colliding_mask)  # [B, B, N, T]

        is_colliding_mask = torch.logical_and(is_colliding_mask, cur_valid_mask)  # [B, B, N, T]

        # Compute penalties
        cur_penalties = torch.where(
            is_colliding_mask,
            1.0 - (min_pair_dists / penalty_dists.unsqueeze(-1).unsqueeze(-1)),
            torch.zeros_like(min_pair_dists),
        )  # [B, B, N, T]
        cur_penalties[~cur_valid_mask] = 0.0  # [B, B, N, T]
        if torch.isnan(cur_penalties).any():
            raise ValueError("Not finite")
        # Compute the collision loss
        collision_loss = cur_penalties

        return collision_loss
    

@register_guidance("ttc")
class TimeToCollisionLossCalculator(Guidance):
    def __init__(
        self,
        distance_bandwidth=1.0,
        time_bandwidth=1.0,
        min_velocity_diff=0.1,
        loss_timesteps=None,
        filter_timesteps=None,
        mode="all",
        loss_scale=1.0,
    ):
        '''
        Higher TTC Loss means:
        Agents are closer to colliding (small TTC)
        Agents' trajectories pass very close to each other (small distance_TTC)
        '''
        super().__init__(loss_timesteps, filter_timesteps, loss_scale)
        self.mode = mode
        self._d_bw = distance_bandwidth
        self._t_bw = time_bandwidth
        self._min_v = min_velocity_diff
        
        # Only keep masks as attributes since they're configuration-dependent
        self.ctrl_indices = None
        self.ego_indices = None
        self.ego_mask = None
        self.ctrl_mask = None


    def update_config(self, config: Dict[str, Any]):
        self.ctrl_indices = config.get("batch_ctrl_indices", self.ctrl_indices)
        self.ego_indices = config.get("batch_ego_indices", self.ego_indices)
        # Create corresponding masks
        self.ego_mask = torch.tensor(self.ego_indices, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.ctrl_mask = torch.tensor(self.ctrl_indices, device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


    def calculate_loss(self, action, state, data_batch_for_guidance):
        # Extract data from data_batch_for_guidance instead of using attributes
        world_from_agent = data_batch_for_guidance["world_from_agent"]
        agent_yaw = data_batch_for_guidance["yaw"]
        ego_plan = data_batch_for_guidance.get("ego_plan").to(self.device)
        batch_size = data_batch_for_guidance["batch_size"]
        scene_ids = data_batch_for_guidance["scene_ids"]
        ego_plan = data_batch_for_guidance.get("ego_plan").to(self.device)
        scene_mask = create_scene_mask(scene_ids).to(agent_yaw.device)

        # Transform coordinates
        world_xy_fut = transform_points_tensor(state[..., :2], world_from_agent)
        world_yaw = state[..., -1] + agent_yaw.unsqueeze(-1)
        
        # Calculate velocities
        vx = state[..., -2] * torch.cos(world_yaw)
        vy = state[..., -2] * torch.sin(world_yaw)
        world_velocity = torch.stack([vx, vy], dim=-1)

        # Reshape tensors
        B_N,T,_ = action.shape
        B = batch_size
        N = B_N // B
        
        world_xy_fut = world_xy_fut.reshape((B, -1, *world_xy_fut.shape[1:]))
        world_velocity = world_velocity.reshape((world_xy_fut.shape))

        # Handle ego trajectories
        ego_indices = torch.nonzero(self.ego_mask.squeeze()).squeeze()
        world_from_ego = world_from_agent.reshape(B, N, 3, 3)[ego_indices, 0]
        
        # Extend ego plan if needed
        ego_plan_world = transform_points_tensor(ego_plan, world_from_ego)

        world_xy_fut[ego_indices] = ego_plan_world[:, None]
        ego_world_velocity = (ego_plan_world[:, 1:] - ego_plan_world[:, :-1]) / data_batch_for_guidance["dt"]
        world_velocity[ego_indices, :, :-1] = ego_world_velocity[:, None]

        # Calculate TTC
        ttc, (TTC, dtc) = self._compute_ttc(world_xy_fut, world_velocity)

        # Apply masks
        identity_mask = torch.eye(ttc.shape[0]).unsqueeze(2).unsqueeze(-1).to(ttc.device)
        ttc = ttc * (1 - identity_mask)
        ttc = ttc.masked_fill(~scene_mask[:, :, None, None], 0.0)

        if self.mode == "ego_only":
            ttc = ttc * self.ego_mask
        elif self.mode == "ego_target":
            ttc = ttc * self.ego_mask * self.ctrl_mask
        elif self.mode == "all":
            pass
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        # Apply scene mask

        ttc = ttc.sum(dim=1).reshape(-1, ttc.shape[-1]) * self.loss_scale
        return -ttc # batch*sample,T

    def _compute_ttc(self, pos, velocity):
        '''
        Combine TTC and distance_TTC for a combined cost, following the reference implementation:https://github.com/TRI-ML/RAP/blob/main/risk_biased/utils/cost.py
        '''
        # pos shape is B,B,sample,T,2
        pos_diff = pos.unsqueeze(1) - pos
        velocity_diff = velocity.unsqueeze(1) - velocity

        dx = pos_diff[..., 0]
        dy = pos_diff[..., 1]
        vx = velocity_diff[..., 0]
        vy = velocity_diff[..., 1]

        speed_diff = torch.square(velocity_diff).sum(-1).clamp(self._min_v * self._min_v + 1e-12, None)

        TTC = -(dx * vx + dy * vy) / (speed_diff + 1e-6)
        distance_TTC = torch.where(
            TTC < 0,
            torch.sqrt(dx * dx + dy * dy + 1e-10),  # this may cause nan error
            torch.abs(vy * dx - vx * dy) / torch.sqrt(speed_diff) + 1e-6,
        )

        TTC = torch.relu(TTC)
        distance_TTC = torch.relu(distance_TTC)

        # Combine TTC and distance_TTC for a combined cost, following the reference implementation:https://github.com/TRI-ML/RAP/blob/main/risk_biased/utils/cost.py
        pre_exp_term = -torch.square(TTC) / (2 * self._t_bw) - torch.square(distance_TTC) / (2 * self._d_bw)
        cost = torch.exp(pre_exp_term)

        return cost, (TTC, distance_TTC)

@register_guidance("strivecollision")
class StriveCollisionLossCalculator(Guidance):
    '''
    This is inspired by the strive paper, where, we use distance to automatically determine which agent is adversarial
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_1 = 1.0  # Weight for collision term in the loss
        # human determine or auto
        # self.prediction_mode = "constant_velocity"
        self.prediction_mode = "ego_plan"

        self.mode = "gaussian"

        # adv mode, we need to mask the loss
        self.ctrl_indices = None
        self.ego_indices = None
        self.ego_mask = None
        self.ctrl_mask = None

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass
        # ego indices

    def update_config(self, config: Dict[str, Any]):
        self.ctrl_indices = config.get("batch_ctrl_indices", self.ctrl_indices)
        self.ego_indices = config.get("batch_ego_indices", self.ego_indices)
        # Create corresponding masks
        self.ego_mask = torch.tensor(self.ego_indices, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.ctrl_mask = torch.tensor(self.ctrl_indices, device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def calculate_loss(self, action, state, data_batch_for_guidance):
        #obtain from data_batch_for_guidance
        world_from_agent = data_batch_for_guidance["world_from_agent"]
        agent_yaw = data_batch_for_guidance["yaw"]
        ego_plan = data_batch_for_guidance.get("ego_plan")
        agent_pos = data_batch_for_guidance.get("agent_pos")
    
        BN = data_batch_for_guidance["BN"] #this batchsize is BN
        batch_size = data_batch_for_guidance["batch_size"]
        # Create scene mask
        scene_ids = data_batch_for_guidance["scene_ids"]
        scene_mask = create_scene_mask(scene_ids).to(agent_yaw.device)
        # enlarge to sample dim
        # Transform state to world coordinates
        world_xy_fut = transform_points_tensor(state[..., :2], world_from_agent)
        # world state to get yaw? THis will cause NAN error!!!!
        world_yaw = agent_yaw.unsqueeze(-1) + state[..., -1]
        # reshape to samples
        world_xy_fut = world_xy_fut.reshape((batch_size, -1, *world_xy_fut.shape[1:]))
        world_yaw = world_yaw.reshape((batch_size, -1, *world_yaw.shape[1:]))[..., None]

        action = action.reshape((batch_size, -1, *action.shape[1:]))

        B, N, T, _ = action.shape

        # Step 1: Get ego and ctrl indices from the masks
        ego_indices = torch.nonzero(self.ego_mask.squeeze()).squeeze()
        ctrl_indices = torch.nonzero(self.ctrl_mask.squeeze()).squeeze()
        # Handle 0-d tensors
        if ego_indices.dim() == 0:
            ego_indices = ego_indices.unsqueeze(0)
        if ctrl_indices.dim() == 0:
            ctrl_indices = ctrl_indices.unsqueeze(0)
        world_xy_fut_egodetach = world_xy_fut.clone()
        for idx in ego_indices:
            world_xy_fut_egodetach[idx] = world_xy_fut[idx].detach()

        #! self.world_from_world_agent is shape B*sample
        world_from_ego = world_from_agent.reshape(B, N, 3, 3)[ego_indices, 0]
        ego_plan_world = transform_points_tensor(ego_plan, world_from_ego)
        world_xy_fut_egodetach[ego_indices] = ego_plan_world[:, None]
        distances, dist_vec = pairwise_distances(world_xy_fut_egodetach, world_xy_fut_egodetach)  # (B,B,sample,T)

        # Iterate each scene, and ego
        epsilon = 1e-8
        collision_loss = torch.zeros((B, N, T), device=action.device)
        for i, ego_idx in enumerate(ego_indices):
            distance2_ego = distances[ego_idx, :]  # (B,sample,T)
            mask = scene_mask[ego_idx]  # (B,)

            mask[ego_idx] = False  # don't calculate distance to it self
            exp_neg_distances = torch.exp(-distance2_ego + epsilon)
            exp_neg_distances = exp_neg_distances * mask.unsqueeze(-1).unsqueeze(-1).float()

            delta_t = exp_neg_distances / (
                exp_neg_distances.sum(dim=(0), keepdim=True) + epsilon
            )  # Normalize across diff agents, timesteps?
            # if the car is benind, we should set to zero
            dist_vec2ego = agent_pos - agent_pos[ego_idx]  # (B,2)
            heading_ego = agent_yaw[ego_idx]
            heading_vec = torch.tensor([torch.cos(heading_ego), torch.sin(heading_ego)], device=self.device)  # (2,)

            # We use ego_plan to prevent noisy headings
            direction_vectors = (
                ego_plan_world[i, 1:10] - ego_plan_world[i, : 10 - 1]
            )  # Compute direction vectors between consecutive points
            average_direction_vector = torch.mean(direction_vectors, dim=0)  # Average the direction vectors
            heading_vec = average_direction_vector / (torch.norm(average_direction_vector) + epsilon)  # Normalize
            if torch.norm(average_direction_vector) < 1e-3:
                heading_ego = agent_yaw[ego_idx]
                heading_vec = torch.tensor([torch.cos(heading_ego), torch.sin(heading_ego)], device=self.device)  # (2,)

            norm_dist_vec2ego = dist_vec2ego / torch.norm(dist_vec2ego + epsilon, dim=1, keepdim=True)

            # Calculate dot product between normalized vectors and heading vector
            dot_product = torch.sum(norm_dist_vec2ego * heading_vec, dim=1)

            # Calculate the angle using the dot product; angles greater than 90 degrees indicate "behind"
            behind_mask = dot_product < 0  # True if the agent is behind the ego vehicle
            behind_mask_expanded = (~behind_mask).unsqueeze(1).unsqueeze(-1).expand(-1, N, -1)
            delta_t = delta_t * behind_mask_expanded.float()

            # calculate adv_term, the norm + smallest distance acrooss timesteps?
            min_distance2_ego = distance2_ego.min(dim=-1)[0]  # Get the minimum distance across timesteps
            adv_term = (distance2_ego + min_distance2_ego.unsqueeze(-1) * 0.1) * delta_t  # B,sample,T

            collision_loss += adv_term

        if self.loss_timesteps is not None:
            collision_loss[:, self.loss_timesteps :] = 0.0
        return collision_loss.view(B * N, T) * 10

@register_guidance("causecollision")
class CauseCollisionLossCalculator(Guidance):
    def __init__(self, loss_timesteps=None, filter_timesteps=None, loss_scale=1.0, *args, **kwargs):
        super().__init__(loss_timesteps, filter_timesteps, loss_scale)
        # human determine or auto
        self.ctrl_indices = None
        self.ego_indices = None
        self.ego_mask = None
        self.ctrl_mask = None
        # self.prediction_mode = "constant_velocity"
        self.prediction_mode = "ego_plan"
        self.adv_term_weight = {}
        
        adv_term_weights = kwargs.get("adv_term_weight", {})
        for key, value in adv_term_weights.items():
            self.adv_term_weight[key] = value
        self.interact_mode = kwargs.get("interact_mode", {})
        self.adv_bound = kwargs.get("adv_bound", 30.0)
        self.choose_adv = kwargs.get("choose_adv", "human")
        self.speed_diff = kwargs.get("speed_diff", 0.5)
        self.interact_dist_thresh = kwargs.get("interact_dist_thresh", 100.0)

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass

    def update_config(self, config: Dict[str, Any]):
        # TODO move to device!
        self.ctrl_indices = config.get("batch_ctrl_indices", self.ctrl_indices)
        self.ego_indices = config.get("batch_ego_indices", self.ego_indices)
        # Create corresponding masks
        self.ego_mask = torch.tensor(self.ego_indices, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.ctrl_mask = torch.tensor(self.ctrl_indices, device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)



    def calculate_loss(self, action, state, data_batch_for_guidance):
        """
        Calculate the collision loss for agents based on predicted trajectories and interactions.

        Parameters:
        - action: Tensor of shape (B_N, T, ...)
        - state: Tensor of shape (B_N, T, ...)
        - data_batch_for_guidance: Dictionary containing guidance data

        Returns:
        - col_loss_batch: Tensor containing the collision loss for each agent,samples
        """
        # Unpack batch size and number of samples
        B_N, T, _ = action.shape
        B = data_batch_for_guidance["batch_size"]
        N = B_N // B

        # Extract necessary data
        world_from_agent = data_batch_for_guidance["world_from_agent"]
        agent_speed = data_batch_for_guidance["curr_speed"]
        agent_yaw = data_batch_for_guidance["yaw"]
        extents = data_batch_for_guidance["ego_extents"]
        ego_plan = data_batch_for_guidance.get("ego_plan").to(self.device)
        lane_avail = data_batch_for_guidance.get("lane_avail")
        centerline = data_batch_for_guidance.get("centerline")

        # Transform state to world coordinates
        world_xy_fut = transform_points_tensor(state[..., :2], world_from_agent)
        world_yaw = agent_yaw.unsqueeze(-1) + state[..., -1]

        # Reshape tensors to group samples per batch
        action = action.reshape(B, -1, *action.shape[1:])
        state = state.reshape(B, -1, *state.shape[1:])
        world_xy_fut = world_xy_fut.reshape(B, -1, *world_xy_fut.shape[1:])
        world_yaw = world_yaw.reshape(B, -1, *world_yaw.shape[1:])[..., None]

        # Get indices of ego and control agents
        ego_indices = torch.nonzero(self.ego_mask.squeeze(), as_tuple=False).squeeze()
        ctrl_indices = torch.nonzero(self.ctrl_mask.squeeze(), as_tuple=False).squeeze()

        # Ensure indices are at least 1D tensors
        ego_indices = ego_indices.unsqueeze(0) if ego_indices.dim() == 0 else ego_indices
        ctrl_indices = ctrl_indices.unsqueeze(0) if ctrl_indices.dim() == 0 else ctrl_indices

        # Clone and detach ego agents' future positions
        world_xy_fut_detached = world_xy_fut.clone()
        world_xy_fut_detached[ego_indices] = world_xy_fut[ego_indices].detach()

        # Compute pairwise distances and vectors based on prediction mode
        if self.prediction_mode == "constant_velocity":
            raise NotImplementedError("constant_velocity not implemented")
        elif self.prediction_mode == "multi_agent":
            distances, dist_vec = pairwise_distances(world_xy_fut_detached, world_xy_fut_detached)
        elif self.prediction_mode == "ego_plan":
            # Use ego_plan for ego agents' future positions
            world_from_ego = world_from_agent.reshape(B, N, 3, 3)[ego_indices, 0]
            ego_plan_world = transform_points_tensor(ego_plan, world_from_ego)
            world_xy_fut_detached[ego_indices] = ego_plan_world[:, None]
            distances, dist_vec = pairwise_distances(world_xy_fut_detached, world_xy_fut_detached)
        else:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")

        # Initialize collision loss tensor
        col_loss_batch = torch.zeros(distances.shape[1:], device=self.device)  # Shape: (B, samples, T)

        # Iterate over each ego and control agent pair
        for i, (ego_idx, ctrl_idx) in enumerate(zip(ego_indices, ctrl_indices)):
            # Extract filtered distances
            filtered_distance = distances[ego_idx, ctrl_idx]  # Shape: (samples, T)
            min_distances, _ = torch.min(filtered_distance, dim=-1, keepdim=True)

            # Get speed and yaw from state
            speed = state.reshape(B, N, T, -1)[..., -2]

            # Get future yaw and speed for agents
            if self.prediction_mode == "multi_agent":
                ego_yaw_fut = world_yaw[ego_idx] + state[ego_idx, ..., -2:-1]
                ego_speed = speed[ego_idx]
            elif self.prediction_mode == "constant_velocity":
                raise NotImplementedError("constant_velocity not implemented")
            else:
                delta_pos = ego_plan_world[i, 1:] - ego_plan_world[i, :-1]
                ego_yaw_fut = torch.atan2(delta_pos[:, 1], delta_pos[:, 0]).unsqueeze(0)
                ego_yaw_fut = torch.cat([ego_yaw_fut, ego_yaw_fut[:, -1:]], dim=1).unsqueeze(-1)
                ego_speed = (torch.norm(delta_pos, dim=-1) / 0.1).unsqueeze(0)
                ego_speed = torch.cat([ego_speed, ego_speed[:, -1:]], dim=1)

            ctrl_speed = speed[ctrl_idx]
            ctrl_yaw_fut = world_yaw[ctrl_idx] + state[ctrl_idx, ..., -2:-1]

            # Get agent extents
            ego_extent = extents[ego_idx]
            ctrl_extent = extents[ctrl_idx]


            # Check for available centerlines
            if lane_avail[ego_idx] and lane_avail[ctrl_idx]:
                # Transform centerlines to world coordinates
                world_from_agent_reshaped = world_from_agent.reshape(B, N, 3, 3)
                ego_centerline = transform_points_tensor(centerline[ego_idx], world_from_agent_reshaped[ego_idx, 0])
                ctrl_centerline = transform_points_tensor(centerline[ctrl_idx], world_from_agent_reshaped[ctrl_idx, 0])

                # Find conflict point
                conflict_point, idx1, idx2 = find_conflict_point(ego_centerline, ctrl_centerline)

                # Find current lane indices
                ego_lane_idx = find_current_lane_idx(ego_centerline, world_xy_fut[ego_idx][0, 0])
                ctrl_lane_idx = find_current_lane_idx(ctrl_centerline, world_xy_fut[ctrl_idx][0, 0])

                # Determine if agents are before the conflict point
                is_ego_before_conflict = ego_lane_idx < idx1
                is_ctrl_before_conflict = ctrl_lane_idx < idx2

                # Skip if not interacting
                if not (is_ego_before_conflict or is_ctrl_before_conflict) or conflict_point is None:
                    continue

                # Calculate distances to conflict point
                ego_dist_to_conflict = torch.norm(world_xy_fut[ego_idx] - conflict_point[None, None], dim=-1)
                ctrl_dist_to_conflict = torch.norm(world_xy_fut[ctrl_idx] - conflict_point[None, None], dim=-1)

                # Initialize collision loss
                collision_loss = torch.zeros_like(ego_dist_to_conflict)

                # Create interaction mask
                if self.interact_mode == "conflict":
                    interaction_mask = ego_dist_to_conflict < self.interact_dist_thresh
                elif self.interact_mode == "distance":
                    interaction_mask = min_distances < self.interact_dist_thresh
                    interaction_mask = interaction_mask.expand(-1, T)
                else:
                    raise ValueError(f"Unknown interact_mode: {self.interact_mode}")

                # Compute penalties
                speed_penalty = calculate_exact_speed_penalty(
                    filtered_distance, ego_speed, ctrl_speed, self.speed_diff
                )  # Shape: (num_samples, T)

                # Update collision loss where interaction occurs
                if interaction_mask.any():
                    # Expand min_distances to match time dimension
                    min_distances_expanded = min_distances.expand(-1, T)  # Shape: (num_samples, T)
                    
                    # Initialize base collision terms for all timesteps
                    distance_term = self.adv_term_weight["distance"] * min_distances_expanded  # Shape: (num_samples, T)
                    speed_term = self.adv_term_weight["speed_penalty"] * speed_penalty  # Shape: (num_samples, T)
                    filtered_term = self.adv_term_weight["filtered_distance"] * filtered_distance  # Shape: (num_samples, T)
                    
                    # Combine terms where interaction occurs
                    collision_loss = torch.zeros_like(filtered_distance)  # Shape: (num_samples, T)
                    collision_loss[interaction_mask] = (
                        distance_term[interaction_mask] + 
                        speed_term[interaction_mask] + 
                        filtered_term[interaction_mask]
                    )

                # Zero out collision loss where no interaction
                collision_loss[~interaction_mask] = 0.0

                # Clamp collision loss
                collision_loss = torch.clamp(collision_loss, 0, self.adv_bound)

                # Update batch collision loss
                col_loss_batch[ctrl_idx] = collision_loss

        # Reshape and return collision loss batch
        return col_loss_batch.reshape(-1, col_loss_batch.shape[-1]) * self.loss_scale

@register_guidance("trajalign")
class TrajectoryAlignmentLoss(Guidance):
    """
    Calculates the alignment loss between generated trajectories/actions and a set of proposed trajectories.

    This loss encourages the generated trajectories (actions and states) to closely follow the proposed trajectories,
    which can represent desired paths or maneuvers.
    Args:
        proposals: A tensor of shape (B, S, T, 2), where:
                   B is the batch size,
                   S is the number of proposals,
                   T is the number of time steps,
                   2 represents the x and y coordinates.
        adv_idx: Index of the trajectory in the batch to focus on for alignment.

    The loss is calculated by comparing the generated trajectories in agent_centric coordinates to the proposals,
    taking into account both their positions and orientations over time.
    Note: If sample_num is greater than S, then each samplenum//S belongs to the same proposals
    """

    def __init__(self, loss_timesteps=None, filter_timesteps=None, loss_scale=1.0, *args, **kwargs):
        super().__init__(loss_timesteps, filter_timesteps, loss_scale)
        self.ego_indices = None
        self.ctrl_indices = None

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:

        adv_proposals_dict = params.get("adv_proposals_dict", None)
        # store the dictionary to each attribute
        if adv_proposals_dict is not None:
            self.adv_idx = adv_proposals_dict["adv_idx"]
            self.adv_proposals = adv_proposals_dict["adv_proposals"]
            self.adv_proposals_states = adv_proposals_dict["adv_proposals_states"]
        else:
            self.adv_proposals = None
            self.adv_proposals_states = None
            self.adv_idx = None
        self.batch_size = params.get("batch_size")

    def update_config(self, config: Dict[str, Any]):
        # TODO move to device!
        self.ctrl_indices = config.get("batch_ctrl_indices", self.ctrl_indices)
        self.ego_indices = config.get("batch_ego_indices", self.ego_indices)
        # Create corresponding masks
        self.ego_mask = torch.tensor(self.ego_indices, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.ctrl_mask = torch.tensor(self.ctrl_indices, device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def calculate_loss(self, action, state, data_batch_for_guidance):
        # collision_loss = collision_loss * (-2 * (self.ego_mask * self.ctrl_mask).to(action.device) + 1)
        adv_idx = data_batch_for_guidance["adv_proposals_dict"].get("adv_idx")
        adv_proposals = data_batch_for_guidance["adv_proposals_dict"].get("adv_proposals")
        adv_proposals_states = data_batch_for_guidance["adv_proposals_dict"].get("adv_proposals_states")
        batch_size = data_batch_for_guidance.get("batch_size")
        
        if adv_proposals is None:
            return 0
        B_N, T, _ = action.shape
        B = batch_size
        N = B_N // B

        action = action.view(B, N, T, 2)
        state = state.view(B, N, T, -1)

        traj_align_loss = torch.zeros(B, N, T, device=self.device)

        # Step 1: expand the proposals to the sample number
        adv_proposals_states = adv_proposals_states.repeat(1, N // adv_proposals.shape[1], 1, 1)
        # calculate the loss w.r.t to proposals trajs
        traj_align_loss[adv_idx] = torch.linalg.norm(state[adv_idx] - adv_proposals_states, dim=-1)

        traj_align_loss[:, self.loss_timesteps :] = 0
        return traj_align_loss.view(B_N, T) 

@register_guidance("drivearea")
class DrivableAreaLossCalculator(Guidance):
    def __init__(self, loss_timesteps=None, filter_timesteps=None):
        super().__init__(loss_timesteps, filter_timesteps)

        self.current_scene_id = None

    def update_config(self, config: Dict[str, Any]):
        pass

    def update_params(self, params: Dict[str, torch.Tensor]):
        # Check if it's a new scene
        current_scene_id = params["scene_ids"]  # Assuming scene_ids is a tensor and you have a single scene ID
        if isinstance(current_scene_id, torch.Tensor) and isinstance(self.current_scene_id, torch.Tensor):
            shapes_equal = current_scene_id.shape == self.current_scene_id.shape
            values_equal = (current_scene_id == self.current_scene_id).all() if shapes_equal else False
            new_scene = (self.current_scene_id is None) or not shapes_equal or not values_equal
        elif isinstance(current_scene_id, list) and isinstance(self.current_scene_id, list):
            new_scene = (self.current_scene_id is None) or (current_scene_id != self.current_scene_id)
        else:
            new_scene = self.current_scene_id is None
        # new_scene = (self.current_scene_id is None or current_scene_id != self.current_scene_id)
        self.current_scene_id = current_scene_id

        # Determine change lane or not

        batch_size = params.get("batch_size")
        self.batch_size = batch_size
        # Update the parameters
        self.raster_from_agent = params.get("raster_from_agent", None)
        self.dis_map = params.get("dis_map", None)
        self.ego_extents = params.get("ego_extents", None)

    def calculate_loss(self, action, state, data_batch_for_guidance):

        raster_from_agent = data_batch_for_guidance.get("raster_from_agent")
        dis_map = data_batch_for_guidance.get("dis_map")
        # Calculate the loss based on the trajectories(x,y,)
        traj = torch.cat([state[..., :2], state[..., -1:]], dim=-1)
        # loss = get_drivable_area_loss(traj[:,None], self.raster_from_agent, self.dis_map, self.ego_extents,enable_grad=True)
        loss = get_lane_loss_simple(traj, raster_from_agent, dis_map)  # can switch to graually increase
        # Return the calculated loss
        # self.visualize_dis_map(self.dis_map)
        return loss * 10

    @staticmethod
    def visualize_dis_map(dis_map):
        """
        Visualize the dis_map tensor.
        Args:
            dis_map (torch.Tensor): Tensor of size (batch_size, 224, 224)

        Returns:
            None
        """
        dis_map = dis_map.detach().cpu().numpy()  # Convert to NumPy array
        # Proceed with visualization using matplotlib.pyplot
        # ...
        batch_size = dis_map.shape[0]

        if batch_size == 1:
            axes = [axes]

        for i in range(batch_size):
            fig, axes = plt.subplots(1, 1)
            axes.imshow(dis_map[i], cmap="hot", interpolation="nearest")
            # axes[i].set_title(f'Dis_map - Batch {i+1}')

            plt.tight_layout()
            plt.savefig(f"Dis_map - Batch {i+1}")
            plt.close()


# ---------------------------- helper function ------------------------------
def create_scene_mask(scene_ids):
    if isinstance(scene_ids, np.ndarray):
        # Create a list of unique scene_ids
        unique_scene_ids = list(set(scene_ids))

        # Map each scene_id to a unique integer
        scene_id_to_int = {scene_id: i for i, scene_id in enumerate(unique_scene_ids)}

        # Convert your scene_ids list to integers
        integer_scene_ids = [scene_id_to_int[scene_id] for scene_id in scene_ids]

        # Convert to PyTorch tensor
        scene_ids_tensor = torch.tensor(integer_scene_ids)

        # Expand dimensions for broadcasting
        expanded_scene_ids = scene_ids_tensor.unsqueeze(1)

        # Create a boolean mask where True means the scene_ids are the same
        mask = expanded_scene_ids == expanded_scene_ids.t()
        return mask
    else:
        # Find unique scene_ids in the tensor
        unique_scene_ids, inverse_indices = torch.unique(scene_ids, return_inverse=True)

        # Use inverse_indices to create a mask
        expanded_scene_ids = inverse_indices.unsqueeze(1)

        # Create a boolean mask where True means the scene_ids are the same
        mask = expanded_scene_ids == expanded_scene_ids.t()

        return mask


def sample_pairwise_distances(sample_pos, predicted_pos):
    """
    Given batch of x coordinates in BxNxDx2, return BxBxN matrix of mean pairwise distances
    Args:
        predicted_pos: (B,N,T,2)
        sample_pos: (B,N,T,2)
    Return:
        mean_dist: (B,B,N,T)
    """
    # Get the dimensions
    B, N, T, _ = sample_pos.shape

    # Expand the tensors to get all pairwise combinations among batches and among samples
    # The resulting shapes will be (B, B, N, N, T, 2)
    sample_pos_exp = sample_pos.unsqueeze(1).unsqueeze(2).expand(-1, B, N, N, T, 2)
    predicted_pos_exp = predicted_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, N, N, T, 2)

    # Compute pairwise differences
    diff = sample_pos_exp - predicted_pos_exp  # Shape: (B, B, N, N, T, 2)

    # Compute distances
    dist = torch.linalg.norm(diff, dim=-1)  # Shape: (B, B, N, N, T)

    # Compute mean along the sample pairwise dimension
    mean_dist = dist.mean(dim=3)  # Shape: (B, B, N, T)
    diff = diff.mean(dim=3)
    return mean_dist, diff


def pairwise_distances(sample_pos, predicted_pos):
    """
    Given batch of x coordinates in BxNxDx2, return BxBxNxDx2 matrix of pairwise distances
    Args:
        predicted_pos: (B,1,T,2) or (B,N,T,2)
        sample_pos: (B,N,T,2)
    Return:
        dist: (B,B,N,T)
        dist_vec: (B,B,N,T,2)
        1st dimension represents the prediction, 2nd dimension represents the sample dimension

        dsit_vec[i,j] is the vector pointing from the predicted_pos[j] to sample_pos[i]
    """
    # Adjust the shape of predicted_pos and sample_pos
    # Note: if the original predicted_pos shape is (B,1,T,2), it becomes (B, B, N, T, 2)
    # If the original predicted_pos shape is (B,N,T,2), it becomes (B, B, N, T, 2)
    predicted_pos = predicted_pos.unsqueeze(1).expand(-1, sample_pos.shape[0], -1, -1, -1)
    sample_pos = sample_pos.unsqueeze(0)  # Now its shape is (1, B, N, T, 2)

    diff = sample_pos - predicted_pos
    dist = torch.linalg.norm(diff, dim=-1)
    return dist, diff


# ----------------------------visualize function----------------------------
def plot_trajectory_with_delta_y(traj, lane, delta_y, left_margin, cost):
    """
    Plot the trajectory with delta y to the reference lane.

    Args:
     traj: Tensor of shape (batch_size, T, 3) representing the trajectory points (x, y, h).
    - lane: Tensor of shape (batch_size, 150, 3) representing the reference lane points (x, y,h).
    - delta_y: Tensor of shape (batch_size, T, 150) representing the delta y values.
    - left_margin: Tensor of shape (batch_size, 32) representing the left margin values.
    - cost: Tensor of shape (batch_size,) representing the cost for each trajectory.
    Returns:
    - None (displays the plot).
    """

    batch_size, T, _ = traj.shape
    delta_y = delta_y.reshape(batch_size, T, -1)

    min_delta_y_to_lane = delta_y.min(axis=2)

    for i in range(batch_size):
        plt.figure(figsize=(8, 6))
        plt.plot(traj[i, :, 0], traj[i, :, 1], label="Trajectory")
        plt.plot(lane[i, :, 0], lane[i, :, 1], label="lane")

        for t in range(T):
            x = traj[i, t, 0]
            y = traj[i, t, 1]

            dy = min_delta_y_to_lane[i, t]

            plt.arrow(x, y, 0, -dy, color="red", width=0.1, alpha=0.5)
            plt.arrow(x, y, 0, dy, color="blue", width=0.1, alpha=0.5)
            # plt.arrow(x - lm, y, 0, dy, color='blue', width=0.1, alpha=0.5)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title(str(cost[i]))
        plt.grid(True)
        plt.savefig(f"margin_batch{i}")
        plt.close()
        break


def plot_intersect_scenario(ego_trajectory, ctrl_trajectory, ego_centerline, ctrl_centerline, conflict_point):
    # Ensure the input data is numpy array for easier handling
    # frst detach.cpu
    i = 0
    ego_trajectory = ego_trajectory[i].detach().cpu().numpy()
    ctrl_trajectory = ctrl_trajectory[i].detach().cpu().numpy()
    ego_centerline = ego_centerline.detach().cpu().numpy()
    ctrl_centerline = ctrl_centerline.detach().cpu().numpy()
    conflict_point = conflict_point.detach().cpu().numpy()

    # Calculate distances from the last positions of the agents to the conflict point
    ego2conflict_distance = np.linalg.norm(ego_trajectory[-1, :2] - conflict_point)
    ctrl2conflict_distance = np.linalg.norm(ctrl_trajectory[-1, :2] - conflict_point)

    # Create a new figure
    plt.figure(figsize=(10, 10))

    # Plot the trajectories of ego and control agent
    plt.plot(
        ego_trajectory[:, 0], ego_trajectory[:, 1], label="Ego Trajectory", color="orange", linewidth=2, marker="o"
    )
    plt.plot(
        ctrl_trajectory[:, 0], ctrl_trajectory[:, 1], label="Ctrl Trajectory", color="green", linewidth=2, marker="o"
    )

    # Plot the centerlines of ego and control agent
    plt.plot(ego_centerline[:, 0], ego_centerline[:, 1], label="Ego Centerline", linestyle="--", color="blue")
    plt.plot(ctrl_centerline[:, 0], ctrl_centerline[:, 1], label="Ctrl Centerline", linestyle="--", color="red")

    # Plot the conflict point
    plt.scatter(*conflict_point, color="black", label="Conflict Point", zorder=5)

    # Label the distances to the conflict point
    # Label the distances to the conflict point
    plt.text(
        (ego_trajectory[-1, 0] + conflict_point[0]) / 2,
        (ego_trajectory[-1, 1] + conflict_point[1]) / 2,
        f"Ego to Conflict: {ego2conflict_distance:.2f} m",
        color="blue",
    )

    plt.text(
        (ctrl_trajectory[-1, 0] + conflict_point[0]) / 2,
        (ctrl_trajectory[-1, 1] + conflict_point[1]) / 2,
        f"Ctrl to Conflict: {ctrl2conflict_distance:.2f} m",
        color="red",
    )

    # Add labels, legend, and grid
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig("intersect.png")


def define_zone_boundaries(ego_pos_global, ego_yaw, ego_length):
    # Calculate the corners of the ego car
    half_length = ego_length / 2
    corners = [
        (ego_pos_global[0] + half_length * torch.cos(ego_yaw), ego_pos_global[1] + half_length * torch.sin(ego_yaw)),
        (ego_pos_global[0] - half_length * torch.cos(ego_yaw), ego_pos_global[1] - half_length * torch.sin(ego_yaw)),
    ]
    return corners


def angle_difference_loss(actual_angle, target_angle):
    # Calculate the cosine and sine of both actual and target angles

    actual_cos = torch.cos(actual_angle)
    actual_sin = torch.sin(actual_angle)
    target_cos = torch.cos(target_angle)
    target_sin = torch.sin(target_angle)

    # Compute the cosine and sine differences
    cos_diff = actual_cos - target_cos
    sin_diff = actual_sin - target_sin

    # Calculate the squared errors
    cos_squared_error = cos_diff**2
    sin_squared_error = sin_diff**2

    # Sum the squared errors to get the total loss
    total_loss = cos_squared_error + sin_squared_error

    return total_loss


def smooth_zone_penalty_loss(
    relative_vec, ego_yaw, ego_extent, ctrl_extent, target_zone="front", distance_threshold=100.0
):
    """
    Calculates a differentiable penalty based on whether another vehicle is in the desired zone of the ego vehicle,
    after mapping the relative vector to the ego-centric coordinate system using the ego vehicle's yaw.

    :param relative_vec: The relative position vector of the other vehicle with respect to the ego vehicle.
    :param ego_yaw: The yaw angle of the ego vehicle.
    :param ego_extent: The length,width of the ego vehicle, used to define the zones.
    :param target_zone: The zone in which we want the other vehicle to be ('front', 'middle', 'back').
    :return: A differentiable loss tensor that penalizes the other vehicle based on its position in the ego-centric coordinates.
    """
    # Rotate the relative vector into the ego vehicle's coordinate frame
    # plot the zone first
    # plot_ego_centric_zone_with_relative_vec(relative_vec, ego_yaw, vehicle_length)
    distances = torch.norm(relative_vec, dim=-1)
    close_enough = distances < distance_threshold

    ego_yaw = ego_yaw.squeeze()

    cos_yaw = torch.cos(-ego_yaw)
    sin_yaw = torch.sin(-ego_yaw)
    ego_centric_x = relative_vec[..., 0] * cos_yaw - relative_vec[..., 1] * sin_yaw
    ego_centric_y = relative_vec[..., 0] * sin_yaw + relative_vec[..., 1] * cos_yaw
    # Define the boundaries of the zones based on the vehicle length
    front_zone_boundary = ego_extent[0] / 2
    back_zone_boundary = -ego_extent[0] / 2

    # Depending on the target zone, calculate the penalty
    if target_zone == "front":
        # Check if the other vehicle is in the front zone
        in_front_zone = ego_centric_x > front_zone_boundary

        # Penalize for being behind the front boundary
        penalty_out_of_zone = torch.relu(front_zone_boundary - ego_centric_x)

        # Encourage collision when in the front zone
        collision_penalty_x = torch.relu(-ego_centric_x) * in_front_zone
        collision_penalty_y = torch.relu(ego_extent[1] / 2 - torch.abs(ego_centric_y)) * in_front_zone

        total_penalty = penalty_out_of_zone + collision_penalty_x + collision_penalty_y

    elif target_zone == "middle":
        # Check if the other vehicle is within the middle zone
        in_middle_zone = (ego_centric_x > back_zone_boundary) & (ego_centric_x < front_zone_boundary)

        # Penalize for being outside the middle zone
        penalty_out_of_zone = torch.relu(back_zone_boundary - ego_centric_x) + torch.relu(
            ego_centric_x - front_zone_boundary
        )

        # Encourage collision when in the middle zone
        collision_penalty_x = (
            torch.relu(torch.min(ego_centric_x - back_zone_boundary, front_zone_boundary - ego_centric_x))
            * in_middle_zone
        )
        collision_penalty_y = torch.relu(ego_extent[1] / 2 - torch.abs(ego_centric_y)) * in_middle_zone

        total_penalty = penalty_out_of_zone + collision_penalty_x + collision_penalty_y

    elif target_zone == "back":
        # Check if the other vehicle is in the back zone
        in_back_zone = ego_centric_x < back_zone_boundary

        # Penalize for being ahead of the back boundary
        penalty_out_of_zone = torch.relu(ego_centric_x - back_zone_boundary)

        # Encourage collision when in the back zone
        collision_penalty_x = torch.relu(-back_zone_boundary - ego_centric_x) * in_back_zone
        collision_penalty_y = torch.relu(ego_extent[1] / 2 - torch.abs(ego_centric_y)) * in_back_zone

        total_penalty = penalty_out_of_zone + collision_penalty_x + collision_penalty_y

    # The penalty is zero if the vehicle is within the target zone and increases outside of it
    # Calculate the penalty based on lateral position
    # penalty_loss_y = ego_centric_y ** 2
    total_penalty_loss = total_penalty * close_enough.float()

    return total_penalty_loss * 10


def smooth_zone_collision_loss(
    relative_vec, ego_yaw, ego_extent, ctrl_extent, target_zone="front", collision_margin=0.2
):
    """
    Calculates a differentiable penalty that encourages another vehicle to collide with specific points
    (front, back, or middle) of the ego vehicle, based on the target zone.
    The function maps the relative vector to the ego-centric coordinate system using the ego vehicle's yaw.

    :param relative_vec: The relative position vector of the other vehicle with respect to the ego vehicle.
    :param ego_yaw: The yaw angle of the ego vehicle.
    :param ego_extent: The length and width of the ego vehicle, used to define the collision points.
    :param target_zone: The zone ('front', 'back', or 'middle') where collision is encouraged.
    :param collision_margin: The margin around the collision points.
    :return: A differentiable loss tensor that encourages collision at the specified points.
    """

    ego_yaw = ego_yaw.squeeze()
    cos_yaw = torch.cos(-ego_yaw)
    sin_yaw = torch.sin(-ego_yaw)
    ego_centric_x = relative_vec[..., 0] * cos_yaw - relative_vec[..., 1] * sin_yaw
    ego_centric_y = relative_vec[..., 0] * sin_yaw + relative_vec[..., 1] * cos_yaw

    # Define collision points based on the target zone
    front_collision_point = ego_extent[0] / 3
    back_collision_point = -ego_extent[0] / 3
    middle_collision_point = 0

    # Initialize the penalty
    penalty_loss = torch.zeros_like(ego_centric_x)

    # Calculate the mask for each zone
    front_zone_mask = torch.abs(ego_centric_x - front_collision_point) < 2.0
    back_zone_mask = torch.abs(ego_centric_x - back_collision_point) < 2.0
    middle_zone_mask = torch.abs(ego_centric_x - middle_collision_point) < 2.0

    # # Depending on the target zone, calculate the penalty
    # if target_zone == ß'front':
    #     penalty_loss = torch.relu(torch.abs(ego_centric_x - front_collision_point) - collision_margin)
    #     penalty_loss_y = torch.relu(torch.abs(ego_centric_y) - ego_extent[1]/2) * front_zone_mask
    # elif target_zone == 'back':
    #     penalty_loss = torch.relu(torch.abs(ego_centric_x - back_collision_point) - collision_margin)
    #     penalty_loss_y = torch.relu(torch.abs(ego_centric_y) - ego_extent[1]/2) * back_zone_mask
    # elif target_zone == 'middle':
    #     penalty_loss = torch.relu(torch.abs(ego_centric_x - middle_collision_point) - collision_margin)
    #     penalty_loss_y = torch.relu(torch.abs(ego_centric_y) - ego_extent[1]/2) * middle_zone_mask

    # Lateral penalty based on proximity to the target zone
    penalty_loss_y = torch.relu(torch.abs(ego_centric_y) - ego_extent[1] / 2)

    total_collision_penalty = penalty_loss + penalty_loss_y
    return total_collision_penalty


def plot_ego_centric_zone_with_relative_vec(relative_vec, ego_yaw, vehicle_length):
    """
    Plots the relative vector position of another vehicle with respect to the ego vehicle's coordinate frame.
    Also, displays the different penalty zones around the ego vehicle.

    :param relative_vec: The relative position vector of the other vehicle with respect to the ego vehicle.
    :param ego_yaw: The yaw angle of the ego vehicle.
    :param vehicle_length: The length of the ego vehicle, used to define the zones.
    """
    # Convert ego_yaw to a tensor for compatibility with torch operations
    ego_yaw = ego_yaw.squeeze()

    # Rotate the relative vector into the ego vehicle's coordinate frame
    cos_yaw = torch.cos(-ego_yaw)
    sin_yaw = torch.sin(-ego_yaw)
    ego_centric_x = relative_vec[..., 0] * cos_yaw - relative_vec[..., 1] * sin_yaw
    ego_centric_y = relative_vec[..., 0] * sin_yaw + relative_vec[..., 1] * cos_yaw

    # Define the boundaries of the zones based on the vehicle length
    front_zone_boundary = vehicle_length / 2
    back_zone_boundary = -vehicle_length / 2

    # Start plotting
    fig, ax = plt.subplots()

    # Plot ego vehicle as a rectangle
    ego_vehicle = plt.Rectangle((-vehicle_length / 2, -1), vehicle_length, 2, color="blue", label="Ego Vehicle")
    ax.add_patch(ego_vehicle)

    # Plot zones as rectangles
    front_zone = plt.Rectangle((front_zone_boundary, -3), 10, 6, color="green", alpha=0.3, label="Front Zone")
    middle_zone = plt.Rectangle(
        (back_zone_boundary, -3), vehicle_length, 6, color="yellow", alpha=0.3, label="Middle Zone"
    )
    back_zone = plt.Rectangle((-10, -3), 10 - back_zone_boundary, 6, color="red", alpha=0.3, label="Back Zone")

    ax.add_patch(front_zone)
    ax.add_patch(middle_zone)
    ax.add_patch(back_zone)

    # Plot the relative vector position
    plt.scatter(
        ego_centric_x.cpu().detach().numpy(), ego_centric_y.cpu().detach().numpy(), c="black", label="Other Vehicle"
    )

    # Set plot limits and labels
    ax.set_xlim(-50, 50)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("Ego-centric X")
    ax.set_ylabel("Ego-centric Y")
    ax.set_title("Ego-centric Zones with Relative Vehicle Position")
    ax.legend()

    # Show grid
    ax.grid(True)

    # Display the plot
    plt.savefig("zone_loss")


def calculate_exact_speed_penalty(distance, ego_speed, ctrl_speed, exact_diff, distance_threshold=5.0, margin=0.0):
    """
    Calculates a penalty if the controlled vehicle's speed does not match the ego vehicle's speed minus an exact difference.

    :param ego_speed: The speed of the ego vehicle.
    :param ctrl_speed: The speed of the controlled vehicle.
    :param exact_diff: The exact difference the controlled vehicle's speed should be below the ego vehicle's speed.
    :return: A tensor representing the speed penalty.
    """
    # Calculate the target speed for the controlled vehicle
    close_enough = distance < distance_threshold

    target_speed = ego_speed + exact_diff

    # Calculate the difference from the target speed
    speed_difference = torch.abs(ctrl_speed - target_speed)

    # Apply penalty only if the difference is outside the allowable margin
    outside_margin = speed_difference > margin
    speed_penalty = (speed_difference - margin) * close_enough.float() * outside_margin.float()

    return speed_penalty


import matplotlib.pyplot as plt
import numpy as np


def plot_vehicle(ax, position, yaw, extent, label, color):
    """
    Helper function to plot a vehicle as a bounding box at its initial position.

    :param ax: The matplotlib axis to plot on.
    :param position: The trajectory of the vehicle (shape: T, 2).
    :param yaw: The yaw angle of the vehicle at the initial position.
    :param extent: The length and width of the vehicle.
    :param label: Label for the plot.
    :param color: Color of the bounding box.
    """
    # Calculate the vehicle's corners for bounding box at the initial position
    dx = extent[0] / 2 * np.cos(yaw[0])
    dy = extent[0] / 2 * np.sin(yaw[0])
    corners = np.array(
        [
            [position[0, 0] + dx, position[0, 1] + dy],
            [position[0, 0] - dx, position[0, 1] - dy],
            [position[0, 0] - dx, position[0, 1] + dy],
            [position[0, 0] + dx, position[0, 1] - dy],
        ]
    )

    # Draw the bounding box
    ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]], label=label, color=color)
    ax.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]], color=color)
    ax.plot([corners[0, 0], corners[3, 0]], [corners[0, 1], corners[3, 1]], color=color)
    ax.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]], color=color)
    # draw trajectory
    ax.plot(position[:, 0], position[:, 1], color=color, marker="x")


def plot_vehicle_with_heading(ax, position, yaw, color, length=1.0):
    """
    Plots a point for the vehicle's position and an arrow to indicate its heading direction.

    :param ax: The matplotlib axis to plot on.
    :param position: The position of the vehicle (x, y).
    :param yaw: The yaw angle of the vehicle.
    :param color: Color of the plot.
    :param length: Length of the heading arrow.
    """
    ax.plot(position[0], position[1], marker="o", color=color)
    ax.arrow(
        position[0],
        position[1],
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=0.1,
        head_length=0.2,
        fc=color,
        ec=color,
    )


def visualize_trajectories_and_distances(
    pred_ego_pos, ego_yaw, ego_plan, ego_extent, ctrl_yaw, ctrl_pos, ctrl_extent, dist_vec
):
    """
    Visualize ego and controlled vehicles' plans and the distance vectors between them.
    Converts tensors to numpy arrays for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert tensors to numpy if they are not already
    pred_ego_pos = pred_ego_pos.detach().cpu().numpy()
    ego_yaw_np = ego_yaw.detach().cpu().numpy()
    ego_plan_np = ego_plan.detach().cpu().numpy()
    ctrl_yaw_np = [y.detach().cpu().numpy() for y in ctrl_yaw]
    ctrl_pos_np = [p.detach().cpu().numpy() for p in ctrl_pos]
    dist_vec_np = dist_vec.detach().cpu().numpy()
    ego_extent = ego_extent.detach().cpu().numpy()
    ctrl_extent = ctrl_extent.detach().cpu().numpy()
    # Plot ego vehicle trajectory and initial position
    plot_vehicle(ax, ego_plan_np, ego_yaw_np[-1, 0], ego_extent, "Ego Vehicle", "blue")

    plot_vehicle(ax, pred_ego_pos[0], ego_yaw_np[-1, 0], ego_extent, "Ego Predict", "orange")
    # Plot controlled vehicles trajectories and initial positions
    for i, (pos, yaw) in enumerate(zip(ctrl_pos_np, ctrl_yaw_np)):
        ax.plot(pos[:, 0], pos[:, 1], marker="o")
        plot_vehicle_with_heading(ax, ctrl_pos_np[i][0], ctrl_yaw_np[i][0].item(), "green")

    # Plot distance vectors at the last timestep
    # Plot distance vectors
    for i in range(dist_vec_np.shape[0]):
        for t in range(dist_vec_np.shape[1]):
            vec = dist_vec_np[i, t]
            ego_pos = ego_plan_np[t]
            ax.arrow(ego_pos[0], ego_pos[1], vec[0], vec[1], head_width=0.05, head_length=0.1, fc="red", ec="red")
        break
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Ego and Controlled Vehicles with Distance Vectors")
    ax.legend()
    ax.grid(True)
    plt.savefig("egoctrl.png")
    plt.show()


# Example usage
# Define your input variables according to the specified shapes


# Example usage
# Define your input variables according to the specified shapes



def find_current_lane_idx(centerline, current_pos):
    """
    Find the index of the closest point on the centerline to the current position.
    """
    distances = torch.norm(centerline - current_pos.unsqueeze(0), dim=-1)
    return torch.argmin(distances)
