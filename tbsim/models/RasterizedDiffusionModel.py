"""Variants of Conditional Variational Autoencoder (C-VAE)"""
from typing import Dict, List
import torch
import torch.nn as nn
from tbsim.utils.config_utils import update_config
import tbsim.models.base_models as base_models

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.guidance_utils import initialize_loss_calculator
from tbsim.utils.adv_utils import generate_collision_paths
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.trajdata_utils import enlarge_batch_samples, extend_ego_plan, get_stationary_mask

from tbsim.models.temporal import TemporalUnet

from tbsim.models.diffusion import DiffusionTraj,VarianceSchedule



class RasterizedDiffusionModel(nn.Module):
    """Raster-based diffusion model for controllable traffic simulation.
     """
    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            dynamics_config:tuple,
            history_encoder_config,
            diffnet_config:tuple,
            diffuse_config:Dict,
            weights_scaling: List[float],
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
            rasterize_mode = "point",
            drop_cond_prob = 0,
            do_guidance = False,
            guide_config = None,
    ) -> None:
    
        super().__init__()
        if rasterize_mode is None:
            rasterize_mode = "point"
        assert rasterize_mode in ["point","square"]
        self.rasterize_mode = rasterize_mode
        self.drop_cond_prob = drop_cond_prob
        self.weights_scaling = weights_scaling

        self.map_encoder = base_models.RasterizedMapEncoder(
                model_arch=model_arch,
                input_image_shape=input_image_shape,
                feature_dim=map_feature_dim,
                use_spatial_softmax=use_spatial_softmax,
                spatial_softmax_kwargs=spatial_softmax_kwargs,
                output_activation=nn.ReLU
            )
        self.agent_history_encoder = base_models.HistoryEncoder(history_encoder_config)
        self.other_history_encoder =  base_models.HistoryEncoder(history_encoder_config)
        
        self.diffnet = TemporalUnet(**diffnet_config,dynamics_config=dynamics_config) ##TODO config of TemperalNet
        self.diffusion = DiffusionTraj(
            net = self.diffnet,
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='cosine'
            )
        )
        self.diffuse_args = diffuse_config
        for key in ["num_samples", "sample_step","sampling_mode"]:
            self.diffuse_args[key] = guide_config.params[key]
        self.do_guidance  = do_guidance
        self.guide_config = guide_config
        self.gen_adv_trajs = guide_config.params.partial_t is not None # Check if partial_t has been set
        if self.do_guidance:
            self.Loss_Calculater = initialize_loss_calculator(self.guide_config)


    def forward(self, data_batch, guide_sample_fn=None, data_batch_for_guidance=None)-> Dict[str, torch.Tensor]:
        #whether to control stationary agents
        stationary_mask = get_stationary_mask(data_batch, disable_control_on_stationary="on_lane")
        ## Calculating conditioning feature
        cond_feat = self._encode_history(data_batch)
        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.diffnet.dyn.type() if self.diffnet.dyn is not None else 0 )

        actions = self.diffusion.sample(
            cond_feat,
            forward_mode="sample",
            guide_sample_fn=guide_sample_fn,
            data_batch_for_guidance=data_batch_for_guidance,
            current_states=curr_states,
            guide_config=self.guide_config,  # guide_config for controllable simulation
            adv_proposals_dict=(data_batch["adv_proposals_dict"] 
                              if "adv_proposals_dict" in data_batch 
                              else None),  # B * 20 * 12 * 2
            **self.diffuse_args.to_dict(),
        )
        if isinstance(actions, dict):
            actions, denoised_actions = actions["sampled_trajectories"], actions["denoised_trajectories"]
            actions = actions * (~stationary_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if denoised_actions is not None:
                denoised_actions = denoised_actions * (~stationary_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            denoised_actions = None
        if self.diffnet.dyn is not None:
            out_dict = self._forward_dynamics(actions,curr_states)
            if denoised_actions is not None:
                denoised_out_dict = self._forward_dynamics(denoised_actions,curr_states)
                out_dict["denoising_predictions"] = {
                    "positions": denoised_out_dict["predictions"]["positions"],
                    "yaws": denoised_out_dict["predictions"]["yaws"],
                    # "adv_proposals": data_batch["adv_proposals_dict"]["padded_adv_proposals_pos"] if "adv_proposals_dict" in data_batch else None #padded for bug in logger
                }
            else:
                out_dict["denoising_predictions"] = {}
            if "adv_proposals_dict" in data_batch and data_batch["adv_proposals_dict"]["padded_adv_proposals_pos"] is not None:
                out_dict["denoising_predictions"]["adv_proposals"] = data_batch["adv_proposals_dict"]["padded_adv_proposals_pos"]
        else:
            raise NotImplementedError

        return out_dict
       
    def update_guide_config(self, update_guide_config, device):
        update_config(self.guide_config, update_guide_config.guide_config)
        self.Loss_Calculater.to_device(device)
        self.Loss_Calculater.update_config(self.guide_config)

    def sample(self, data_batch) -> Dict[str, torch.Tensor]:
        """Generate trajectory samples, optionally with guidance.
        
        Args:
            data_batch (Dict): Input batch containing scene and agent information
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predicted trajectories and states
        """
        if not self.do_guidance:
            return self.forward(data_batch, guide_sample_fn=None)
        
        # Prepare guidance data
        data_batch_for_guidance = self._prepare_guidance_data(data_batch)
        
        # Generate adversarial trajectories if needed
        if self.gen_adv_trajs:
            adv_proposals_dict = generate_collision_paths(
                data_batch,
                self.guide_config["batch_ego_indices"],
                self.guide_config["batch_ctrl_indices"],
                self.guide_config.params.desired_delta_s,
                self.guide_config.params.normal_offset,
                self.guide_config.params.ref_idx,
                T=self.diffuse_args["num_points"],
                dt=data_batch["dt"][0].item()
            )
            data_batch_for_guidance.update(adv_proposals_dict)
        
        # Run forward pass with guidance
        out_dict = self.forward(
            data_batch,
            guide_sample_fn=self.Loss_Calculater,
            data_batch_for_guidance=data_batch_for_guidance
        )
        
        return out_dict

    def _prepare_guidance_data(self, data_batch) -> Dict:
        """Prepare data needed for guided sampling.
        
        Args:
            data_batch (Dict): Input batch containing scene and agent information
            
        Returns:
            Dict: Processed data batch for guidance
        """
        # Calculate drivable region and distance maps
        drivable_map = batch_utils().get_drivable_region_map(data_batch["image"]).float()
        dis_map = GeoUtils.calc_distance_map(drivable_map)
        
        # Calculate batch dimensions
        batch_size = data_batch["dt"].shape[0]
        BN = batch_size * self.diffuse_args["num_samples"]
        
        guidance_data = {
            "batch_size": batch_size,
            "centroid": data_batch["centroid"],
            "curr_speed": data_batch["curr_speed"],
            "yaw": data_batch["yaw"],
            "raster_from_agent": data_batch["raster_from_agent"],
            "dis_map": dis_map,
            "agent_fut_extent": data_batch["agent_fut_extent"][:,0,:2],
            "centerline": data_batch["extras"]["centerline_xy"],
            "agent_pos": data_batch["world_from_agent"][:,:2,-1],
            "lane_avail": data_batch["extras"]["has_lane"],
            "ego_extents": data_batch["agent_fut_extent"][:,0,:2],
            "raw_types": data_batch["all_other_agents_types"],
            "world_from_agent": data_batch["world_from_agent"],
            "scene_ids": data_batch.get("scene_index", data_batch.get("scene_ids")),
            "ego_plan": data_batch.get("ego_plan"),
            "BN": BN,
            "num_samples": self.diffuse_args["num_samples"],
            "dt": data_batch["dt"][0].item()
        }
        
        # Adjust shapes for batch processing
        self._adjust_batch_shapes(guidance_data, batch_size)
        
        return guidance_data

    def _adjust_batch_shapes(self, guidance_data: Dict, batch_size: int):
        """Adjust shapes of guidance data for batch processing.
        
        Args:
            guidance_data (Dict): Data to be adjusted
            batch_size (int): Base batch size
        """
        keys_to_adjust = ["centerline", "lane_avail","world_from_agent","yaw", "raster_from_agent","dis_map"]
        
        for key in keys_to_adjust:
            guidance_data[key] = enlarge_batch_samples(
                guidance_data[key],
                batch_size,
                num_samples=self.diffuse_args["num_samples"]
            )
        # Extend ego plan if present
        if guidance_data["ego_plan"] is not None:
            guidance_data["ego_plan"] = extend_ego_plan(
                guidance_data["ego_plan"],
                target_length=self.diffuse_args["num_points"]
            )

    
    def _forward_dynamics(self,actions,curr_states) -> Dict[str,torch.Tensor]:
        #TODO if actions is more than 1 sample, need to do in batch
        if  len(actions.shape) == 3:
            traj, x    =  self.diffusion.net._forward_dynamics(actions=actions.squeeze(), current_states=curr_states)
        else:
            traj, x    =  self.diffusion.net._forward_dynamics(actions=actions, current_states=curr_states)
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:]
        
        out_dict = {
            "states": x,
            "controls": actions,
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

        return out_dict

    def _encode_history(self,data_batch):
        ## Calculating conditioning feature
        map_feat = self.map_encoder(data_batch["image"])
        target_traj_feat = self.agent_history_encoder(data_batch["agent_hist"]).squeeze(1)
        other_traj_feat = self.other_history_encoder(data_batch["neigh_hist"].to(data_batch["agent_hist"].device)).squeeze(1)

        cond_feat = torch.cat([map_feat,target_traj_feat,other_traj_feat],dim = -1)
        return cond_feat

