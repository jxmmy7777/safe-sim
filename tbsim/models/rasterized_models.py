from turtle import forward
from typing import Dict, List
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import random

import tbsim.models.base_models as base_models
import tbsim.dynamics as dynamics
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation, DynOrnsteinUhlenbeckPerturbation
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.trajdata_utils as TrajdataUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.models.unet import UNet, DirectRegressionPostProcessor

from tbsim.models.roi_align import ROI_align, generate_ROIs, Indexing_ROI_result


class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
            output_frenet = False,
            use_ROI = False,
            roi_config = None,
    ) -> None:

        super().__init__()
        self.use_ROI = use_ROI
        if use_ROI:
            self.map_encoder = base_models.RasterizeROIEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,  # [C, H, W]
            global_feature_dim=roi_config.global_feature_dim,
            agent_feature_dim=roi_config.agent_feature_dim,
            output_activation=nn.ReLU,
            use_rotated_roi=roi_config.use_rotated_roi,
            roi_layer_key=roi_config.roi_layer_key
        )
            self.use_rotated_roi = roi_config.use_rotated_roi
            roi_size = roi_config.roi_size
            assert len(roi_size) == 2
            self.roi_size = nn.Parameter(
                torch.Tensor([roi_size[0], roi_size[0], roi_size[1],
                            roi_size[1]]),  # [W1, W2, H1, H2]
                requires_grad=False
            )
            self.roi_config = roi_config
            self.transformer = None

        else:    
            self.map_encoder = base_models.RasterizedMapEncoder(
                model_arch=model_arch,
                input_image_shape=input_image_shape,
                feature_dim=map_feature_dim,
                use_spatial_softmax=use_spatial_softmax,
                spatial_softmax_kwargs=spatial_softmax_kwargs,
                output_activation=nn.ReLU
            )
        self.traj_decoder = trajectory_decoder
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        self.output_frenet = output_frenet
        
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    
        if self.use_ROI:
            map_feat = self.extract_features(data_batch).squeeze()
        else:
            map_feat = self.map_encoder(data_batch["image"])

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
            if  self.traj_decoder.dyn.type() == 5: #frenet
                self.traj_decoder.dyn.update_centerline(data_batch["extras"])
        else:
            curr_states = None
        dec_output = self.traj_decoder.forward(inputs=map_feat, current_states=curr_states)
        traj = dec_output["trajectories"]
        ## --------------if output frenet-----------
        # if self.output_frenet:
        #     out_dict = transform_nt2xy(traj,data_batch["extras"])
        #     return out_dict
        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]
            out_dict["curr_states"] = curr_states
            out_dict["states"] = dec_output["states"]
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        if self.output_frenet:
            # target_traj = torch.cat((data_batch["extras"]["nt_delta_target"], 
            #                          torch.zeros_like(data_batch["extras"]["nt_delta_target"][:,:,0:1])), dim=2)
            
            # pred_loss = trajectory_loss(
            #     predictions=data_batch["dt"][:,None,None]* pred_batch["controls"],
            #     targets= data_batch["extras"]["nt_delta_target"],
            #     availabilities=data_batch["target_availabilities"],
            #     weights_scaling=self.weights_scaling
            # )
            target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
            
            prediction_loss_xy = trajectory_loss(
                predictions=pred_batch["trajectories"],
                targets=target_traj,
                availabilities=data_batch["target_availabilities"],
            )

            losses = OrderedDict(
                prediction_loss=prediction_loss_xy,
                # prediction_loss_xy = prediction_loss_xy
                # goal_loss=goal_loss,
                # collision_loss=coll_loss
            )
        else:
            target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
            pred_loss = trajectory_loss(
                predictions=pred_batch["trajectories"],
                targets=target_traj,
                availabilities=data_batch["target_availabilities"],
                weights_scaling=self.weights_scaling
            )
            goal_loss = goal_reaching_loss(
                predictions=pred_batch["trajectories"],
                targets=target_traj,
                availabilities=data_batch["target_availabilities"],
                weights_scaling=self.weights_scaling
                )
            losses = OrderedDict(
                prediction_loss=pred_loss,
                goal_loss=goal_loss,
                # collision_loss=coll_loss
            )

        # compute collision loss
        # pred_edges = batch_utils().get_edges_from_batch(
        #     data_batch=data_batch,
        #     ego_predictions=pred_batch["predictions"]
        # )
        #
        # coll_loss = collision_loss(pred_edges=pred_edges)
       
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses
    ## Adding this for closed-loop rotated ROI
    def extract_features(self, data_batch, return_encoder_feats=False, extract_ego_only = True):
        image_batch = data_batch["image"]
        if extract_ego_only:
            states_all = {
                "history_positions":data_batch["history_positions"].unsqueeze(1),
                "history_yaws":data_batch["history_yaws"].unsqueeze(1),
                "history_availabilities":data_batch["history_availabilities"].unsqueeze(1),
            }
        else:
            states_all = batch_utils().batch_to_raw_all_agents(
                data_batch, step_time=None)
        
        b, a = states_all["history_positions"].shape[:2]

        # extract agent-wise features
        if self.use_rotated_roi:
            rois, indices = self._get_roi_boxes_rotated(
                pos=states_all["history_positions"],
                yaw=states_all["history_yaws"],
                avails=states_all["history_availabilities"],
                trans_mat=data_batch["raster_from_agent"],
                patch_size=self.roi_size
            )

            all_feats, _, global_feats, encoder_feats = self.map_encoder(
                image_batch, rois=rois)  # approximately B * A
            split_sizes = [len(l) for l in indices]
            all_feats_list = torch.split(all_feats, split_sizes)
            all_feats = Indexing_ROI_result(
                all_feats_list, indices, emb_size=(b, a, all_feats.shape[-1]))
            assert torch.isfinite(all_feats).all(), "Found non-finite values in all_feats"

        else:
            curr_pos_all = torch.cat((
                data_batch["history_positions"].unsqueeze(1),
                data_batch["all_other_agents_history_positions"],
            ), dim=1)[:, :, 0]  # histories are reversed
            rois = self._get_roi_boxes_upright(
                curr_pos_all,
                trans_mat=data_batch["raster_from_agent"],
                patch_size=self.roi_size[[0, 2]]
            ) #[ B * A, 5] first dim is indices
            all_feats, _, global_feats, encoder_feats = self.map_encoder(
                image_batch, rois=rois)

        # tile global feature and concat w/ agent-wise features
        all_feats = all_feats.reshape(b, a, -1)
        if not extract_ego_only: # if closed-loop we need to crop agent feat all the time, maybe don't cat global?
            all_feats = torch.cat(
                (all_feats, TensorUtils.unsqueeze_expand_at(global_feats, a, 1)), dim=-1)

        if self.roi_config.history_conditioning:
            hist_traj = torch.cat((states_all["history_positions"], states_all["history_yaws"]), dim=-1)
            hist_feats = TensorUtils.time_distributed(hist_traj, self.history_encoder)
            all_feats = torch.cat((all_feats, hist_feats), dim=-1)

        # optionally pass information using transformer
        if self.transformer is not None:
            all_feats = self.transformer(
                all_feats,
                states_all["history_availabilities"][:, :, -1],
                states_all["history_positions"][:, :, -1]
            )

        if not return_encoder_feats:
            return all_feats
        else:
            return all_feats, encoder_feats
    def _get_roi_boxes_upright(self, pos, trans_mat, patch_size):
        b, a = pos.shape[:2]
        curr_pos_raster = transform_points_tensor(pos, trans_mat.float())
        extents = torch.ones_like(curr_pos_raster) * patch_size  # [B, A, 2]
        rois_raster = get_upright_box(
            curr_pos_raster, extent=extents).reshape(b * a, 2, 2)
        rois_raster = torch.flatten(rois_raster, start_dim=1)  # [B * A, 4]

        roi_indices = torch.arange(0, b).unsqueeze(
            1).expand(-1, a).reshape(-1, 1).to(rois_raster.device)  # [B * A, 1]
        indexed_rois_raster = torch.cat(
            (roi_indices, rois_raster), dim=1)  # [B * A, 5]
        return indexed_rois_raster

    def _get_roi_boxes_rotated(self, pos, yaw, avails, trans_mat, patch_size):
        rois, indices = generate_ROIs(
            pos, yaw, trans_mat, avails, patch_size, mode="last")
        return rois, indices
