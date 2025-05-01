from collections import OrderedDict
from random import sample
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities import move_data_to_device
import torch.nn.functional as F
from tbsim import dynamics
import tbsim.utils.torch_utils as TorchUtils
import tbsim.utils.geometry_utils as GeoUtils

from tbsim.models.rasterized_models import RasterizedPlanningModel
from tbsim.models.RasterizedDiffusionModel import RasterizedDiffusionModel
from tbsim.models.base_models import (
    MLPTrajectoryDecoder,
    RasterizedMapUNet,
)


import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Plan, Action
import tbsim.algos.algo_utils as AlgoUtils
from tbsim.utils.geometry_utils import transform_points_tensor


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb


class BehaviorCloning(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(BehaviorCloning, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=algo_config.state_dim,
            num_steps=algo_config.prediction_length,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0]*algo_config.state_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
            output_frenet = algo_config.output_frenet,
            use_ROI = algo_config.use_ROI,
            roi_config = algo_config.roi_config,

        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        if self.algo_config.closed_loop_train:
            predict_length = pred_batch["predictions"]["positions"].shape[-2]
            gt = gt[:,:predict_length]
            avail = avail[:,:predict_length]

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)


        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """

        batch = batch_utils().parse_batch(batch)
        
        pout = self.nets["policy"](batch)


        losses = self.nets["policy"].compute_losses(pout, batch)

        total_loss = 0.0
        for lk, l in losses.items():
            if lk in ["a","w","x","y","v","yaw"]:
                continue
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)
        
        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)

        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}
class DiffusionTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(DiffusionTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        self.nets["policy"] = RasterizedDiffusionModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,

            dynamics_config = {
                            'feature_dim':algo_config.map_feature_dim,
                            'state_dim': 3,
                            'num_steps': algo_config.horizon,
                            'dynamics_type': algo_config.dynamics.type,
                            'dynamics_kwargs': algo_config.dynamics,
                            'step_time': algo_config.step_time
                        },
            history_encoder_config      = algo_config.history_encoder_config,
            # ref_encoder_config  = algo_config.ref_encoder_config,
            # nt_history_encoder_config   = algo_config.nt_history_encoder_config,
            diffnet_config = {"horizon": algo_config.horizon,
                                "transition_dim": algo_config.transition_dim,
                                "context_dim": algo_config.context_dim,
                                "scale_net_output": algo_config.get("scale_net_output", False)
                            },
            diffuse_config= algo_config.diffuse_config,
            weights_scaling= algo_config.weight_scaling,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
            rasterize_mode = algo_config.rasterize_mode if hasattr(algo_config,"rasterize_mode") else "point",
            drop_cond_prob=algo_config.drop_cond_prob,
            do_guidance =  algo_config.get("do_guidance", False),
            guide_config  = algo_config.guide_config
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}
    
    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])
        if self.algo_config.closed_loop_train:
            predict_length = sample_batch["predictions"]["positions"].shape[-2]
            gt = gt[:,:predict_length]
            avail = avail[:,:predict_length]
        # compute ADE & FDE based on posterior params

        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"].squeeze())
        # if len(recon_preds.shape) ==3:
        #     metrics["ego_ADE"] = Metrics.single_mode_metrics(
        #         Metrics.batch_average_displacement_error, gt, recon_preds, avail
        #     ).mean()
        #     metrics["ego_FDE"] = Metrics.single_mode_metrics(
        #         Metrics.batch_final_displacement_error, gt, recon_preds, avail
        #     ).mean()

        # compute ADE & FDE based on trajectory samples
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()
        # print(metrics)
        return metrics

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        batch = batch_utils(rasterize_mode = self.nets["policy"].rasterize_mode).parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].get_loss(batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch)
        metrics = self._compute_metrics(pout, samples, batch)

        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        if len(outputs) ==0:
            return
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}


    def get_action(self, obs_dict, sample=True, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds_dict = self.nets["policy"].sample(obs_dict)
            preds = preds_dict["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds_dict["predictions"]["positions"],
                    yaws=preds_dict["predictions"]["yaws"]
                ).to_dict(),
                change_lane_state =  preds_dict.get("change_lane_state", None),
                #add visualization information
                denoising_predictions = preds_dict.get("denoising_predictions", None),
                # adv_proposals = preds_dict.get("adv_proposals", None),
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info

class BehaviorCloningGC(BehaviorCloning):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        pl.LightningModule.__init__(self)
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + algo_config.goal_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        self.nets["policy"] = RasterizedGCModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            goal_feature_dim=algo_config.goal_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    def get_action(self, obs_dict, **kwargs):
        obs_dict = dict(obs_dict)
        if "plan" in kwargs:
            plan = kwargs["plan"]
            assert isinstance(plan, Plan)
            obs_dict["target_positions"] = plan.positions
            obs_dict["target_yaws"] = plan.yaws
            obs_dict["target_availabilities"] = plan.availabilities
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}


class SpatialPlanner(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(SpatialPlanner, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        self.nets["policy"] = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=4,  # (pixel, x_residual, y_residual, yaw)
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        keys = {"posErr": "val/metrics_goal_pos_err"}
        if self.algo_config.loss_weights.pixel_bce_loss > 0:
            keys["valBCELoss"] = "val/losses_pixel_bce_loss"
        if self.algo_config.loss_weights.pixel_ce_loss > 0:
            keys["valCELoss"] = "val/losses_pixel_ce_loss"
        return keys

    def forward(self, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        pred_map = self.nets["policy"](obs_dict["image"])
        return self.forward_prediction(
            pred_map,
            obs_dict,
            mask_drivable=mask_drivable,
            num_samples=num_samples,
            clearance=clearance
        )

    @staticmethod
    def forward_prediction(pred_map, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        assert pred_map.shape[1] == 4  # [location_logits, residual_x, residual_y, yaw]

        pred_map[:, 1:3] = torch.sigmoid(pred_map[:, 1:3])
        location_map = pred_map[:, 0]

        # get normalized probability map
        location_prob_map = torch.softmax(location_map.flatten(1), dim=1).reshape(location_map.shape)

        if mask_drivable:
            # At test time: optionally mask out undrivable regions
            if "drivable_map" not in obs_dict:
                drivable_map = batch_utils().get_drivable_region_map(obs_dict["image"])
            else:
                drivable_map = obs_dict["drivable_map"]
            for i, m in enumerate(drivable_map):
                if m.sum() == 0:  # if nowhere is drivable, set it to all True's to avoid decoding problems
                    drivable_map[i] = True

            location_prob_map = location_prob_map * drivable_map.float()
        
        prob_sum = location_prob_map.sum([1,2])
        zero_index = torch.where(prob_sum==0)[0]
        if zero_index.nelement()>0:
            location_prob_map[zero_index] = torch.ones_like(location_prob_map[zero_index])
            location_prob_map[zero_index] = location_prob_map[zero_index]/location_prob_map[zero_index].sum([1,2])

        # decode map as predictions
        pixel_pred, res_pred, yaw_pred, pred_prob = AlgoUtils.decode_spatial_prediction(
            prob_map=location_prob_map,
            residual_yaw_map=pred_map[:, 1:],
            num_samples=num_samples,
            clearance = clearance,
        )

        # transform prediction to agent coordinate
        pos_pred = transform_points_tensor(
            (pixel_pred + res_pred),
            obs_dict["agent_from_raster"].float()
        )

        return dict(
            predictions=dict(
                positions=pos_pred,
                yaws=yaw_pred
            ),
            log_likelihood=torch.log(pred_prob),
            spatial_prediction=pred_map,
            location_map=location_map,
            location_prob_map=location_prob_map
        )

    @staticmethod
    def compute_metrics(pred_batch, data_batch):
        metrics = dict()
        goal_sup = data_batch["goal"]
        goal_pred = TensorUtils.squeeze(pred_batch["predictions"], dim=1)

        pos_norm_err = torch.norm(
            goal_pred["positions"] - goal_sup["goal_position"], dim=-1
        )
        metrics["goal_pos_err"] = torch.mean(pos_norm_err)

        metrics["goal_yaw_err"] = torch.mean(
            torch.abs(goal_pred["yaws"] - goal_sup["goal_yaw"])
        )

        pixel_pred = torch.argmax(
            torch.flatten(pred_batch["location_map"], start_dim=1), dim=1
        )  # [B]
        metrics["goal_selection_err"] = torch.mean(
            (goal_sup["goal_position_pixel_flat"].long() != pixel_pred).float()
        )
        metrics["goal_cls_err"] = torch.mean((torch.exp(pred_batch["log_likelihood"]) < 0.5).float())
        metrics = TensorUtils.to_numpy(metrics)
        for k, v in metrics.items():
            metrics[k] = float(v)
        return metrics

    @staticmethod
    def compute_losses(pred_batch, data_batch):
        losses = dict()
        pred_map = pred_batch["spatial_prediction"]
        b, c, h, w = pred_map.shape

        goal_sup = data_batch["goal"]
        # compute pixel classification loss
        location_prediction = pred_map[:, 0]
        losses["pixel_bce_loss"] = torch.binary_cross_entropy_with_logits(
            input=location_prediction,  # [B, H, W]
            target=goal_sup["goal_spatial_map"],  # [B, H, W]
        ).mean()

        losses["pixel_ce_loss"] = torch.nn.CrossEntropyLoss()(
            input=location_prediction.flatten(start_dim=1),  # [B, H * W]
            target=goal_sup["goal_position_pixel_flat"].long(),  # [B]
        )

        # compute residual and yaw loss
        gather_inds = TensorUtils.unsqueeze_expand_at(
            goal_sup["goal_position_pixel_flat"].long(), size=c, dim=1
        )[..., None]  # -> [B, C, 1]

        local_pred = torch.gather(
            input=torch.flatten(pred_map, 2),  # [B, C, H * W]
            dim=2,
            index=gather_inds  # [B, C, 1]
        ).squeeze(-1)  # -> [B, C]
        residual_pred = local_pred[:, 1:3]
        yaw_pred = local_pred[:, 3:4]
        losses["pixel_res_loss"] = torch.nn.MSELoss()(residual_pred, goal_sup["goal_position_residual"])
        losses["pixel_yaw_loss"] = torch.nn.MSELoss()(yaw_pred, goal_sup["goal_yaw"])

        return losses

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.forward(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = self.compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        with torch.no_grad():
            metrics = self.compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = TensorUtils.detach(self.compute_losses(pout, batch))
        metrics = self.compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, mask_drivable=False, sample=False, num_plan_samples=1, clearance=None, **kwargs):
        num_samples = num_plan_samples if sample else None
        preds = self.forward(obs_dict, mask_drivable=mask_drivable, num_samples=num_samples,clearance=clearance)  # [B, num_sample, ...]
        b, n = preds["predictions"]["positions"].shape[:2]
        plan_dict = dict(
            predictions=TensorUtils.unsqueeze(preds["predictions"], dim=1),  # [B, 1, num_sample...]
            availabilities=torch.ones(b, 1, n).to(self.device),  # [B, 1, num_sample]
        )
        # pad plans to the same size as the future trajectories
        n_steps_to_pad = self.algo_config.future_num_frames - 1
        plan_dict = TensorUtils.pad_sequence(plan_dict, padding=(n_steps_to_pad, 0), batched=True, pad_values=0.)
        plan_samples = Plan(
            positions=plan_dict["predictions"]["positions"].permute(0, 2, 1, 3),  # [B, num_sample, T, 2]
            yaws=plan_dict["predictions"]["yaws"].permute(0, 2, 1, 3),  # [B, num_sample, T, 1]
            availabilities=plan_dict["availabilities"].permute(0, 2, 1)  # [B, num_sample, T]
        )

        # take the first sample as the plan
        plan = TensorUtils.map_tensor(plan_samples.to_dict(), lambda x: x[:, 0])
        plan = Plan.from_dict(plan)

        return plan, dict(location_map=preds["location_map"], plan_samples=plan_samples, log_likelihood=preds["log_likelihood"])

