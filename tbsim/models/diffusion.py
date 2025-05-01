import numpy as np
import torch
from torch.nn import Module, ModuleList, Parameter

from tbsim.models.common import *
from tbsim.utils.geometry_utils import batch_nd_transform_points
from tbsim.utils.guidance_utils import CombinedLossCalculator, Guidance
from tbsim.utils.tensor_utils import compare_tensors_ignore_nan
from tbsim.utils.trajdata_utils import enlarge_batch_samples


class VarianceSchedule(Module):

    def __init__(self, num_steps, mode="cosine", beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ("linear", "cosine")
        print(f"diffusion mode: {mode}")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "cosine":
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffusionTraj(Module):

    def __init__(self, net, var_sched: VarianceSchedule, predict_epsilon=False):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.predict_epsilon = predict_epsilon

    def sample(
        self,
        context,
        num_points,
        bestof,
        num_samples,
        guide_sample_fn=None,
        data_batch_for_guidance=None,
        forward_mode="train",
        current_states=None,
        point_dim=2,
        flexibility=0.0,
        ret_traj=False,
        sampling_mode="ddpm",
        sample_step=1,
        guide_config=None,
        adv_proposals_dict=None,
    ):
        """
        Args:
        context: Traffic context conditioning
        num_points: the number of points in the trajectory
        num_samples: the number of parallel samples for diffusion models
        starting_noisy_a_t: the starting noisy trajectory
        initial steps: the current diffusion steps of the starting_noisy_a_t for the truncated diffusion
        adv_indices: which noisy trajectory to replace
        """
        batch_size = context.size(0)
        ## ENLARGE context and states (batch,dim)->(batch*sample,dim)
        context = enlarge_batch_samples(context, batch_size, num_samples)
        if current_states is not None:
            current_states = enlarge_batch_samples(current_states, batch_size, num_samples)  # [:batch_size] are the same
            assert torch.all(torch.eq(current_states[: num_samples - 1, :], current_states[1:num_samples, :]))
        assert compare_tensors_ignore_nan(context[: num_samples - 1, :], context[1:num_samples, :])
        if bestof:
            a_T = torch.randn([batch_size * num_samples, num_points, point_dim]).to(context.device)
        else:
            a_T = torch.zeros([batch_size * num_samples, num_points, point_dim]).to(context.device)
        # 1) Prepare partial diffusion config if needed
        partial_diffusion_cfg = self._setup_partial_diffusion(guide_config, adv_proposals_dict, num_samples)
        
        # Main diffusion loop
        a_t = a_T
        stride = sample_step
        denoised_traj_list = []
        for t in range(self.var_sched.num_steps, 0, -stride):
            z = torch.randn_like(a_T) if t > 1 else torch.zeros_like(a_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t - stride]
            beta_val = self.var_sched.betas[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            # replace the adv a_t if passes the diffusion steps
            if partial_diffusion_cfg["enabled"] and t == partial_diffusion_cfg["partial_step"]:
                a_t = self._replace_with_partial_diffusion(
                    a_t, 
                    batch_size, 
                    num_samples, 
                    num_points, 
                    point_dim,
                    partial_diffusion_cfg["adv_proposals"],
                    partial_diffusion_cfg["adv_idx"]
                )
            c0 = torch.sqrt(alpha_bar_next) * beta_val / (1 - alpha_bar)
            c1 = torch.sqrt(alpha) * (1 - alpha_bar_next) / (1 - alpha_bar)
            with torch.enable_grad() if guide_sample_fn is not None else torch.no_grad():
                if guide_sample_fn is not None:
                    a_t.requires_grad_(True)
                beta = self.var_sched.betas[[t] * batch_size * num_samples]  # beta means time here?

                ## forward dyanmics
                if self.net.dyn is not None:
                    traj_t = self.net.dyn.forward_dynamics(current_states, a_t, self.net.step_time)[0]
                    tau_t = torch.cat([a_t, traj_t], dim=-1)
                else:
                    tau_t = a_t
                if self.predict_epsilon:
                    raise NotImplementedError
                else:
                    a0_predict = self.net(
                        tau_t, context, time=beta, current_states=current_states, output_type="control"
                    )
                    if guide_sample_fn is not None and guide_config.params.grad_wrt == "clean_guide":
                        a0_predict = self.n_step_guided_p_sample(
                            a_t,
                            a0_predict,
                            guide=guide_sample_fn,
                            beta=beta_val,
                            current_states=current_states,
                            data_batch_for_guidance=data_batch_for_guidance,
                            inner_lr = guide_config.params.inner_lr,
                            inner_beta = guide_config.params.inner_beta,
                            guidance_horizon = guide_config.params.guidance_horizon if guide_config.params.guidance_horizon is not None else 32,
                            n_guide_steps = guide_config.params.n_guide_steps,
                            scale_grad_by_std = guide_config.params.scale_grad_by_std,
                            multiple_guidance_strategy = guide_config.params.multiple_guidance_strategy,
                            grad_wrt = guide_config.params.grad_wrt,
                        )
                       
                    if sampling_mode == "ddpm":
                        a_next = c0 * a0_predict + c1 * a_t + sigma * z
                        if guide_sample_fn is not None and guide_config.params.grad_wrt == "noisy_guide":
                            a_next = self.n_step_guided_p_sample(
                                a_t,
                                a_next,
                                guide=guide_sample_fn,
                                beta=beta_val,
                                current_states=current_states,
                                **guide_config.params.to_dict(),
                            )

                    elif sampling_mode == "ddim":
                        e_theta = (a_t - alpha_bar.sqrt() * a0_predict) / (1 - alpha_bar).sqrt()
                        a_next = alpha_bar_next.sqrt() * a0_predict + (1 - alpha_bar_next).sqrt() * e_theta
                    else:
                        raise NotImplementedError
                a_t = a_next.detach()  # Stop gradient and save trajectory
                if ret_traj:
                    denoised_traj_list.append(a_next.detach())
                else:
                    pass
        if forward_mode == "train":
            return a_t

        if guide_sample_fn is None:
            return a_next.view(batch_size, -1, num_points, point_dim)

        ## do filtration after guidance
        if guide_sample_fn is not None:
            if self.net.dyn is not None:
                traj_final = self.net.dyn.forward_dynamics(current_states, a_t, self.net.step_time)[0]
            else:
                raise NotImplementedError

            a_traj_best_sample = guide_sample_fn.filter(a_t, traj_final, data_batch_for_guidance)  # input shape(B*sample,...)
            a_traj_samples = a_t.view(batch_size, -1, num_points, point_dim)
            a_traj_samples[:, 0:1] = a_traj_best_sample
            if ret_traj:
                a_denoised = torch.stack(denoised_traj_list, dim=1).view(
                    batch_size, -1, num_points, point_dim
                )  # (B, sample*diffusionstep,T,2)
                # TODO add index to store?

            return {"sampled_trajectories": a_traj_samples, "denoised_trajectories": a_denoised if ret_traj else None}
        return a_t.view(batch_size, -1, num_points, point_dim)  # (B, sample, T,2)

    def n_step_guided_p_sample(
        self,
        a_t,  # The input action tau_k
        action,  # action is the noisy action or the cleaned action
        guide: Guidance,  # The guidance object
        beta,  # Covariance
        current_states=None,  # Current states, if applicable
        data_batch_for_guidance=None,
        scale: float = 0.001,  # Scaling factor
        guidance_horizon: int = 32,  # Threshold for stopping gradient computation
        n_guide_steps: int = 1,  # Number of guidance steps
        scale_grad_by_std: bool = False,  # Whether to scale gradient by standard deviation
        inner_lr: float = 1.0,  # Inner learning rate
        inner_beta: float = 0.5,  # Inner beta value
        multiple_guidance_strategy: str = "weight_guide",  # Strategy for multiple guidance
        grad_wrt: str = "noisy_guide",  # Gradient with respect to
    ):
        for i in range(n_guide_steps):
            action_original = action.clone()  # Keep a copy of original action

            # pass through the dynamics
            if self.net.dyn is not None:
                traj = self.net.dyn.forward_dynamics(current_states, action, self.net.step_time, mode="parallel")[0]
            else:
                raise NotImplementedError
            grad = guide.calculate_grad(action, traj, a_t, grad_wrt, data_batch_for_guidance=data_batch_for_guidance)  # detach action from its computation history

            if scale_grad_by_std:
                grad = beta * grad

            # Update action with gradient and clip changes
            action = action_original + torch.clamp(
                inner_lr * grad,
                -inner_beta,
                inner_beta
            )

        return action

   
    def apply_conditioning(x, conditions, action_dim):
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
        return x

    # -------------------------------------Partial Diffusion Related code ---------------------------------
    def add_diffusion_noise(self, a_0, t=10, sample_size=20):
        # expand the shape to sample_size
        num_adv, num_proposals, T, _ = a_0.shape
        assert sample_size % num_proposals == 0
        scale_factor = sample_size // num_proposals

        # reshape a_0 to (num_adv, num_proposals, T, 2) -> (num_adv, num_proposals,scale_sample T, 2)
        a_0 = a_0.unsqueeze(2).repeat(1, 1, scale_factor, 1, 1).reshape(num_adv, num_proposals * scale_factor, T, 2)
        # create t of shape (num_adv*num_proposals*scale_sample)
        t = [t] * num_adv

        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1, 1).cuda()  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1).cuda()  # (B, 1, 1)

        e_rand = torch.randn_like(a_0).cuda()  # (B, N, d)

        ## add perturb noise to action
        a_pert = c0 * a_0 + c1 * e_rand

        return a_pert  # (num_adv, sample, T, 2)
    def handle_partial_diffusion(
        a_t: torch.Tensor,
        batch_size: int,
        sample: int,
        num_points: int,
        point_dim: int,
        current_step: int,
        partial_cfg: dict,
    ) -> torch.Tensor:
        """
        Replaces noisy trajectories with the partially diffused ones
        if `current_step` matches `partial_cfg["partial_step"]`.
        Parameters:
        - a_t (Tensor): Current noisy trajectory of shape (B*sample, T, 2)
        - batch_size (int): Original batch size
        - sample (int): Number of sampled trajectories
        - num_points (int): Horizon length (T)
        - point_dim (int): Dimensionality of each point (2D or more)
        - current_step (int): Current iteration in the denoising loop
        - partial_cfg (dict): Config with keys:
            * "enabled" (bool): If partial diffusion is used
            * "partial_step" (int): Step index at which partial diffusion is applied
            * "adv_proposals" (Tensor): Proposed partial trajectories (B_adv, ...)
            * "adv_idx" (Tensor): Indices of batch elements that need partial replacement

        Returns:
        - Tensor: Possibly updated `a_t`.
        """
        if partial_cfg is None or not partial_cfg["enabled"]:
            return a_t  # No partial diffusion

        if current_step == partial_cfg["partial_step"]:
            # Perform the partial diffusion replacement
            a_t = a_t.view(batch_size, sample, num_points, point_dim)
            a_t[partial_cfg["adv_idx"]] = partial_cfg["adv_proposals"]
            a_t = a_t.view(batch_size * sample, num_points, point_dim)
        return a_t
    
    @staticmethod
    def apply_fixed_actions(action, fixed_actions, indices):
        action[:, indices] = fixed_actions[:, indices]
        return action

    def _setup_partial_diffusion(self, guide_config, adv_proposals_dict, sample):
        """Setup configuration for partial diffusion if enabled."""
        if guide_config.params.partial_t is not None and adv_proposals_dict is not None:
            return {
                "enabled": True,
                "partial_step": guide_config.params.partial_t,
                "adv_proposals": self.add_diffusion_noise(
                    adv_proposals_dict["adv_proposals"], 
                    t=guide_config.params.partial_t, 
                    sample_size=sample
                ),
                "adv_idx": adv_proposals_dict["adv_idx"],
            }
        return {"enabled": False}

    def _replace_with_partial_diffusion(self, a_t, batch_size, sample, num_points, point_dim, adv_proposals, adv_idx):
        """Replace specified trajectories with partially diffused ones."""
        a_t = a_t.view(batch_size, sample, num_points, point_dim)
        a_t[adv_idx] = adv_proposals
        return a_t.view(batch_size * sample, -1, point_dim)


import matplotlib.pyplot as plt

def plot_variance_schedule(var_schedule):
    # Assuming var_schedule is an instance of VarianceSchedule with the 'cosine' mode
    num_steps = var_schedule.num_steps
    t = torch.arange(0, num_steps + 1).float()  # Assuming t starts from 1 to num_steps

    alpha_bar = var_schedule.alpha_bars[t.long()]
    alpha = var_schedule.alphas[t.long()]
    beta_val = var_schedule.betas[t.long()]
    sigma = var_schedule.sigmas_flex[t.long()]

    alpha_bar_next = var_schedule.alpha_bars[(t - 1).long()]
    c0 = torch.sqrt(alpha_bar_next) * beta_val / (1 - alpha_bar)
    c1 = torch.sqrt(alpha) * (1 - alpha_bar_next) / (1 - alpha_bar)

    c0 = torch.sqrt(alpha_bar)  # (B, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar)  # (B, 1, 1)
    plt.figure(figsize=(12, 8))

    # Plotting c0 and c1
    plt.plot(t, c0.cpu(), label="c0", color="red")
    plt.plot(t, c1.cpu(), label="c1", color="blue")

    # Plotting sigmas, alphas, and alpha_bars for reference
    plt.plot(t, sigma.cpu(), label="Sigmas", color="black")
    plt.plot(t, alpha_bar.cpu(), label="Alpha bars", color="green", linestyle="--")
    plt.plot(t, alpha.cpu(), label="Alphas", color="purple", linestyle="--")

    plt.plot(t, beta_val.cpu(), label="Beta", color="orange", linestyle="--")
    plt.title("Variance Schedule and Parameters")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("Schedules")


if __name__ == "__main__":
    var_schedule = VarianceSchedule(100, beta_T=5e-2, mode="cosine")
    plot_variance_schedule(var_schedule)
