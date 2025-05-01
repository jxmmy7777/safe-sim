import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
from tbsim.models.base_models import TrajectoryDecoder
from .diffuser_helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, context_dim=384, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels//2),
            Rearrange('batch t -> batch t 1'),
        )

        self.context_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(context_dim+embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )


        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, context, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.context_mlp(torch.cat([t,(context)],dim = 1))
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(TrajectoryDecoder):

    def __init__(
        self,
        horizon,
        transition_dim,
        context_dim,
        dim=32,
        dim_mults=(1, 2, 4),
        attention=False,
        dynamics_config=None,
        scale_net_output = False
    ):
        super(TrajectoryDecoder,self).__init__()
        # self.feature_dim = dynamics_config["feature_dim"]
        self.scale_net_output = scale_net_output
        self.state_dim = dynamics_config["state_dim"]
        self.num_steps = dynamics_config["num_steps"]
        self.step_time = dynamics_config["step_time"]
        self._dynamics_type = dynamics_config["dynamics_type"]
        self._dynamics_kwargs = dynamics_config["dynamics_kwargs"]
        self._create_dynamics()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, context_dim=context_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, context_dim=context_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, context_dim=context_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, context_dim=context_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, context_dim=context_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, context_dim=context_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, 2, 1),
        )
    

    def _forward_networks(self, x, context, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, context, t)
            x = resnet2(x, context, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, context, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, context, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, context, t)
            x = resnet2(x, context, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
    
    def forward(self, x, context,time, current_states=None, num_steps=None,output_type="traj", **kwargs):
        pred_trajs = self._forward_networks(x,context,time)
        if self.scale_net_output:
            # raise ValueError("actions are in different scale, this should cause problems for denoising!")
            # actions are in different scale, this should cause problems for denoising!
            pred_trajs[...,1] = pred_trajs[...,1] * 0.0552
        if output_type == "control":
            return pred_trajs
        else:
            preds = {"trajectories":pred_trajs}
            if self.dyn is not None:
                preds["controls"] = preds["trajectories"]
                preds["trajectories"], x = self._forward_dynamics(
                    current_states=current_states,
                    actions=preds["trajectories"],
                    **kwargs
                )
                preds["terminal_state"] = x[...,-1,:]
            
            if output_type == "dict":
                return preds
            elif output_type == "traj":
                return preds["trajectories"][:,:,:2]
            else:
                raise NotImplementedError
