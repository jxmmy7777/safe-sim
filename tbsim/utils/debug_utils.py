import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from tbsim.utils.batch_utils import batch_utils

from torch.utils.data import DataLoader
from trajdata.visualization.vis import plot_agent_batch
from tqdm import tqdm
from trajdata import AgentBatch, AgentType, UnifiedDataset


class ScatterPlotCallback(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        #obtain batch size!
        # loader = iter(trainer.train_dataloader)
        batch_size = 10
        
        # batch = next(iter(trainer.datamodule.val_dataloader()))
        dataloader_dict = DataLoader(
            trainer.datamodule.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn= trainer.datamodule.valid_dataset.get_collate_fn(return_dict=True),
            num_workers=0,
        )
        iter_batch_loader  = iter(dataloader_dict)

        dataloader = DataLoader(
            trainer.datamodule.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn= trainer.datamodule.valid_dataset.get_collate_fn(),
            num_workers=0,
        )
        
        iter_plot_loader = iter(dataloader)
        for iter_num, (batch, plot_batch) in enumerate(zip(iter_batch_loader, iter_plot_loader),start=0):

            # batch = next(iter_plot_loader)
            # plot_batch: AgentBatch = next(iter_batch_loader)
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            batch = batch_utils().parse_batch(batch)    
            sample = pl_module.nets["policy"].sample(batch)
            if sample["trajectories"].shape[1]>=100:
                sample["trajectories"] = sample["trajectories"].reshape(batch_size, -1,100,32,3).detach().cpu().numpy()
            for i in range(0,batch_size):
                fig, (ax,ax2)= plt.subplots(2, 1, figsize=(6, 9),gridspec_kw={'height_ratios': [1, 1]})
                ax = plot_agent_batch(plot_batch, batch_idx=i,ax=ax)
                # fig.savefig("test_plpt" + str(trainer.current_epoch)+str(i)+str(iter_num)+".png")
                ax.scatter(batch["target_positions"][i,:,0].detach().cpu(), batch["target_positions"][i,:,1].detach().cpu(), label="gt", s= 2)
                ax2.scatter(batch["target_positions"][i,:,0].detach().cpu(), batch["target_positions"][i,:,1].detach().cpu(), label="gt",s= 0.6, zorder= 2)
                # ax.scatter(batch["history_positions"][i,:,0].detach().cpu(), batch["history_positions"][i,:,1].detach().cpu(), label="hist gt")
        
                if len(sample["trajectories"].shape) == 5:  # Assuming 5D tensor with diffusion steps
                    for step in range(0, sample["trajectories"].shape[2], 9):  # Iterate over diffusion steps
                        fig, ax = plt.subplots()
                        # Re-draw the initial plot for each step and sample
                        ax = plot_agent_batch(plot_batch, batch_idx=i, show=False, close=False, ax=ax)
                        for j in range(sample["trajectories"].shape[1]):  # Iterate over samples
                            # Plot for the current sample and step
                            ax.plot(sample["trajectories"][i, j, step, :, 0], sample["trajectories"][i, j, step, :, 1], label=f"Sample {i}, Step {step}",alpha=0.2,color="blue")
                            # ax.legend()
                            # plt.close(fig)
                        plt.savefig(f"{trainer.default_root_dir}/test_epo_{trainer.current_epoch}__batch_{iter_num}_batchidx{i}_Step_{step}_sample.png")
                        plt.close(fig)
                
                else:
                    for j in range(sample["trajectories"].shape[1]):
                        ax.plot(sample["trajectories"][i,j,:,0].detach().cpu(), sample["trajectories"][i,j,:,1].detach().cpu(),alpha = 0.2)
                        ax2.plot(sample["trajectories"][i,j,:,0].detach().cpu(), sample["trajectories"][i,j,:,1].detach().cpu(),alpha = 0.2)
                        ax2.scatter(sample["trajectories"][i,j,-1,0].detach().cpu(), sample["trajectories"][i,j,-1,1].detach().cpu(),s = 2)
                    # break
                # ax.axis('equal')
                ax.legend()
                ax2.set_ylim([-3,3])
                # Save the figure
                fig.savefig(f"{trainer.default_root_dir}/test_epo_{trainer.current_epoch}__batch_{iter_num}_batchidx{i}.png")
                plt.close(fig)
            break

            
            # Plot the scatter plot
            # for i in range (batch["target_positions"].shape[0]):
            #     plt.scatter(batch["target_positions"][i,:,0].detach().cpu(), batch["target_positions"][i,:,1].detach().cpu(), label="gt")
            #     plt.scatter(batch["history_positions"][i,:,0].detach().cpu(), batch["history_positions"][i,:,1].detach().cpu(), label="hist gt")
            #     # for j in range(sample["trajectories"].shape[1]):
            #     #     plt.scatter(sample["trajectories"][i,j,:,0].detach().cpu(), sample["trajectories"][i,j,:,1].detach().cpu(), label="predict")
            #     plt.axis('equal')
            #     plt.xlim(-40, 60)
            #     plt.ylim(-50, 50)
                
            #     # Save the figure
            #     plt.savefig("test" + str(trainer.current_epoch)+str(i)+".png")
            #     plt.close()
def inverse_dyn(x,xp,dt):
        if len(dt.shape) ==1:
            return (xp[...,2:]-x[...,2:])/dt[:,None,None]
        return (xp[...,2:]-x[...,2:])/dt
class GetDataStatistics(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        if True:
            # batch = next(iter(trainer.datamodule.val_dataloader()))
            dataloader = DataLoader(
                trainer.datamodule.train_dataset,
                batch_size=120*8,
                shuffle=False,
                collate_fn= trainer.datamodule.train_dataset.get_collate_fn(return_dict=True),
                num_workers=0,
            )
    
            iter_plot_loader = iter(dataloader)
            total_stats = []
            for batch in tqdm(iter_plot_loader):
                # batch = next(iter_plot_loader)
                # plot_batch: AgentBatch = next(iter_batch_loader)
                batch = {key: value.to(pl_module.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                batch = batch_utils().parse_batch(batch)
                # sample = pl_module.nets["policy"].sample(batch)
                target_states  = torch.cat([batch["target_positions"],
                                        batch["target_speeds"][...,None],batch["target_yaws"]],dim=-1) 
                target_actions = inverse_dyn(target_states[...,:-1,:],target_states[...,1:,:],batch["dt"])
                tau_targets = torch.cat([target_actions,target_states[:,1:]],dim=-1)
                total_stats.append(tau_targets)
                # plot_agent_batch(batch, batch_idx=0)
            total_stats = torch.cat(total_stats,dim=0)
            mean = torch.mean(total_stats,dim=(0,1))
            std  = torch.std(total_stats,dim=(0,1))
            print("mean",mean)
            print("std",std)
    