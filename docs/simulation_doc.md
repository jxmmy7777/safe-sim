This document provides a comprehensive list of simulation arguments for `run_adv_simulation.py`.

## **1. General Arguments**

| Argument | Description |
| --- | --- |
| `--results_root_dir` | Directory to store simulation results |
| `--num_scenes_per_batch` | Number of scenes processed per batch |
| `--dataset_path` | Path to the NuScenes dataset |
| `--env` | Specifies the dataset environment (e.g., `nusc`) |
| `--eval_class` | The planner to evaluate (`StrivePolicy_trajdata`, `HierAgentAware`,`IDMPolicy` etc.) |
| `--agent_eval_class` | The reactive agent  model (e.g., `Diffusion`) |
| `--ckpt_yaml` | Path to model checkpoint configuration file |
| `--scene_select_mode` | subsets for evaluation, e.g. `ttc` ,`partial_diffusion` , `human` |
| `—sim_steps` | Total simulation length |

Note that `StrivePolicy_trajdata` is the reimplementation borrowd from [strive](https://github.com/nv-tlabs/STRIVE), where it is a lane-graph based rule-based planner

### **Guidance Options**

| Argument | Description |
| --- | --- |
| `--guidance` | Enables adversarial guidance |
| `--guidance_fn` | Guidance function to apply (e.g., `route`, `collision`, `causecollision`, `ttc`) |
| `--guidance_params` | Hyperparameters for the selected guidance function |

### Planner Description

| Planner | Description |
| --- | --- |
| `StrivePolicy_trajdata` | A reimplementation borrowed from STRIVE, a **lane-graph-based rule-based planner** |
| `HierAgentAware` | A hybrid planning policy that incorporates agent awareness from [tbsim](https://github.com/NVlabs/traffic-behavior-simulation). Should download BITS model from [tbsim](https://github.com/NVlabs/traffic-behavior-simulation), and update `evaluation/BITS.yaml` |
| `IDMPolicy` | A rule-based **Intelligent Driver Model (IDM)** planner |


### guidance weight

`ctrl_weights`: The parameter in `guidance_params` is used to **control the strength of adversarial guidance of adv-agents**. 

`weights`: Controls the guidance of non-adv agents.

For a detailed breakdown of guidance functions and parameters, refer to [**`configs/guidance_config.py`**](https://www.notion.so/configs/guidance_config.py).
## **2. Scenario-Specific Simulations**

This section provides commands for specific simulation settings, such as **Partial Diffusion**, **Time-To-Collision (TTC)**

### **2.1 Running Partial Diffusion**

Partial Diffusion allows the model to start from trajectory proposals instead of sampling  from pure gaussian noise

### **Command Template**

```bash
python scripts/run_adv_simulation.py \
    --results_root_dir=/path/to/results \
    --num_scenes_per_batch=10 \
    --dataset_path=/path/to/nuscenes \
    --env=nusc \
    --eval_class=StrivePolicy_trajdata \
    --agent_eval_class=Diffusion \
    --ckpt_yaml=evaluation/Diffusion.yaml \
    --guidance \
    --guidance_fn=route_collision_trajalign_causecollision \
    --guidance_params="params,partial_t,10;combine_loss,ctrl_weights,[0.5,0.5,0.25,0.5];params,ref_idx,2" \
    --sim-steps=100 \
    --num_scenes_to_evaluate=5 \
    --scene_select_mode=partial_diffusion \
    --render
```

### **Key Parameters for Partial Diffusion**

| Parameter | Description |
| --- | --- |
| `params,partial_t,10` | **Controls how much noise is added to trajectory proposals**. Higher values increase randomness. |
| `params,ref_idx,2` | **Determines which reference centerline to generate proposals from**. Different indices correspond to different centerlines. |
| `combine_loss,ctrl_weights,[0.5,0.5,0.25,0.5]` | **Adversarial guidance weights (explained above)** |

### **2.2 Running TTC**

Time-To-Collision (TTC) controls the aggressiveness of adversarial agents based on collision risk. Control the <ttc_weight> to change the aggressiveness of the adversarial agents.

```markdown
python scripts/run_adv_simulation.py \
    --results_root_dir=/path/to/results \
    --num_scenes_per_batch=5 \
    --dataset_path=/path/to/nuscenes \
    --env=nusc \
    --eval_class=StrivePolicy_trajdata \
    --agent_eval_class=Diffusion \
    --ckpt_yaml=evaluation/Diffusion.yaml \
    --guidance \
    --guidance_fn=route_collision_ttc_causecollision \
    --prefix=ttc_experiment \
    --guidance_params="combine_loss,ctrl_weights,[0.5,0.5,<ttc_weight>,0.5];causecollision,adv_bound,200" \
    --seed=0 \
    --scene_select_mode=ttc \
    --sim-steps=100 \
    --skip_first_n=0 \
    --render
```

### **2.3 Running non-adv simulation**

```python
python scripts/run_adv_simulation.py \
  --results_root_dir=path/to/results \
  --num_scenes_per_batch=1 \
  --dataset_path=/path/to/nuscenes \
  --env=nusc \
  --eval_class=StrivePolicy_trajdata \
  --agent_eval_class=Diffusion \
  --ckpt_yaml=evaluation/Diffusion.yaml \
  --guidance \
  --guidance_fn=route_collision \
  --prefix=0215_no_collision_mask_stationary \
  --guidance_params="combine_loss,weights,[0.5,0.5]" \
  --scene_select_mode=no_collision \
  --sim-steps=100 \
  --num_scenes_to_evaluate=60 \
  --skip_first_n=0
```