# Code Structure Overview

Since the codebase is quite large, this section provides a **high-level overview** of the main directories and their functionalities.

- `evaluation/` contains the config files for TBSim:
    - `evaluation/Diffusion.yaml` → Defines checkpoint directories for running simulations.

- `tbsim/` contains the core components of TBSim:
    - `configs/`
        - `algo_configs.py` → Defines diffusion model configurations.
        - `eval_config.py` → Configurations for simulation evaluation.
            - **Important parameters**:  
              - `num_scenes_to_evaluate`
              - `n_step_action`
              - `num_simulation_steps`
        - `guidance_config.py` → Defines hyperparameters for each guidance method.
        - `registry.py` → Registers available algorithms.
    - `utils/`
        - `guidance_utils.py` → Defines guidance objectives.
        - `env_utils.py` → `rollout_episodes` is the main function for closed-loop simulation.
    
    - `policies/` serves as a wrapper for different models for closed-loop simulation:
        - `wrapper.py` → `RolloutWrapper` controlling both the ego and agent policy.
    
    - `models/` contains core machine learning models:
        - `RasterizedDiffusionModel.py` → The main diffusion model framework.
        - `Temporal.py` → Implements the **U-Net** neural network structure (adapted from [Diffuser](https://github.com/jannerm/diffuser)).
        - `Diffusion.py` → Defines the diffusion process and **guided sampling** procedures.
            - `n_step_guided_p_sample` → Function for adding guidance during sampling.

    - `envs/` contains the simulation environment:
        - `env_metrics.py`
        - `env_trajdata.py` → `EnvUnifiedSimulation` is the interface for closed-loop simulation (calls **simulation from trajdata**).
    
    - `evaluation/`
        - `policy_composers.py` → Transforms trained models/rule-based policies into policies for simulation.
    - `policies/`
        - `hardcoded.py` → Implementation for evluated planners like IDM, StrivePolicy, etc.
        - `strive_planner_trajdata.py` → Implementation for lane-graph rule-based planner from [STRIVE](https://github.com/nv-tlabs/STRIVE).
---
