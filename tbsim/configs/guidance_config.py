from tbsim.configs.config import Dict

# Default parameters for optimization and sampling
DEFAULT_PARAMS = {
    "inner_lr": 0.2,
    "scale_grad_by_std": True,
    "inner_beta": 0.5,
    "multiple_guidance_strategy": "weight_guide",
    "grad_wrt": "clean_guide",
    ## diffusion related params
    "sampling_mode": "ddpm",
    "num_samples": 20,
    "sample_step": 1,
    
    "n_guide_steps": 1,
    "partial_t": None,
    # partial diffusion related params
    "desired_delta_s": 0,
    "normal_offset": 0,
    "ref_idx": 0,

}

# Default combined loss configuration
DEFAULT_COMBINE_LOSS = {
    "filter_criterion": "individual",
    "filter_target": "collision",
    "weights": [0.5, 0.5, 0.0, 0.0],  # Default weights for different losses
    "filter_weights": [0.05, 1.0],
    "ctrl_filter_criterion": "individual",
    "ctrl_filter_target": "causecollision",
    "ctrl_filter_weights": [0.0, 0.5, 0.0, 1.0],
    "ctrl_weights": [0.5, 0.5, 0.0, 0.5]
}

# Default configurations for each guidance type
DEFAULT_GUIDANCE_CONFIGS = {
    "collision": {
        "radius": 2,
        "mode": "gaussian",
        "sigma": 1,
        "heading_weight":1,
        "buff_dist": 1.5,
        "prediction_mode": "multi_agent",
        "loss_timesteps": 20,
        "filter_timesteps": 20,
        "adv_mode": True, #whether to ignore adv non-collision loss with non-adv cars
        "loss_scale": 10.0
    },
    "route": {
        "lane_margin": 1.0,
        "nonlinear_factor": 5.0,
        "loss_timesteps": None,
        "filter_timesteps": None,
        "loss_scale": 1.0
    },
    "speed": {
        "desired_speed": 20,
        "loss_timesteps": None,
        "filter_timesteps": None,
        "mode": "hard"
    },
    "ttc": {
        "distance_bandwidth": 1.0,
        "time_bandwidth": 1.0,
        "min_velocity_diff": 0.1,
        "loss_timesteps": None,
        "filter_timesteps": None,
        "mode": "all"
    },
    "trajalign": {
        "loss_timesteps": 32,
        "filter_timesteps": 32
    },
    "causecollision": {
        "prediction_mode": "multi_agent",
        "loss_timesteps": 20,
        "filter_timesteps": 20,
        "adv_term_weight":  {
                "distance": 1.0,
                "speed_penalty": 0.0,
                "filtered_distance": 0.1,
        },
        "interact_mode": "distance",
        "adv_bound": 30,
        "speed_diff": 2.0,
        "interact_dist_thresh": 100.0,
    },
    "drivearea": {
        "loss_timesteps": None,
        "filter_timesteps": None
    },
}
# guidance_config.py
from copy import deepcopy

class GuidanceConfig:
    def __init__(self):
        """Initialize guidance configuration with default parameters"""
        self.configs = deepcopy(DEFAULT_GUIDANCE_CONFIGS)
        self.params = deepcopy(DEFAULT_PARAMS)
        self.combineloss_config = deepcopy(DEFAULT_COMBINE_LOSS)
        self.guidance_fn = []  # List of active guidance functions

    def update_params(self, params_dict):
        """Update optimization parameters"""
        self.params.update(params_dict)

    def update_combine_loss(self, combine_loss_dict):
        """Update combined loss configuration"""
        self.combineloss_config.update(combine_loss_dict)

    def set_guidance_fn(self, guidance_fn):
        """Set the active guidance functions"""
        for name in guidance_fn:
            if name not in self.configs:
                raise ValueError(f"Unknown guidance type: {name}")
        self.guidance_fn = guidance_fn

    def update_config(self, guidance_type, config_dict):
        """Update configuration for a specific guidance type"""
        if guidance_type not in self.configs:
            raise ValueError(f"Unknown guidance type: {guidance_type}")
        self.configs[guidance_type].update(config_dict)

    def update_configs(self, config_dict):
        """Update multiple guidance configurations at once"""
        for guidance_type, config in config_dict.items():
            self.update_config(guidance_type, config)

    def get_config(self, guidance_type):
        """Get configuration for a specific guidance type"""
        if guidance_type not in self.configs:
            raise ValueError(f"Unknown guidance type: {guidance_type}")
        return deepcopy(self.configs[guidance_type])

    def get_all_configs(self):
        """Get all guidance configurations"""
        return deepcopy(self.configs)

    def to_dict(self):
        """Convert configuration to dictionary format"""
        return {
            "guidance_fn": self.guidance_fn,
            "params": deepcopy(self.params),
            "combineloss_config": deepcopy(self.combineloss_config),
            "guidance_configs": {
                name: deepcopy(self.configs[name])
                for name in self.guidance_fn
            }
        }