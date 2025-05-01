import json
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.base import ExperimentConfig
from tbsim.configs.config import Dict



def get_experiment_config_from_file(file_path, locked=False):
    ext_cfg = Dict(json.load(open(file_path, "r")))
    cfg = Dict(get_registered_experiment_config(ext_cfg["registered_name"]))

    cfg.update(**ext_cfg)
    ''' modify here for updating configs for models!!!'''
    for key in list(cfg["algo"].keys()):  # Create a copy of keys
        if key not in ext_cfg["algo"] and key != 'guide_config':
            cfg.algo[key] = None if not isinstance(cfg.algo[key], bool) else False

    if ext_cfg["registered_name"]=="nusc_bc" and "prediction_length" not in ext_cfg["algo"]:
        cfg.algo["prediction_length"] = ext_cfg["algo"]["future_num_frames"]
        cfg.algo["state_dim"] = 3
    cfg.lock(locked)
    return cfg


def translate_trajdata_cfg(cfg: ExperimentConfig):
    rcfg = Dict()
    # assert cfg.algo.step_time == 0.5  # TODO: support interpolation
    if "scene_centric" in cfg.algo and cfg.algo.scene_centric:
        rcfg.centric="scene"
    else:
        rcfg.centric="agent"
    if "standardize_data" in cfg.env.data_generation_params:
        rcfg.standardize_data = cfg.env.data_generation_params.standardize_data
    else:
        rcfg.standardize_data = True
    rcfg.step_time = cfg.algo.step_time
    rcfg.trajdata_source_root = cfg.train.trajdata_source_root
    rcfg.trajdata_source_train = cfg.train.trajdata_source_train
    rcfg.trajdata_source_valid = cfg.train.trajdata_source_valid
    rcfg.load_cache = cfg.train.load_cache
    rcfg.data_filter = cfg.train.data_filter
    rcfg.dataset_path = cfg.train.dataset_path
    rcfg.history_num_frames = cfg.algo.history_num_frames
    rcfg.future_num_frames = cfg.algo.future_num_frames
    rcfg.max_agents_distance = cfg.env.data_generation_params.max_agents_distance
    rcfg.num_other_agents = cfg.env.data_generation_params.other_agents_num
    rcfg.max_agents_distance_simulation = cfg.env.simulation.distance_th_close
    rcfg.pixel_size = cfg.env.rasterizer.pixel_size
    rcfg.raster_size = int(cfg.env.rasterizer.raster_size)
    rcfg.raster_center = cfg.env.rasterizer.ego_center
    rcfg.yaw_correction_speed = cfg.env.data_generation_params.yaw_correction_speed
    rcfg.incl_neighbor_map = cfg.env.incl_neighbor_map
    rcfg.other_agents_num = cfg.env.data_generation_params.other_agents_num
    if "vectorize_lane" in cfg.env.data_generation_params:
        rcfg.vectorize_lane = cfg.env.data_generation_params.vectorize_lane
    else:
        rcfg.vectorize_lane = "None"
        
    rcfg.lock()
    return rcfg
def update_config(attr, config):
    for key, value in config.items():
        if isinstance(value, dict):
            # Handle nested dictionary
            if hasattr(attr, key):
                nested_attr = getattr(attr, key)
                update_config(nested_attr, value)
            else:
                raise KeyError(f"Key {key} not found")
        else:
            if hasattr(attr, key):
                setattr(attr, key, value)
            else:
                raise KeyError(f"Key {key} not found")