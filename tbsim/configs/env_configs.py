
##--------------- For scene initializations ------------------
ENV_CONFIG_MODULES = {
    "nusc": "tbsim.configs.selected_scene_config",  # This is the module that has e.g. TRAIN_SCENE_IDX_MAP
    "nuplan": "tbsim.configs.selected_scene_config_nuplan_mini_val",
    # You can add "drivesim": "some_other_module",
    # or "l5kit": "some_l5_module",
    # etc.
}
import importlib
import numpy as np
import json
from tbsim.utils.agent_rel_classify import load_predefined_scene_init_from_json

def import_env_configs(env_name):
    """
    Given an environment name (e.g. 'nusc'), look up the corresponding
    module in ENV_CONFIG_MODULES, then dynamically import it.
    
    Expects that module to define the following names:
      TRAIN_SCENE_IDX_MAP, SCENE_IDX_MAP, SCENE_NAMES, 
      PREDEFINED_SCENE_INIT, PREDEFINED_SCENE_ALL_INIT

    Returns them as a tuple.
    """
    mod_path = ENV_CONFIG_MODULES.get(env_name, None)
    if mod_path is None:
        raise ValueError(f"No config module found for env '{env_name}'!")
    
    # import the chosen module
    mod = importlib.import_module(mod_path)
    
    # now extract each item
    TRAIN_SCENE_IDX_MAP = getattr(mod, "TRAIN_SCENE_IDX_MAP", {})
    SCENE_IDX_MAP = getattr(mod, "SCENE_IDX_MAP", {})
    SCENE_NAMES = getattr(mod, "SCENE_NAMES", [])
    PREDEFINED_SCENE_INIT = getattr(mod, "PREDEFINED_SCENE_INIT", {})
    PREDEFINED_SCENE_ALL_INIT = getattr(mod, "PREDEFINED_SCENE_ALL_INIT", {})
    
    return (
        TRAIN_SCENE_IDX_MAP,
        SCENE_IDX_MAP,
        SCENE_NAMES,
        PREDEFINED_SCENE_INIT,
        PREDEFINED_SCENE_ALL_INIT,
    )
def get_eval_scenes_and_predefined_init(
    import_scene_list_mode,
    eval_cfg,
    SCENE_IDX_MAP,
    SCENE_NAMES,
    PREDEFINED_SCENE_INIT,
    PREDEFINED_SCENE_ALL_INIT,
    TRAIN_SCENE_IDX_MAP=None
):
    """
    Helper function that returns the list of scene indices (eval_scenes)
    and the dict for predefined_scene_init, based on the chosen mode.

    :param import_scene_list_mode: str
        e.g. "human", "collision", "intersection", etc.
    :param eval_cfg:
        The global eval configuration object, which might contain e.g. eval_cfg.eval_scenes.
    :param SCENE_IDX_MAP: dict
        Maps scene_name -> index
    :param SCENE_NAMES: list
        List of scene names (for "human" etc.).
    :param PREDEFINED_SCENE_INIT: dict
        Scenes -> initialization (for collisions etc.).
    :param PREDEFINED_SCENE_ALL_INIT: dict
        Scenes -> initialization for "human_all".
    :param TRAIN_SCENE_IDX_MAP: dict or None
        If using train_* modes, where we have a separate mapping.

    :return:
        (eval_scenes, new_predefined_scene_init)
    """
    # Default to what's in the config
    eval_scenes = eval_cfg.eval_scenes
    new_predef_init = {}

    if import_scene_list_mode == "human":
        eval_scenes = [SCENE_IDX_MAP[scene_name] for scene_name in SCENE_NAMES]
        new_predef_init = PREDEFINED_SCENE_INIT
    elif import_scene_list_mode == "no_collision":
        eval_scenes =  np.arange(0, 150)
        new_predef_init = None
    elif import_scene_list_mode == "collision":
        eval_scenes = [SCENE_IDX_MAP[sn] for sn in PREDEFINED_SCENE_INIT.keys()]
        new_predef_init = PREDEFINED_SCENE_INIT
    elif import_scene_list_mode == "ttc":
        # Filter scenes marked for TTC from PREDEFINED_SCENE_INIT for demo purpose
        ttc_scenes = {
            scene_name: scene_data 
            for scene_name, scene_data in PREDEFINED_SCENE_INIT.items()
            if scene_name in ['scene-0910', 'scene-0106', 'scene-0638']  # TTC scenes
        }
        eval_scenes = [SCENE_IDX_MAP[sn] for sn in ttc_scenes.keys()]
        new_predef_init = ttc_scenes
    elif import_scene_list_mode == "partial_diffusion":
        # Filter scenes marked for Partial Diffusion
        pd_scenes = {
            scene_name: scene_data 
            for scene_name, scene_data in PREDEFINED_SCENE_INIT.items()
            if scene_name in ['scene-0096', 'scene-0561']  # Partial Diffusion scenes
        }
        eval_scenes = [SCENE_IDX_MAP[sn] for sn in pd_scenes.keys()]
        new_predef_init = pd_scenes

    elif import_scene_list_mode in ["intersection", "merging", "nearby"]:
        raise NotImplementedError("This mode is not implemented yet")
        relationship = [import_scene_list_mode]
        if import_scene_list_mode == "nearby":
            interaction_list = [3, 4]
        else:
            interaction_list = ["equal", "car2_behind_car1"]

        tmp_init = load_predefined_scene_init_from_json(relationship, interaction_list)
        eval_scenes = []
        for scene_name, scene_data in tmp_init.items():
            count = len(scene_data["indices"])
            eval_scenes.extend([SCENE_IDX_MAP[scene_name]] * count)
        new_predef_init = tmp_init
    else:
        raise ValueError(f"Invalid import_scene_list_mode: {import_scene_list_mode}")

    return eval_scenes, new_predef_init
