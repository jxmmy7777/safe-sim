import abc
from pathlib import Path
from typing import Dict, OrderedDict, Tuple

import torch
from trajdata import MapAPI, VectorMap

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.algos.algo_utils import yaw_from_pos
from tbsim.policies.common import Action, Plan, RolloutAction
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import calc_distance_map
from tbsim.utils.planning_utils import ego_sample_planning


class Wrappers(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        pass

    @abc.abstractmethod
    def unwrap(self):
        pass


class HierarchicalWrapper(Wrappers):
    """A wrapper policy that feeds subgoal from a planner to a controller"""

    def __init__(self, planner, controller):
        self.device = planner.device
        self.planner = planner
        self.controller = controller

    def eval(self):
        self.planner.eval()
        self.controller.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        plan, plan_info = self.planner.get_plan(obs)
        actions, action_info = self.controller.get_action(obs, plan=plan, init_u=plan.controls)
        action_info["plan"] = plan.to_dict()
        plan_info.pop("plan_samples", None)
        action_info["plan_info"] = plan_info
        return actions, action_info

    def unwrap(self):
        controller_unwrap = self.controller.unwrap() if isinstance(self.controller, Wrappers) else self.controller
        planner_unwrap = self.planner.unwrap() if isinstance(self.planner, Wrappers) else self.planner
        return {"Hierarchical.controller": controller_unwrap, "Hierarchical.planner": planner_unwrap}


class HierarchicalSamplerWrapper(HierarchicalWrapper):
    """A wrapper policy that feeds plan samples from a stochastic planner to a controller"""

    def get_action(self, obs, **kwargs) -> Tuple[None, Dict]:
        _, plan_info = self.planner.get_plan(obs)
        plan_samples = plan_info.pop("plan_samples")
        b, n = plan_samples.positions.shape[:2]
        actions_tiled, _ = self.controller.get_action(obs, plan_samples=plan_samples, init_u=plan_samples.controls)

        action_samples = TensorUtils.reshape_dimensions(
            actions_tiled.to_dict(), begin_axis=0, end_axis=1, target_dims=(b, n)
        )
        action_samples = Action.from_dict(action_samples)

        action_info = dict(
            plan_samples=plan_samples,
            action_samples=action_samples,
            plan_info=plan_info,
        )
        if "log_likelihood" in plan_info:
            action_info["log_likelihood"] = plan_info["log_likelihood"]
        return None, action_info


class SamplingPolicyWrapper(Wrappers):
    def __init__(self, ego_action_sampler, agent_traj_predictor):
        """

        Args:
            ego_action_sampler: a policy that generates N action samples
            agent_traj_predictor: a model that predicts the motion of non-ego agents
        """
        self.device = ego_action_sampler.device
        self.sampler = ego_action_sampler
        self.predictor = agent_traj_predictor
        self.vector_maps = dict()
        cache_path = Path("~/.unified_data_cache").expanduser()
        self.mapAPI = MapAPI(cache_path)

    def eval(self):
        self.sampler.eval()
        self.predictor.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # actions of shape [B, num_samples, ...]
        _, action_info = self.sampler.get_action(obs)
        action_samples = action_info["action_samples"]
        agent_preds, _ = self.predictor.get_prediction(obs)  # preds of shape [B, A - 1, ...]

        if isinstance(action_samples, dict):
            action_samples = Action.from_dict(action_samples)

        ego_trajs = action_samples.trajectories
        agent_pred_trajs = agent_preds.trajectories

        agent_extents = obs["all_other_agents_history_extents"][..., :2].max(dim=-2)[0]
        drivable_map = batch_utils().get_drivable_region_map(obs["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        log_likelihood = action_info.get("log_likelihood", None)
        for map_name in obs["map_names"]:
            if map_name not in self.vector_maps:
                self.vector_maps[map_name] = self.mapAPI.get_map(map_name, scene_cache=None)
        vector_map = [self.vector_maps[map_name] for map_name in obs["map_names"]]

        action_idx = ego_sample_planning(
            ego_trajectories=ego_trajs,
            agent_trajectories=agent_pred_trajs,
            ego_extents=obs["extent"][:, :2],
            agent_extents=agent_extents,
            raw_types=obs["all_other_agents_types"],
            raster_from_agent=obs["raster_from_agent"],
            dis_map=dis_map,
            log_likelihood=log_likelihood,
            weights=kwargs["cost_weights"],
            vector_map=vector_map,
            world_from_agent=obs["world_from_agent"],
            yaw=obs["yaw"],
        )

        ego_trajs_best = torch.gather(
            ego_trajs, dim=1, index=action_idx[:, None, None, None].expand(-1, 1, *ego_trajs.shape[2:])
        ).squeeze(
            1
        )  # shape batch,20,3

        # if planner only plan 20 timesteps, we use constant velocity models to padd to 32
        num_padding = 12
        last_position = ego_trajs_best[:, -1, :2]
        second_last_position = ego_trajs_best[:, -2, :2]
        velocity = last_position - second_last_position
        # Create padding positions based on constant velocity
        padding_positions = (
            last_position.unsqueeze(1)
            + velocity.unsqueeze(1)
            * torch.arange(1, num_padding + 1, device=last_position.device).unsqueeze(0).unsqueeze(2)
            / num_padding
        )
        # Repeat the last yaw for padding
        last_yaw = ego_trajs_best[:, -1, 2].unsqueeze(1).unsqueeze(2)
        padding_yaws = last_yaw.expand(-1, num_padding, -1)
        # Concatenate the original trajectories with the padded trajectories
        ego_trajs_best_padded = torch.cat([ego_trajs_best, torch.cat([padding_positions, padding_yaws], dim=-1)], dim=1)
        assert ego_trajs_best_padded.shape[1] == 32, f"Expected 32 timesteps, got {ego_trajs_best_padded.shape[1]}"
        ego_actions = Action(positions=ego_trajs_best_padded[..., :2], yaws=ego_trajs_best_padded[..., 2:])
        # enlarge from 20 to 32
        padded_zeros = torch.zeros_like(action_samples.positions[:, :, -12:])
        padded_positions = torch.cat([action_samples.positions, padded_zeros], dim=-2)
        action_samples.positions = padded_positions[:, :32]
        padded_zeros = torch.zeros_like(action_samples.yaws[:, :, -12:])
        padded_yaws = torch.cat([action_samples.yaws, padded_zeros], dim=-2)
        action_samples.yaws = padded_yaws[:, :32]

        # action_info["action_samples"] = action_samples.to_dict()
        # if "plan_samples" in action_info:
        #     action_info["plan_samples"] = action_info["plan_samples"].to_dict()
        # enlarge positions, yaws to 32

        return ego_actions, {}  # action_info

    def unwrap(self):
        sampler_unwrap = self.sampler.unwrap() if isinstance(self.sampler, Wrappers) else self.sampler
        predictor_unwrap = self.predictor.unwrap() if isinstance(self.predictor, Wrappers) else self.predictor
        return {"Sampling.sampler": sampler_unwrap, "Sampling.predictor": predictor_unwrap}


class PolicyWrapper(Wrappers):
    """A convenient wrapper for specifying run-time keyword arguments"""

    def __init__(self, model, get_action_kwargs=None, get_plan_kwargs=None):
        self.model = model
        self.device = model.device
        self.action_kwargs = get_action_kwargs
        self.plan_kwargs = get_plan_kwargs

    def eval(self):
        self.model.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        return self.model.get_action(obs, **self.action_kwargs, **kwargs)

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        return self.model.get_plan(obs, **self.plan_kwargs, **kwargs)

    @classmethod
    def wrap_controller(cls, model, **kwargs):
        return cls(model=model, get_action_kwargs=kwargs)

    @classmethod
    def wrap_planner(cls, model, **kwargs):
        return cls(model=model, get_plan_kwargs=kwargs)

    def unwrap(self):
        return self.model.unwrap() if isinstance(self.model, Wrappers) else self.model


class RefineWrapper(Wrappers):
    """A wrapper that feeds coarse motion plan to a optimization-based planner for refinement"""

    def __init__(self, initial_planner, refiner, device):
        """
        Args:
            planner: a policy that generates a coarse motion plan
            refiner: a policy (optimization based) that takes the coarse motion plan and refine it
            device: device for torch
        """
        self.initial_planner = initial_planner
        self.refiner = refiner
        self.device = device

    def eval(self):
        self.initial_planner.eval()
        self.refiner.eval()

    def get_action(self, obs, **kwargs):
        coarse_plan, _ = self.initial_planner.get_action(obs, **kwargs)
        action, action_info = self.refiner.get_action(obs, coarse_plan=coarse_plan)
        return action, {"coarse_plan": coarse_plan.to_dict(), **action_info}

    def unwrap(self):
        initial_planner_unwrap = (
            self.initial_planner.unwrap() if isinstance(self.initial_planner, Wrappers) else self.initial_planner
        )
        refiner_unwrap = self.refiner.unwrap() if isinstance(self.refiner, Wrappers) else self.refiner
        return {"Refine.initial_planner": initial_planner_unwrap, "Refine.refiner": refiner_unwrap}


# from tbsim.policies.hardcoded import PDMClosed_trajdata


class Pos2YawWrapper(Wrappers):
    """A wrapper that computes action yaw from action positions"""

    def __init__(self, policy, dt, yaw_correction_speed):
        """

        Args:
            policy: policy to be wrapped
            dt:
            speed_filter:
        """
        self.device = policy.device
        self.policy = policy
        self._dt = dt
        self._yaw_correction_speed = yaw_correction_speed

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs):
        action, action_info = self.policy.get_action(obs, **kwargs)
        curr_pos = torch.zeros_like(action.positions[..., [0], :])
        pos_seq = torch.cat((curr_pos, action.positions), dim=-2)

        yaws = yaw_from_pos(
            pos_seq,
            dt=self._dt,
            yaw_correction_speed=self._yaw_correction_speed,
            mode=kwargs.get("mode", "append_last"),
        )

        action.yaws = yaws
        return action, action_info

    def unwrap(self):
        return self.policy.unwrap() if isinstance(self.policy, Wrappers) else self.policy


class RolloutWrapper(Wrappers):
    """A wrapper policy that can (optionally) control both ego and other agents in a scene"""

    def __init__(self, ego_policy=None, agents_policy=None, pass_agent_obs=True):
        self.device = ego_policy.device if agents_policy is None else agents_policy.device
        self.ego_policy = ego_policy
        self.agents_policy = agents_policy
        self.pass_agent_obs = pass_agent_obs

    def eval(self):
        self.ego_policy.eval()
        self.agents_policy.eval()

    def get_action(self, obs, step_index) -> RolloutAction:
        ego_action = None
        ego_action_info = None
        agents_action = None
        agents_action_info = None
        if self.ego_policy is not None:
            assert obs["ego"] is not None
            with torch.no_grad():
                if self.pass_agent_obs:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index=step_index, agent_obs=obs["agents"]
                    )
                else:
                    ego_action, ego_action_info = self.ego_policy.get_action(obs["ego"], step_index=step_index)
        if self.agents_policy is not None:
            assert obs["agents"] is not None
            if obs["agents"]["agent_type"].nelement() == 0:
                agents_action = None
                agents_action_info = None
            else:
                with torch.no_grad():
                    obs["agents"]["ego_plan"] = ego_action.positions if ego_action is not None else None
                    agents_action, agents_action_info = self.agents_policy.get_action(
                        obs["agents"], step_index=step_index
                    )
                    # if ego_idx is in obs["agents"] slice the action
                    if "ego_idx" in obs["agents"]:
                        # remove the ego_idx
                        agents_action.eliminate_ego_action(obs["agents"]["ego_idx"])
                        self.eliminate_ego_action_in_dict(agents_action_info, obs["agents"]["ego_idx"])

        return RolloutAction(ego_action, ego_action_info, agents_action, agents_action_info)

    def unwrap(self):
        ego_unwrap = self.ego_policy.unwrap() if isinstance(self.ego_policy, Wrappers) else self.ego_policy
        agent_unwrap = self.agents_policy.unwrap() if isinstance(self.agents_policy, Wrappers) else self.agents_policy
        return {"Rollout.ego_policy": ego_unwrap, "Rollout.agents_policy": agent_unwrap}

    @staticmethod
    def eliminate_ego_action_in_dict(agents_action_info, ego_idx):
        """
        Eliminate the ego action for all Tensors in a dictionary.

        Args:
            agents_action_info: A dictionary containing Tensor objects.
            ego_idx: Indices of the ego actions to be removed.
        """
        # Ensure the ego_idx is a tensor for index selection
        if not isinstance(ego_idx, torch.Tensor):
            ego_idx = torch.Tensor(ego_idx)

        # Use boolean indexing to keep elements not in ego_idx
        mask = torch.ones(
            agents_action_info[list(agents_action_info.keys())[0]]["positions"].shape[0],
            dtype=bool,
            device=ego_idx.device,
        )
        mask[ego_idx.long()] = False

        # Apply the mask to each Tensor in the dictionary
        for key in agents_action_info.keys():
            if isinstance(agents_action_info[key], dict):
                for sub_key in agents_action_info[key].keys():
                    if agents_action_info[key][sub_key] is None:
                        continue
                    else:
                        agents_action_info[key][sub_key] = agents_action_info[key][sub_key][mask]
            elif key not in ["change_lane_state", "adv_proposals"]:  # TODO modify the ego agent combined actions
                if agents_action_info[key] is not None:
                    agents_action_info[key] = agents_action_info[key][mask]
            else:
                agents_action_info[key]


class PerturbationWrapper(Wrappers):
    """A wrapper policy that perturbs the policy action with Ornstein Uhlenbeck noise"""

    def __init__(self, policy, noise):
        self.device = policy.device
        self.noise = noise
        self.policy = policy

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        actions, action_info = self.policy.get_action(obs, **kwargs)
        actions_dict = OrderedDict(target_positions=actions.positions, target_yaws=actions.yaws)
        perturbed_action_dict = self.noise.perturb(TensorUtils.to_numpy(actions_dict))
        perturbed_action_dict = TensorUtils.to_torch(perturbed_action_dict, self.device)
        perturbed_actions = Action(perturbed_action_dict["target_positions"], perturbed_action_dict["target_yaws"])
        return perturbed_actions, action_info

    def unwrap(self):
        return self.policy.unwrap() if isinstance(self.policy, Wrappers) else self.policy
