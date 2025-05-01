import math

from tbsim.configs.base import AlgoConfig


class BehaviorCloningConfig(AlgoConfig):
    def __init__(self):
        super(BehaviorCloningConfig, self).__init__()
        self.eval_class = "BC"

        self.name = "bc"
        self.model_architecture = "resnet18"
        self.map_feature_dim = 256
        self.history_num_frames = 10
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 32
       
        self.prediction_length = 32
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True
        # self.dynamics.type = None
        self.dynamics.type = "Unicycle" #Unicycle, Frenet
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        # self.dynamics.type = "Frenet" #Unicycle, Frenet
        # self.dynamics.max_n_dot = 5
        # self.dynamics.max_t_dot = 50
        
        self.output_frenet = False
        if self.output_frenet:
            # self.dynamics.type = None
            self.dynamics.type = "Frenet" #Unicycle, Frenet
            self.dynamics.max_n_dot = 5
            self.dynamics.max_t_dot = 50
            self.loss_weights.prediction_loss_xy = 1.0
        # self.output_frenet = True
        self.use_ROI = False
        
        ## closed-loop training config
        self.closed_loop_train = False
        if self.closed_loop_train:
            self.future_num_frames = 20
            self.prediction_length = 20
            self.num_step_actions = 10
            self.use_ROI = True
        
            if self.use_ROI:
                self.roi_config.agent_feature_dim = 256
                self.roi_config.global_feature_dim = 128
                self.roi_config.roi_size = (50, 50)
                self.roi_config.use_rotated_roi = True
                self.roi_config.use_transformer = False
                self.roi_config.roi_layer_key = "layer2"
                self.roi_config.use_GAN = False
                self.roi_config.history_conditioning = False
        else:
            assert self.prediction_length == self.future_num_frames

        if not self.use_ROI:
            self.roi_config = None
        self.state_dim = 2 if self.output_frenet else 3

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.001

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

class DiffusionTrafficConfig(AlgoConfig):
    def __init__(self):
        super(DiffusionTrafficConfig, self).__init__()
        self.eval_class = "Diffusion" # For policy_composer in evaluation/policy_composers.py

        self.name = "diffusion"
        self.model_architecture = "resnet18"
        self.rasterize_mode = "point"
        self.map_feature_dim = 256
        self.history_num_frames = 10
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 32 ## should be 2^N for Unet!
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        # self.dynamics.type = None
        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.001

        self.optim_params.policy.learning_rate.initial = 1e-5  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


        ## Training weight scaling
        self.weight_scaling = None

        
        ## Model architecture
        self.drop_cond_prob = 0.0



        self.use_ROI = False

        self.get_map_mode = False

        ## Diffusion net configs
        self.guidance_mode = False
        self.guidance_fn = None
        self.horizon = self.future_num_frames
        self.n_diffusion_steps = 100
        self.action_weight = 10

        self.ref_encoder_config = {
                "position_lstm_config": 
                    {"input_size": 2,
                    "hidden_size": 64 }
                    }

        self.transition_dim = 6
        self.context_dim = 384 #+64
        self.nt_history_encoder_config = None


        self.context_dim = self.context_dim 

        self.scale_net_output = False
        self.loss_discount = 1
        self.predict_epsilon = False
        self.dim_mults = (1, 2, 4)
        self.attention = False

        assert self.horizon == self.future_num_frames
        ## History Encoder Config
        self.history_encoder_config = {
            "position_lstm_config": 
                {"input_size": 9,
                "hidden_size": 64 }
                }

        self.diffuse_config.num_points = self.horizon  # horizon
        self.diffuse_config.num_samples = 20
        self.diffuse_config.bestof = True
        self.diffuse_config.flexibility = 0.0
        self.diffuse_config.ret_traj = False
        self.diffuse_config.sampling_mode = 'ddpm'
        self.diffuse_config.sample_step = 1
        

class DiffSceneConfig(AlgoConfig):
    def __init__(self):
        super(DiffSceneConfig, self).__init__()
        self.eval_class = "DiffScene" # For policy_composer in evaluation/policy_composers.py

        self.name = "diffscene"
        self.model_architecture = "resnet18"
        self.rasterize_mode = "point"
        self.map_feature_dim = 256
        self.history_num_frames = 10
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 60 ## should be 2^N for Unet!
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = None
        # self.dynamics.type = "Unicycle"
        # self.dynamics.max_steer = 0.5
        # self.dynamics.max_yawvel = math.pi * 2.0
        # self.dynamics.acce_bound = (-10, 8)
        # self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        # self.dynamics.max_speed = 40.0  # roughly 90mph

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.001

        self.optim_params.policy.learning_rate.initial = 1e-5  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


        ## Training weight scaling
        self.weight_scaling = None
        # self.weight_scaling = [1.0,1.0,1.0,1.0,1.0,1.0]
        # self.weight_scaling =[0.1,2.0,0.5,5,0.5,10.0] #a w x,y,v,theta
        # self.weight_scaling = [0.0,0.0,1,10,0.0,0.0]

        
        ## Model architecture
        self.query_unet = False
        self.only_cnn = False
        self.drop_cond_prob = 0.0
        self.output_frenet = False


        self.use_ROI = False

        

        ## Diffusion net configs
        self.guidance_mode = False
        self.guidance_fn = None
        self.horizon = self.future_num_frames
        self.n_diffusion_steps = 100
        self.action_weight = 10

        self.ref_encoder_config = {
                "position_lstm_config": 
                    {"input_size": 2,
                    "hidden_size": 64 }
                    }
        self.encode_ref = False
        self.ref_attention = False
        if self.dynamics.type is None:
            self.transition_dim = 2
            self.context_dim = 384 
            self.nt_history_encoder_config = None
        elif self.only_cnn:
            self.transition_dim = 6
            self.context_dim = 256 #+64
            self.nt_history_encoder_config = None
        elif self.query_unet == False and self.output_frenet == False:
            self.transition_dim = 6
            self.context_dim = 384 #+64
            self.nt_history_encoder_config = None
        elif self.output_frenet==True:
            self.transition_dim = 6
            self.context_dim = 384 +64
            # self.dynamics.type = None
            self.nt_history_encoder_config = {
                "position_lstm_config": 
                    {"input_size": 2,
                    "hidden_size": 64 }
                    }
            self.weight_scaling = [10.0,1.0]
        else:
            self.transition_dim = 7
            self.context_dim = 128 
            self.nt_history_encoder_config = None

        self.context_dim = self.context_dim + 64 * int(self.encode_ref) 

        self.scale_net_output = False
        self.loss_discount = 1
        self.predict_epsilon = False
        self.dim_mults = (1, 2, 4)
        self.attention = False

        ## closed-loop training config
        self.closed_loop_train = False
        if self.closed_loop_train:
            self.use_ROI = True
            assert self.use_ROI
            self.future_num_frames = 40
            self.horizon = 20
            self.num_step_actions = 10
        else:
            assert self.horizon == self.future_num_frames
        
        if not self.use_ROI:
            self.roi_config = None
        else:
            self.roi_config.agent_feature_dim = 256
            self.roi_config.global_feature_dim = 128
            self.roi_config.roi_size = (150, 150)
            self.roi_config.use_rotated_roi = True
            self.roi_config.use_transformer = False
            self.roi_config.roi_layer_key = "layer2"
            self.roi_config.use_GAN = False
            self.roi_config.history_conditioning = False
        ## History Encoder Config
        self.history_encoder_config = {
            "position_lstm_config": 
                {"input_size": 9,
                "hidden_size": 64 }
                }


        self.diffuse_config.num_points = self.horizon  # horizon
        self.diffuse_config.sample = 20
        self.diffuse_config.bestof = True
        self.diffuse_config.flexibility = 0.0
        self.diffuse_config.ret_traj = False
        self.diffuse_config.sampling = 'ddpm'
        self.diffuse_config.step = 1
        
   
        # self.diffuse_config.guidance_mode = "clean_guide"
        # Additional diffuse_config guidance attributes
        self.guide_config.batch_ctrl_indices = None #Which agent to control
        self.guide_config.guidance_fn = ["route","collision","speed", "speed"]
        self.guide_config.params.guidance_horizon = None #how many steps we calculate strategy
        self.guide_config.params.n_guide_steps = 1
        self.guide_config.params.scale_grad_by_std = False
        self.guide_config.params.inner_lr = 0.5 
        self.guide_config.params.inner_beta = 0.5
        self.guide_config.params.multiple_guidance_strategy = "weight_guide"
        self.guide_config.params.grad_wrt = "clean_guide"
        
        # adv guided  --temporary for ddim
        self.guide_config.truncated_t = 30
        self.guide_config.sample_mode = 'ddim'
        self.guide_config.sample_step = 5
        

        # STL guidance initialize_parameters static!
        self.guide_config.route_config.lane_margin = 1.0
        self.guide_config.route_config.loss_timesteps = self.horizon
        self.guide_config.route_config.filter_timesteps = self.horizon
        self.guide_config.route_config.change_lane_prob = 0.0
        
        self.guide_config.speed_config.desired_speed = 5
        # self.guide_config.speed_config.loss_timesteps = self.horizon
        # self.guide_config.speed_config.filter_timesteps = self.horizon

        self.guide_config.collision_config.radius = 2
        self.guide_config.collision_config.mode = "gaussian"
        self.guide_config.collision_config.sigma = 1
        self.guide_config.collision_config.heading_weight = 1
        self.guide_config.collision_config.buff_dist = 1.5
        self.guide_config.collision_config.prediction_mode = "multi_agent"
        self.guide_config.collision_config.loss_timesteps = 20
        self.guide_config.collision_config.filter_timesteps = 20

        self.guide_config.ttc_config.distance_bandwidth = 1.0
        self.guide_config.ttc_config.time_bandwidth = 1.0
        self.guide_config.ttc_config.min_velocity_diff = 0.1
        self.guide_config.ttc_config.loss_timesteps = 20
        self.guide_config.ttc_config.filter_timesteps = 20
        
        self.guide_config.smooth_config.loss_timesteps = 20
        self.guide_config.smooth_config.filter_timesteps = 20

        self.guide_config.combineloss_config.filter_criterion = "individual"
        self.guide_config.combineloss_config.filter_target = "collision" #collision, route, or combine
        self.guide_config.combineloss_config.weights = [0.5,0.5,0.0,0.0]
        
        # self.guide_config.combineloss_config.ctrl_weights = [0.5,0.5,0.0,0.5]
        # self.guide_config.combineloss_config.ctrl_filter_target = "causecollision"
        # self.guide_config.combineloss_config.ctrl_filter_criterion = "heirachical" 
        
        self.guide_config.combineloss_config.ctrl_weights = [0.5,0.5,0.0,0.5]
        self.guide_config.combineloss_config.ctrl_filter_target = "causecollision"
        self.guide_config.combineloss_config.ctrl_filter_criterion = "individual" 
        
        self.guide_config.combineloss_config.switching_mode = False
        
        # self.do_guidance = True




        

class SpatialPlannerConfig(BehaviorCloningConfig):
    def __init__(self):
        super(SpatialPlannerConfig, self).__init__()
        self.eval_class = None

        self.name = "spatial_planner"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.loss_weights.pixel_res_loss = 1.0
        self.loss_weights.pixel_yaw_loss = 1.0


class AgentPredictorConfig(BehaviorCloningConfig):
    def __init__(self):
        super(AgentPredictorConfig, self).__init__()
        self.eval_class = "HierAgentAware"

        self.name = "agent_predictor"
        self.agent_feature_dim = 128
        self.global_feature_dim = 128
        self.context_size = (30, 30)
        self.goal_conditional = True
        self.goal_feature_dim = 32
        self.decoder.layer_dims = (128, 128, 128)

        self.use_rotated_roi = False
        self.use_transformer = False
        self.roi_layer_key = "layer2"
        self.use_GAN = False
        self.history_conditioning = False

        self.loss_weights.lane_reg_loss = 0.5
        self.loss_weights.GAN_loss = 0.5

        self.optim_params.GAN.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.GAN.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.GAN.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.GAN.regularization.L2 = 0.00  # L2 regularization strength


class BehaviorCloningGCConfig(BehaviorCloningConfig):
    def __init__(self):
        super(BehaviorCloningGCConfig, self).__init__()
        self.eval_class = None
        self.name = "bc_gc"
        self.goal_feature_dim = 32
        self.decoder.layer_dims = (128, 128)


class EBMMetricConfig(BehaviorCloningConfig):
    def __init__(self):
        super(EBMMetricConfig, self).__init__()
        self.eval_class = None
        self.name = "ebm"
        self.negative_source = "permute"
        self.map_feature_dim = 64
        self.traj_feature_dim = 32
        self.embedding_dim = 32
        self.embed_layer_dims = (128, 64)
        self.loss_weights.infoNCE_loss = 1.0


class OccupancyMetricConfig(BehaviorCloningConfig):
    def __init__(self):
        super(OccupancyMetricConfig, self).__init__()
        self.eval_class = "metric"
        self.name = "occupancy"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.agent_future_cond.enabled = True
        self.agent_future_cond.every_n_frame = 5


class VAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(VAEConfig, self).__init__()
        self.eval_class = "TrafficSim"
        self.name = "vae"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32

        self.dynamics.type = "Unicycle" #Unicycle, Frenet
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.vae.latent_dim = 4
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)

        self.loss_weights.kl_loss = 1e-4


class DiscreteVAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(DiscreteVAEConfig, self).__init__()
        self.eval_class = "TPP"

        self.name = "discrete_vae"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32
        

        self.ego_conditioning = False
        self.EC_feat_dim = 64
        self.vae.latent_dim = 10
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.Gaussian_var = True
        self.vae.recon_loss_type = "NLL"
        self.vae.logpi_clamp = -6.0

        self.loss_weights.kl_loss = 100
        self.loss_weights.EC_coll_loss = 20
        self.loss_weights.deviation_loss = 0.5
        self.eval.mode = "mean"

        self.agent_future_cond.enabled = False
        self.agent_future_cond.feature_dim = 32
        self.agent_future_cond.transformer = True

        self.min_std = 0.1


class TreeAlgoConfig(BehaviorCloningConfig):
    def __init__(self):
        super(TreeAlgoConfig, self).__init__()
        self.eval_class = None

        self.name = "tree"
        self.module_name =  "CNN"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32
        self.stage = 2
        self.num_frames_per_stage = 10
        self.prob_ego_condition = 0.8

        self.ego_conditioning = True
        self.ego_cond_length = self.num_frames_per_stage
        
        self.unet.channels = [32, 64, 128, 128, 256]
        self.unet.strides=[2, 2, 2, 2, 2]
        self.unet.decoder_strides=[2, 2, 2, 2, 2]
        self.unet.desired_size = (256,256)
        self.unet.logpi_clamp = -6.0
        self.M = 3
        self.Gaussian_var = False
        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph
        self.dynamics.axy_bound = [-6.0,6.0]

        self.scene_centric = True

        self.rasterize_mode = "point"

        self.gamma = 0.5
        self.EC_col_adjust = True
        self.vae.latent_dim = 25
        self.vae.num_latent_sample = 4
        self.vae.latent_before_trans = False
        self.vae.latent_embed_dim = 32
        self.vae.condition_dim = 128
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.Gaussian_var = False
        self.vae.recon_loss_type = "MSE"
        self.vae.logpi_clamp = -6.0
        self.num_eval_samples = 10
        self.EC_feat_dim = 64
        self.loss_weights.EC_collision_loss = 10
        self.loss_weights.deviation_loss = 0.5
        self.loss_weights.kl_loss = 20
        self.loss_weights.collision_loss = 8.0
        self.loss_weights.diversity_loss = 0.3
        self.loss_weights.input_loss = 1.0

        self.input_weight_scaling = [0.01,0.03]
        self.eval.mode = "sum"

        
        self.shuffle = True
        self.vectorize_lane = False
        self.min_std = 0.1
        self.perturb.enabled=True
        self.perturb.N_pert = 2
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = 2.0
        # self.perturb.OU.scale = [1.0,0.3]
        self.perturb.OU.scale = [1.0,1.0,0.1]


class BehaviorCloningECConfig(BehaviorCloningConfig):
    def __init__(self):
        super(BehaviorCloningECConfig, self).__init__()
        self.eval_class = None

        self.name = "bc_ec"
        self.map_feature_dim = 256
        self.goal_conditional = True
        self.goal_feature_dim = 32

        self.EC.feature_dim = 64
        self.EC.RNN_hidden_size = 32
        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.yaw_reg_loss = 0.01
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 4
        self.loss_weights.EC_collision_loss = 5
        self.loss_weights.deviation_loss = 0.2

class UnetConfig(BehaviorCloningConfig):
    def __init__(self):
        super(UnetConfig, self).__init__()
        self.ego_conditioning = True
        self.ego_cond_length = 10
        self.unet.channels = [32, 64, 128, 128, 256, 256]
        self.unet.strides=[2, 2, 2, 2, 2, 2]
        self.unet.decoder_strides=[2, 2, 2, 2, 2, 1]
        self.unet.desired_size = (256,256)
        self.M = 4
        self.Gaussian_var = False
        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

class ScePTConfig(BehaviorCloningConfig):
    def __init__(self):
        super(ScePTConfig, self).__init__()
        self.name = "scept"
        self.eval_class = "scept"
        self.adj_radius.PEDESTRIAN.PEDESTRIAN = 3.0
        self.adj_radius.PEDESTRIAN.VEHICLE = 5.0
        self.adj_radius.VEHICLE.PEDESTRIAN = 5.0
        self.adj_radius.VEHICLE.VEHICLE = 20.0
        self.use_lane_info = False
        self.use_lane_dec = True
        self.use_scaler = False
        self.pred_num_samples = 4
        self.eval_num_smaples = 10
        self.safety_horizon = 10
        self.log_pi_clamp = -8.0
        self.max_clique_size = 4
        self.enc_rnn_dim_edge = 32
        self.enc_rnn_dim_history = 32
        self.enc_rnn_dim_future = 32
        self.dec_rnn_dim = 128
        self.RNN_proj_hidden_dim = [64]
        self.edge_encoding_dim = 32
        self.node_encoding_dim = 32
        self.log_p_yt_xz_max = 6
        self.latent_dim = 4
        self.score_net_hidden_dim = [32]
        self.obs_enc_dim = 32
        self.obs_net_internal_dim = 16
        self.policy_obs_LSTM_hidden_dim = 64
        self.policy_state_LSTM_hidden_dim = 64
        self.policy_FC_hidden_dim = [128,64]
        self.max_greedy_sample = 10
        self.max_random_sample = 10
        self.node_pre_encode_dim = 32
        self.ego_conditioning = True
        self.use_map_encoding = True
        self.map_feature_dim = 128
        self.scene_centric = True
        self.use_proj_dis = True
        self.goal_conditional = False
        self.gamma = 0.5
        self.stage = 1
        self.num_frames_per_stage = 20
        self.num_eval_samples = 10
        self.output_var = True
        self.UAC = True

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.kl_loss = 1.0
        self.loss_weights.collision_loss = 3.0
        self.loss_weights.diversity_loss = 0.3
        self.loss_weights.deviation_loss = 0.1

        self.map_encoder.model_arch = "resnet18"
        self.node_pre_encode_net.VEHICLE = "VEH_pre_encode"
        self.node_pre_encode_net.PEDESTRIAN = "PED_pre_encode"

        self.edge_pre_enc_net.VEHICLE.VEHICLE = "VEH_VEH_encode"
        self.edge_pre_enc_net.VEHICLE.PEDESTRIAN = "VEH_PED_encode"
        self.edge_pre_enc_net.PEDESTRIAN.VEHICLE = "PED_VEH_encode"
        self.edge_pre_enc_net.PEDESTRIAN.PEDESTRIAN = "PED_PED_encode"


        self.dynamics.vehicle.type = "Unicycle"
        self.dynamics.vehicle.max_steer = 0.5
        self.dynamics.vehicle.max_yawvel = math.pi * 2.0
        self.dynamics.vehicle.acce_bound = (-10, 8)
        self.dynamics.vehicle.max_speed = 40.0  # roughly 90mph

        self.dynamics.pedestrain.type = "DoubleIntegrator"
        self.dynamics.pedestrain.axy_bound = [-6.0,6.0]
        self.dynamics.pedestrain.max_speed = 5.0

        self.perturb.enabled=True
        self.perturb.N_pert = 1
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = 2.0
        self.perturb.OU.scale = [1.0,0.3]


class AgentFormerConfig(AlgoConfig):
    def __init__(self):
        super(AgentFormerConfig, self).__init__()
        self.name = "agentformer"
        self.seed = 1
        self.load_map = False
        self.step_time = 0.1
        self.history_num_frames = 10
        self.future_num_frames = 20
        self.traj_scale = 10
        self.nz = 32
        self.sample_k = 4
        self.tf_model_dim = 256
        self.tf_ff_dim = 512
        self.tf_nhead = 8
        self.tf_dropout = 0.1
        self.z_tau.start = 0.5
        self.z_tau.finish = 0.0001
        self.z_tau.decay = 0.5
        self.input_type=['scene_norm', 'vel', 'heading']
        self.fut_input_type = ['scene_norm', 'vel', 'heading']
        self.dec_input_type = ['heading']
        self.pred_type = "scene_norm"
        self.sn_out_type = 'norm'
        self.sn_out_heading = False
        self.pos_concat = True
        self.rand_rot_scene = False
        self.use_map = True
        self.pooling = "mean"
        self.agent_enc_shuffle = False
        self.vel_heading = False
        self.max_agent_len = 128
        self.agent_enc_learn = False
        self.use_agent_enc = False
        self.motion_dim = 2
        self.forecast_dim = 2
        self.z_type = "gaussian"
        self.nlayer = 6
        self.ar_detach = True
        self.pred_scale = 1.0
        self.pos_offset = False
        self.learn_prior = True
        self.discrete_rot = False
        self.map_global_rot = False
        self.ar_train = True
        self.max_train_agent = 100
        self.num_eval_samples = 5
        
        self.UAC = True  # compare unconditional and conditional prediction

        self.loss_cfg.kld.min_clip = 1.0
        self.loss_cfg.sample.weight = 1.0
        self.loss_cfg.sample.k = 20
        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.kl_loss = 1.0
        self.loss_weights.collision_loss = 3.0
        self.loss_weights.EC_collision_loss = 5.0
        self.loss_weights.diversity_loss = 0.3
        self.loss_weights.deviation_loss = 0.1
        self.scene_orig_all_past = False
        self.conn_dist = 100000.0
        self.scene_centric = True
        self.stage = 2
        self.num_frames_per_stage = 10

        self.ego_conditioning = True
        self.perturb.enabled=True
        self.perturb.N_pert = 1
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = 2.0
        self.perturb.OU.scale = [1.0,0.3]

        self.map_encoder.model_architecture = "resnet18"
        self.map_encoder.image_shape = [3,224,224]
        self.map_encoder.feature_dim = 32
        self.map_encoder.spatial_softmax.enabled=False
        self.map_encoder.spatial_softmax.kwargs.num_kp = 32
        self.map_encoder.spatial_softmax.kwargs.temperature = 1.0
        self.map_encoder.spatial_softmax.kwargs.learnable_temperature = False

        self.context_encoder.nlayer = 2

        self.future_decoder.nlayer = 2
        self.future_decoder.out_mlp_dim = [512,256]
        self.future_encoder.nlayer = 2

        self.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength






class GANConfig(BehaviorCloningConfig):
    def __init__(self):
        super(GANConfig, self).__init__()
        self.eval_class = "GAN"

        self.name = "gan"

        self.map_feature_dim = 256
        self.optim_params.disc.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.policy.learning_rate.initial = 1e-4  # generator learning rate

        self.decoder.layer_dims = (128, 128)

        self.traj_encoder.rnn_hidden_size = 100
        self.traj_encoder.feature_dim = 32
        self.traj_encoder.mlp_layer_dims = (128, 128)

        self.gan.latent_dim = 4
        self.gan.loss_type = "lsgan"
        self.gan.disc_layer_dims = (128, 128)
        self.gan.num_eval_samples = 10

        self.loss_weights.prediction_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.0
        self.loss_weights.gan_gen_loss = 1.0
        self.loss_weights.gan_disc_loss = 1.0

        self.optim_params.disc.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.disc.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.disc.regularization.L2 = 0.00  # L2 regularization strength



class SQPMPCConfig(AlgoConfig):
    def __init__(self):
        super(SQPMPCConfig, self).__init__()

        self.name = "MPC"

        self.dt = 0.3
        self.pred_dt = 0.1
        self.horizon_sec = 3.0
        self.distance_threshold = 15.0
        self.delta_t_max = 4

        self.dynamic.PEDESTRIAN.name="DoubleIntegrator"
        self.dynamic.PEDESTRIAN.attributes = dict(abound=[2.0,2.0],vbound=[2.0,2.0])
        self.dynamic.PEDESTRIAN.limits = {}

        self.dynamic.VEHICLE.name = "Unicycle"
        self.dynamic.VEHICLE.attributes = dict(max_steer=0.5, max_yawvel=8, acce_bound=[-6, 4], vbound=[-10, 30])

        self.dynamic.BICYCLE = self.dynamic.VEHICLE
        self.dynamic.MOTORCYCLE = self.dynamic.VEHICLE
        self.dynamic.ego = self.dynamic.VEHICLE


        self.loss_weights.collision_weight = 10.0
        self.loss_weights.lane_weight = 1.0
        self.loss_weights.progress_weight = 0.3
        self.loss_weights.likelihood_weight = 0.2
        
        self.MPCCost.EGO.Q = [0.3, 1.0, 0.3, 0.4]
        self.MPCCost.EGO.Qf = [0.0, 0.0, 0.0, 0.0]
        self.MPCCost.EGO.R = [0.1, 0.6]
        self.MPCCost.EGO.dR = [0.1,0.5]
        
        self.MPCCost.VEHICLE.Q = [0.4, 0.4,0,0.3]
        self.MPCCost.VEHICLE.R = [0.1, 0.1]
        self.MPCCost.VEHICLE.dR = [0.1,0.5]
        self.MPCCost.Mcoll = 1e4
        self.MPCCost.Mlane = 3e3
        self.MPCCost.ego_weight = 1.0
        self.MPCCost.obj_weight = 1.5
        
        self.rot_Q = False
        self.slack_strat = "Linf"
        self.solver_name = "FPMPC"

        self.homo_candiate_num = 5
        self.num_dynamic_object = 4
        self.num_static_object = 6
        self.code_gen = False
        self.qp_solver = "GUROBI"
        self.angle_constraint = False

        self.offsetX = 0.5
        self.offsetY = 0.2
        self.angle_scale=0.5
        self.temp=5.0
        self.ignore_heading_grad = True
        self.num_rounds = 5
        self.lane_change_interval = 6
