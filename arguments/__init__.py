

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.batch_size = 2**16
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_resolution = 128
        self.relight = False
        self.envmap_init_value = 1.5
        self.envmap_activation = 'exp'

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False
        self.light_sample_num = 0
        self.diffuse_sample_num = 512
        self.specular_sample_num = 256
        self.light_t_min = 0.05#0.15#0.05

        self.mlp_from_iter = 20000#20000 
        
        self.wo_indirect = False
        self.wo_indirect_relight = False
        self.detach_indirect = False

        self.is_train_indir = True

        self.use_asg = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 100_000 #100_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.residual_fea_lr = 0.0075 
        self.asg_lr = 0.0075
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lr_scale = 0.0

        self.base_color_lr = 0.0075#0.0075 
        self.base_color_lr_init = self.base_color_lr
        #self.base_color_lr_init = 0.015
        self.base_color_lr_final = 0.00015
        self.base_color_lr_max_steps = 20_000

        self.metallic_lr = 0.005 
        self.metallic_lr_init = self.metallic_lr
        #self.metallic_lr_init = 0.01
        self.metallic_lr_final = 0.0001
        self.metallic_lr_max_steps = 20_000

        self.roughness_lr = 0.005 
        self.roughness_lr_init = self.roughness_lr
        #self.roughness_lr_init = 0.01
        self.roughness_lr_final = 0.0001
        self.roughness_lr_max_steps = 20_000

        self.envmap_cubemap_lr = 0.01
        self.envmap_cubemap_lr_init = self.envmap_cubemap_lr
        #self.envmap_cubemap_lr_init = 0.01
        self.envmap_cubemap_lr_final = 0.0001
        self.envmap_cubemap_lr_max_steps = 20_000

        self.features_lr_max_steps = 20_000

        #self.indir_until_iter = 15000#20000

        self.normal_lr = 0.006

        self.shader_mlp_lr_init = 0.0015

        self.encoding_lr = 0.002
        self.mlp_lr = 0.0005
        
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.01
        self.lambda_depth_smooth = 0.0
        self.lambda_mask_entropy = 0.01
        
        self.lambda_base_color_smooth = 2.0
        self.lambda_roughness_smooth = 2.0
        self.lambda_metallic_smooth = 2.0
        self.lambda_light = 0.1
        self.lambda_light_smooth = 0.0005

        self.lambda_diffuse_smooth = 0.0

        self.lambda_energy_constraint = 0.0
        self.lambda_BRDF_weighted_specular = 0.0

        self.init_roughness_value = 0.2
        self.init_base_color_value = 0.3
        self.init_metallic_value = 0.8

        self.percent_dense = 0.01
        self.densification_interval = 1000#100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000#25000 
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05#0.005

        self.normal_loss_start = 1000
        self.dist_loss_start = 1000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0
        
        self.train_ray = False
        self.trace_num_rays = 2**20#2**18
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
