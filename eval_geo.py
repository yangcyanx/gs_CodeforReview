import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_surfel
import torchvision
from scene.dataset_readers import load_img_rgb
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments.geo import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene.ref_gaussian_model import RefGaussianModel as GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

def get_mae(gt_normal, normal):
    mae = (gt_normal*normal).sum(0).clamp(-1, 1).arccos().mean() * 180 / np.pi
    return mae

def render_set(model_path, views, gaussians, pipeline, background, save_ims, opt):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "test", "renders")
        color_path = os.path.join(render_path, 'rgb')
        normal_path = os.path.join(render_path, 'normal')
        metallic_path = os.path.join(render_path, 'metallic')
        diffuse_path = os.path.join(render_path, 'diffuse')
        specular_path = os.path.join(render_path, 'specular')
        base_color_path = os.path.join(render_path, 'base_color')
        roughness_path = os.path.join(render_path, 'roughness')

        visibility_path = os.path.join(render_path, 'visibility')
        indirect_light_path = os.path.join(render_path, 'indirect_light')
        direct_light_path = os.path.join(render_path, 'direct_light')
        indirect_color_path = os.path.join(render_path, 'indirect_color')
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(metallic_path, exist_ok=True)
        makedirs(diffuse_path, exist_ok=True)
        makedirs(specular_path, exist_ok=True)
        makedirs(base_color_path, exist_ok=True)
        makedirs(roughness_path, exist_ok=True)
        makedirs(visibility_path, exist_ok=True)
        makedirs(indirect_light_path, exist_ok=True)
        makedirs(direct_light_path, exist_ok=True)
        makedirs(indirect_color_path, exist_ok=True)

    ssims = []
    psnrs = []
    lpipss = []
    mae_normal = []
    ssims_normal = []
    lpipss_normal = []
    normal_bg = torch.tensor([0.0, 0.0, 1.0], device='cuda')
    render_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        t1 = time.time()
        
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        render_time = time.time() - t1
        
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(lpips(render_color, gt, net_type='vgg').item())

        if view.mask is None or view.mask.shape[-1] == 0:
            mask = np.ones((render_color.shape[0], render_color.shape[1], 1))
        else:
            mask = view.mask.float()
        normal = rendering['rend_normal']
        alpha = rendering['rend_alpha']
        normal = normal * alpha + normal_bg[:, None, None] * (1.0 - alpha)
        normal = F.normalize(normal, dim=0)
        
        normal_gt_path = os.path.join(args.source_path, "normals", view.image_name + "-normal.png")
        if os.path.exists(normal_gt_path) and os.path.isdir(normal_gt_path):
            gt_normal_img = torch.from_numpy(load_img_rgb(normal_gt_path)[..., :3]).float().cuda().permute(2, 0, 1)
            gt_normal = (gt_normal_img - 0.5) * 2.0
            gt_normal = gt_normal * mask + normal_bg[:, None, None] * (1.0 - mask)
            gt_normal = F.normalize(gt_normal, dim=0)
            mae_normal.append(get_mae(gt_normal, normal).mean().double().item())
            ssims_normal.append(ssim(normal, gt_normal).item())
            lpipss_normal.append(lpips(normal, gt_normal, net_type='vgg').item())

        render_times.append(render_time)

        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            if 'visibility' in rendering:
                torchvision.utils.save_image(rendering['visibility'], os.path.join(visibility_path, '{0:05d}.png'.format(idx)))
            if 'indirect_light' in rendering:
                torchvision.utils.save_image(rendering['indirect_light'], os.path.join(indirect_light_path, '{0:05d}.png'.format(idx)))
            if 'direct_light' in rendering:
                torchvision.utils.save_image(rendering['direct_light'], os.path.join(direct_light_path, '{0:05d}.png'.format(idx)))
            if 'indirect_color' in rendering:
                torchvision.utils.save_image(rendering['indirect_color'], os.path.join(indirect_color_path, '{0:05d}.png'.format(idx)))
            if 'metallic_map' in rendering:
                torchvision.utils.save_image(rendering['metallic_map'], os.path.join(metallic_path, '{0:05d}.png'.format(idx)))
            if 'diffuse_map' in rendering:
                torchvision.utils.save_image(rendering['diffuse_map'], os.path.join(diffuse_path, '{0:05d}.png'.format(idx)))
            if 'specular_map' in rendering:
                torchvision.utils.save_image(rendering['specular_map'], os.path.join(specular_path, '{0:05d}.png'.format(idx)))
            if 'base_color_map' in rendering:
                torchvision.utils.save_image(rendering['base_color_map'], os.path.join(base_color_path, '{0:05d}.png'.format(idx)))
            if 'roughness_map' in rendering:
                torchvision.utils.save_image(rendering['roughness_map'], os.path.join(roughness_path, '{0:05d}.png'.format(idx)))
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            
            
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()

    
    mae_normal_v = 0 if len(mae_normal) == 0 else np.array(mae_normal).mean()
    ssim_normal_v = 0 if len(ssims_normal) == 0 else np.array(ssims_normal).mean()
    lpipss_normal_v = 0 if len(lpipss_normal) == 0 else np.array(lpipss_normal).mean()
    fps = 1.0 / np.array(render_times).mean()
    print('psnr:{}, ssim:{}, lpips:{}, fps:{}, mae_normal:{}, ssims_normal:{}, lpipss_normal:{}'.format(psnr_v, ssim_v, lpip_v, fps, mae_normal_v, ssim_normal_v, lpipss_normal_v))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}, mae_normal:{}, ssims_normal:{}, lpipss_normal:{}'.format(psnr_v, ssim_v, lpip_v, fps, mae_normal_v, ssim_normal_v, lpipss_normal_v))

def render_set_train(model_path, views, gaussians, pipeline, background, save_ims, opt):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "train", "renders")
        color_path = os.path.join(render_path, 'rgb')
        gt_path = os.path.join(render_path, 'gt')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
 
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, :3, :, :]

        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}.png'.format(idx)))
            # Save the normal map if available
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            

            

   
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        #gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)

        
        # render_set_train(dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, save_ims, op)
        render_set(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background, save_ims, op)
        
        env_dict = gaussians.render_env_map_1()
        grid = [
            env_dict["env1"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        grid = [
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env2.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, True)
