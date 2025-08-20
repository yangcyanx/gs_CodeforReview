import json
import math
import sys
import os
from gaussian_renderer import render_ir
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.light import EnvMap, EnvLight
from utils.graphics_utils import focal2fov, fov2focal, rgb_to_srgb, srgb_to_rgb
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.system_utils import Timing
from scene.dataset_readers import load_img_rgb
import warnings
warnings.filterwarnings("ignore")
import pickle
import glob

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('-bg', "--background_color", type=float, default=1,
                        help="If set, use it as background color")
    parser.add_argument("--albedo_rescale", nargs=3, default=[0, 0, 0], type=float, help="")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("-e", "--extra", default='', type=str)
    parser.add_argument("--mesh_path", type=str, default = None)
    parser.set_defaults(diffuse_sample_num=2048)
    parser.set_defaults(specular_sample_num=2048)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    scene_name = args.model_path.split('/')[-2]
    # load gaussians
    gaussians = GaussianModel(3)
    
    if args.iteration < 0:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"),args) 
    gaussians.build_bvh()
    
    task_dict = {
        #"raw": {
        #    "capture_list": ["render", "render_env", "roughness", "metallic", "base_color", "base_color_linear", "visibility", "light", "light_indirect", "light_direct", "rend_normal"],
        #    "envmap_path": "data/relight_gt/corridor.exr",
        #},
        "corridor": {
            "capture_list": ["render", "render_env", "roughness", "metallic", "base_color", "base_color_linear", "visibility", "light", "light_indirect", "light_direct", "rend_normal"],
            "envmap_path": "data/relight_gt/corridor.exr",
        },
        "golf": {
            "capture_list": ["render", "render_env", "roughness", "metallic", "base_color", "base_color_linear", "visibility", "light", "light_indirect", "light_direct", "rend_normal"],
            "envmap_path": "data/relight_gt/golf.exr",
        },
        "neon": {
            "capture_list": ["render", "render_env", "roughness", "metallic", "base_color", "base_color_linear", "visibility", "light", "light_indirect", "light_direct", "rend_normal"],
            "envmap_path": "data/relight_gt/neon.exr",
        }
    }
    results_dict = {}

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    
    results_dir = os.path.join(args.model_path, "test_rli" + (f"_{args.extra}" if len(args.extra)>0 else ""))
    os.makedirs(results_dir, exist_ok=True)
    full_cmd = f"python {' '.join(sys.argv)}"
    print("Command: " + full_cmd)
    with open(os.path.join(results_dir, "cmd.txt"), 'w') as cmd_f:
        cmd_f.write(full_cmd)
    
    if args.albedo_rescale == [0, 0, 0]:
        base_color_scale = torch.tensor([0.98, 1.0, 0.9], dtype=torch.float32, device="cuda")
    else:
        base_color_scale = torch.tensor(tuple(args.albedo_rescale), dtype=torch.float32, device="cuda")

    for task_name in task_dict:
        results_dict[task_name] = {}
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        if task_name != "raw":
            gaussians.env_map = EnvLight(path=task_dict[task_name]["envmap_path"], device='cuda', max_res=1024, activation='none').cuda()

        transform = torch.tensor([
            [0, 0, 1], 
            [0, 1, 0], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)

        gaussians.env_map.build_mips()
        gaussians.env_map.update_pdf()

        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "training": False,
            "relight": True,
            "base_color_scale": base_color_scale,
        }
        
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        
        capture_list = task_dict[task_name]["capture_list"]
        if not args.no_save:
            for capture_type in capture_list:
                capture_type_dir = os.path.join(task_dir, capture_type)
                os.makedirs(capture_type_dir, exist_ok=True)

            os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "gt_env"), exist_ok=True)
            
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]

        env_path = os.path.join(args.source_path, scene_name + "_" + envname)
        img_num = len(glob.glob(f'{env_path}/*.pkl'))
        print(img_num)
    
        cams = [read_pickle(f'{env_path}/{k}-camera.pkl') for k in range(img_num)]

        for idx, cam in enumerate(tqdm(cams, leave=False)):
            image_path = os.path.join(env_path,  str(idx) + ".png")
            
            image_rgba = load_img_rgb(image_path)
            image = image_rgba[..., :3]
            mask = image_rgba[..., 3:]
            if mask.shape[-1] == 0:
                mask = np.ones((mask.shape[0], mask.shape[1], 1))
            gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
            gt_image = gt_image * mask
            
            w2c = np.array(cam[0].tolist()+[[0,0,0,1]])
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            fl_x = float(cam[1][0,0])
            fl_y = float(cam[1][1,1])
            
            H = image.shape[0]
            W = image.shape[1]

            fovx = 2 * math.atan(W / (2 * fl_x))
            fovy = 2 * math.atan(H / (2 * fl_y))

            custom_cam = Camera(colmap_id=0, R=R, T=T,
                                FoVx=fovx, FoVy=fovy,
                                image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

            with torch.no_grad():
                render_pkg = render_ir(viewpoint_camera=custom_cam, **render_kwargs)

            render_pkg["render"] = render_pkg["render"] * mask# + (1 - mask) * bg
            gt_image = gt_image * mask# + (1 - mask) * bg
            gt_image_env = gt_image * mask + render_pkg["env_only"] * (1 - mask)
            if not args.no_save:
                save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
                save_image(gt_image_env, os.path.join(task_dir, "gt_env", f"{idx}.png"))
                for capture_type in capture_list:
                    out = render_pkg[capture_type]
                    if 'normal' in capture_type:
                        out = (out + 1) / 2
                    if 'position' in capture_type:
                        out = (out + 1) / 2
                    if out.shape[0] == 1:
                        out = out.repeat(3, 1, 1)
                    save_image(out, os.path.join(task_dir, capture_type, f"{idx}.png"))
            
            with torch.no_grad():
                psnr_pbr += psnr(render_pkg['render'], gt_image).mean().double().item()
                ssim_pbr += ssim(render_pkg['render'], gt_image).mean().double().item()
                if not args.no_lpips:
                    lpips_pbr += lpips(render_pkg['render'], gt_image, net_type='vgg').mean().double().item()
                else:
                    lpips_pbr += 0.0
                    
            # tqdm.write(f"AVG PBR PSNR: {psnr_pbr / (idx + 1): .4f}")
        psnr_pbr /= img_num
        ssim_pbr /= img_num
        lpips_pbr /= img_num
        
        results_dict[task_name]["psnr_pbr"] = psnr_pbr
        results_dict[task_name]["ssim_pbr"] = ssim_pbr
        results_dict[task_name]["lpips_pbr"] = lpips_pbr

        print("\nEvaluating {}: PSNR_PBR {: .4f} SSIM_PBR {: .4f} LPIPS_PBR {: .4f}".format(task_name, psnr_pbr, ssim_pbr, lpips_pbr))

    task_names = list(task_dict.keys())
    results_dict["psnr_pbr_avg"] = np.mean([results_dict[task_name]["psnr_pbr"] for task_name in task_names])
    results_dict["ssim_pbr_avg"] = np.mean([results_dict[task_name]["ssim_pbr"] for task_name in task_names])
    results_dict["lpips_pbr_avg"] = np.mean([results_dict[task_name]["lpips_pbr"] for task_name in task_names])
    print("\nEvaluating AVG: PSNR_PBR {: .4f} SSIM_PBR {: .4f} LPIPS_PBR {: .4f}".format(results_dict["psnr_pbr_avg"], results_dict["ssim_pbr_avg"], results_dict["lpips_pbr_avg"]))
    with open(os.path.join(results_dir, "relighting_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(results_dir, "relighting_results.json"))