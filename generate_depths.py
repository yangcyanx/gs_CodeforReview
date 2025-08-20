import torch
from PIL import Image
from pathlib import Path
from diffusers import DiffusionPipeline
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Marigold depth estimation batch processing tool")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="models/marigold-depth-v1.1")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--valid_depth_min", type=float, default=0.0)
    parser.add_argument("--valid_depth_max", type=float, default=200.0)
    parser.add_argument("--near", type=float, default=0.0)
    parser.add_argument("--far", type=float, default=13)
    parser.add_argument("--reverse_colormap", action="store_true")
    return parser.parse_args()

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

depth_pipe = DiffusionPipeline.from_pretrained(
    args.model_path,
    variant="fp16",
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True
).to(device)

def visualize_depth(
    depth_np: np.ndarray,
    valid_depth_min: float,
    valid_depth_max: float,
    near: float = None,
    far: float = None,
    reverse_colormap: bool = False
) -> np.ndarray:
    valid_mask = (depth_np >= valid_depth_min) & (depth_np <= valid_depth_max)
    if not np.any(valid_mask):
        return np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
    
    valid_depths = depth_np[valid_mask]
    auto_near = valid_depths.min() if near is None else near
    auto_far = valid_depths.max() if far is None else far
    
    eps = np.finfo(np.float32).eps
    curve_fn = lambda x: -np.log(x + eps) 
    depth_curve = curve_fn(depth_np)
    near_curve = curve_fn(auto_near)
    far_curve = curve_fn(auto_far)
    
    if np.isclose(near_curve, far_curve):
        depth_norm = np.zeros_like(depth_curve)
    else:
        depth_norm = (depth_curve - near_curve) / (far_curve - near_curve)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    
    colormap = plt.get_cmap('turbo')
    if reverse_colormap:
        colormap = colormap.reversed()
    depth_vis = colormap(depth_norm)[:, :, :3]
    
    depth_vis[~valid_mask] = 0.0
    
    return (depth_vis * 255).astype(np.uint8)

def process_depth(image_path: Path, output_dir: Path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return
    
    try:
        depth_result = depth_pipe(
            image,
            processing_resolution=args.resolution,
            output_type="np"
        )
        depth_map = depth_result.prediction[0].squeeze()  # (H, W)
    except Exception as e:
        return
    
    output_path = output_dir / image_path.stem
    np.save(output_path.with_suffix(".npy"), depth_map)
    
    try:
        depth_vis = visualize_depth(
            depth_np=depth_map,
            valid_depth_min=args.valid_depth_min,
            valid_depth_max=args.valid_depth_max,
            near=args.near,
            far=args.far,
            reverse_colormap=args.reverse_colormap
        )
        Image.fromarray(depth_vis).save(output_path.with_suffix(".png"))
    except Exception as e:
        print(f"Exception {image_path}: {e}")

if __name__ == "__main__":
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"The input directory does not exist:{input_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "predict_depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if not image_paths:
        raise ValueError(f"No image file in the input directory {input_dir}(surport .png/.jpg/.jpeg)")
    if "shiny_blender" in args.input_dir:
        pattern = re.compile(r'^r_\d+\.(png|jpg|jpeg)$')
        image_paths = [
            p for p in 
            (list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
            if pattern.match(p.name)  # 通过正则精确匹配文件名
        ]
    print(f"Start processing: {len(image_paths)} images in total")
    for img_path in image_paths:
        process_depth(img_path, output_dir)
    print(f"Processing completed: save the result to {output_dir}")

    #print(f"start generate_depth: " + str(input_dir))
    #i = 0
    #while True:
    #    if "glossy_syn" in args.input_dir:
    #        img_path = input_dir / (str(i)+".png")
    #    elif "shiny_blender" in args.input_dir:
    #        img_path = input_dir / ("r_"+str(i)+".png")
    #    elif "shiny_real" in args.input_dir:
    #        img_path = input_dir / ("r_"+str(i)+".png")
    #    if os.path.exists(img_path):
    #        process_depth(img_path, output_dir)
    #    else:
    #        break
    #    i += 1
    #print("end generate_depth: " + str(input_dir))