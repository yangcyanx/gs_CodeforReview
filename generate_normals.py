import re
import torch
from PIL import Image
from pathlib import Path
from diffusers import MarigoldNormalsPipeline
import numpy as np
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="Marigold batch processing tool for normal estimation")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/marigold-normals-v1.1"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768
    )
    return parser.parse_args()

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/marigold-normals-v1.1"

normals_pipe = MarigoldNormalsPipeline.from_pretrained(
    model_path,
    variant="fp16",        
    torch_dtype=torch.float16,
    use_safetensors=True, 
    local_files_only=True  
).to(device)

def process_normals(image_path: Path, output_dir: Path):
    image = Image.open(image_path).convert("RGB")
    normals_result = normals_pipe(
        image,
        #denoising_steps=4,          
        processing_resolution=768,    
        output_type="np"              
    )
    normals = normals_result.prediction[0] 
    normals = np.clip(normals, -1.0, 1.0)
    normals_uint8 = (normals * 127.5 + 127.5).astype(np.uint8)
    output_path = output_dir / f"{image_path.stem}.png"
    Image.fromarray(normals_uint8).save(output_path)
    return normals

def validate_normal_magnitude(normals):
    magnitudes = np.linalg.norm(normals, axis=-1)
    max_error = np.max(np.abs(magnitudes - 1.0))
    avg_error = np.mean(np.abs(magnitudes - 1.0))
    print(f"Peak modulus length error:{max_error:.6f}\n average module length error:{avg_error:.6f}")

input_dir = Path(args.input_dir)
if not input_dir.exists():
    raise ValueError(f"The input directory does not exist:{input_dir}")

output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "predict_normals"
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
    normals = process_normals(img_path, output_dir)
    #validate_normal_magnitude(normals)
print(f"Processing completed: save the result to {output_dir}")