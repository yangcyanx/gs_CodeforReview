# 3D Vision 2026 Submission: [oIoNLgPNmX/GOGS: High-Fidelity Geometry and Relighting for Glossy Objects via Gaussian Surfels]
**Anonymous Code Repository for 3DV 2026 Submission**  

## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## ⚙️ Installation
```bash
git clone https://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@github.com/Anonymous-yxy/CodeForReview.git

conda env create --file environment.yml
conda activate gogs

# Install diff-surfel-rasterization and simple-knn
pip install submodules/diff-surfel-rasterization submodules/simple-knn

# Install raytracing (for stage-1 from Ref-Gaussian)
pip install submodules/raytracing

# Install 2D Gaussian Ray Tracer (for stage-2 from IRGS)
cd submodules/surfel_tracer && rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../ && cd ../../
pip install submodules/surfel_tracer

# Install nvdiffrast
pip install submodules/nvdiffrast-main
```

## 📦 Dataset
We mainly test our method on [Shiny Blender Synthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip), [Shiny Blender Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), [Glossy Synthetic](https://liuyuan-pal.github.io/NeRO/). Please run the script `nero2blender.py` to convert the format of the Glossy Synthetic dataset. We use relight_gt from the Glossy Synthetic dataset as the evaluation for relighting.

Put them under the `data` folder:
```bash
data
└── glossy_syn
    └── angel
    └── ...
└── shiny_blender
    └── ball
    └── ...
└── shiny_real
    └── gardenspheres
    └── ...
└── relight_gt
    └── angel_corridor
    └── angel_golf
    └── angel_neon
    └── ...
    └── corridor.exr
    └── golf.exr
    └── neon.exr
```

## 🧠 Models
The pre-trained Marigold models referenced in this work are publicly available on Hugging Face:

Depth estimation: [marigold-depth-v1.1](https://huggingface.co/prs-eth/marigold-depth-v1-1/tree/main)

Normal estimation: [marigold-normals-v1.1](https://huggingface.co/prs-eth/marigold-normals-v1-1/tree/main).

Put them under the `models` folder:
```bash
models
└── marigold-depth-v1.1
└── marigold-normals-v1.1
```
To generate geometric priors for the `Glossy Synthetic`, `Shiny Blender Synthetic`, and `Shiny Blender Real` datasets, execute the following shell scripts respectively:
`generate_normals_depths_glossySyn.sh`, `generate_normals_depths_shinyBlender.sh`, and `generate_normals_depths_shinyReal.sh`.The generated normal prediction and depth prediction are placed as follows：

```bash
data
└── glossy_syn
    └── angel
        └── predict_depth
        └── predict_normals
        └── ...
    └── ...
└── shiny_blender
    └── ball
        └── predict_depth
        └── predict_normals
        └── ...
    └── ...
```
## 🚀 Training
See `train.sh` for training scripts.
### Stage 1: geometry reconstruction
Our method is based on [Ref-Gaussian](https://github.com/fudan-zvg/ref-gaussian) for robust geometry reconstruction.
```bash
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/angel -m outputs/glossy_syn/angel/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
```
### Stage 2: material decomposition
```bash
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/angel --eval --start_checkpoint_geo outputs/glossy_syn/angel/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 -m outputs/glossy_syn/angel/ir --train_ray
```

## 📊 Evaluation
This is the [result](https://huggingface.co/prs-eth/marigold-depth-v1-1/tree/main) of our pre-training for evaluation, just place it in the root directory

See `eval.sh` for evaluation scripts.
### Stage 1: geometry reconstruction
```bash
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/angel/geo
```
### Stage 2: material decomposition
```bash
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/angel/ir --eval --skip_train
```
### Relighting
```bash
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/angel/ir -e light
```
