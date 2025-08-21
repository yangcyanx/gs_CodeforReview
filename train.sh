####################################Glossy_Syn#########################################
# angel
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/angel -m outputs/glossy_syn/angel/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/angel --eval --start_checkpoint_geo outputs/glossy_syn/angel/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 -m outputs/glossy_syn/angel/ir --train_ray
# bell
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/bell -m outputs/glossy_syn/bell/geo --eval -w --lambda_mask_entropy 0.05
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/bell --eval --start_checkpoint_geo outputs/glossy_syn/bell/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 --envmap_resolution 512 -m outputs/glossy_syn/bell/ir --train_ray
# cat
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/cat -m outputs/glossy_syn/cat/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/cat --eval --start_checkpoint_geo outputs/glossy_syn/cat/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 --init_metallic_value 0.9 --envmap_resolution 512 -m outputs/glossy_syn/cat/ir --train_ray
# horse
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/horse -m outputs/glossy_syn/horse/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/horse --eval --start_checkpoint_geo outputs/glossy_syn/horse/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 -m outputs/glossy_syn/horse/ir --train_ray
# luyu
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/luyu -m outputs/glossy_syn/luyu/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/luyu --eval --start_checkpoint_geo outputs/glossy_syn/luyu/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 -m outputs/glossy_syn/luyu/ir --train_ray
# potion
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/potion -m outputs/glossy_syn/potion/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/potion --eval --start_checkpoint_geo outputs/glossy_syn/potion/geo/chkpnt50000.pth --lambda_base_color_smooth 1 --lambda_roughness_smooth 1 --lambda_metallic_smooth 1 --init_roughness_value 0.5 --init_metallic_value 0.5 -m outputs/glossy_syn/potion/ir --train_ray
# tbell
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/tbell -m outputs/glossy_syn/tbell/geo --eval -w --lambda_mask_entropy 0.05 --lambda_normal_smooth 1.0
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/tbell --eval --start_checkpoint_geo outputs/glossy_syn/tbell/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 --envmap_resolution 512 -m outputs/glossy_syn/tbell/ir --train_ray
# teapot
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/glossy_syn/teapot -m outputs/glossy_syn/teapot/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/glossy_syn/teapot --eval --start_checkpoint_geo outputs/glossy_syn/teapot/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --lambda_diffuse_smooth 2 --envmap_resolution 512 -m outputs/glossy_syn/teapot/ir --train_ray

#################################ShinyBlender##########################################
# ball
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_blender/ball -m outputs/shiny_blender/ball/geo --eval -w --lambda_normal_smooth 1.0
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_blender/ball --eval -w --start_checkpoint_geo outputs/shiny_blender/ball/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --init_metallic_value 0.2 -m outputs/shiny_blender/ball/ir --train_ray
# car
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_blender/car -m outputs/shiny_blender/car/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_blender/car --eval --start_checkpoint_geo outputs/shiny_blender/car/geo/chkpnt50000.pth --lambda_base_color_smooth 1 --lambda_roughness_smooth 1 --lambda_metallic_smooth 1 -m outputs/shiny_blender/car/ir --train_ray
# coffee
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_blender/coffee -m outputs/shiny_blender/coffee/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_blender/coffee --eval --start_checkpoint_geo outputs/shiny_blender/coffee/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --init_roughness_value 0.3 --init_metallic_value 0.5 -m outputs/shiny_blender/coffee/ir --train_ray
# helmet
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_blender/helmet -m outputs/shiny_blender/helmet/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_blender/helmet --eval --start_checkpoint_geo outputs/shiny_blender/helmet/geo/chkpnt50000.pth --lambda_base_color_smooth 1 --lambda_roughness_smooth 1 --lambda_metallic_smooth 1 --init_roughness_value 0.3 --init_metallic_value 0.5 -m outputs/shiny_blender/helmet/ir --train_ray
# teapot
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_blender/teapot -m outputs/shiny_blender/teapot/geo --eval -w --lambda_mask_entropy 0.05
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_blender/teapot --eval --start_checkpoint_geo outputs/shiny_blender/teapot/geo/chkpnt50000.pth --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --lambda_metallic_smooth 2 --init_roughness_value 0.3 --init_metallic_value 0.5 -m outputs/shiny_blender/teapot/ir --train_ray
# toaster
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_blender/toaster -m outputs/shiny_blender/toaster/geo --eval -w --lambda_mask_entropy 0.05 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_blender/toaster --eval --start_checkpoint_geo outputs/shiny_blender/toaster/geo/chkpnt50000.pth --lambda_base_color_smooth 1 --lambda_roughness_smooth 1 --lambda_metallic_smooth 1 -m outputs/shiny_blender/toaster/ir --train_ray
###################################ShinyReal##############################################
# gardenspheres
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_real/gardenspheres -m outputs/shiny_real/gardenspheres/geo --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 8 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_real/gardenspheres -m outputs/shiny_real/gardenspheres/ir --start_checkpoint_geo outputs/shiny_real/gardenspheres/geo/chkpnt20000.pth --eval -r 8 --init_roughness_value 0.3 --init_metallic_value 0.7 --train_ray --use_env_scope --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974
# sedan
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_real/sedan -m outputs/shiny_real/sedan/geo --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 8 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_real/sedan -m outputs/shiny_real/sedan/ir --start_checkpoint_geo outputs/shiny_real/sedan/geo/chkpnt20000.pth --eval -r 8 --init_roughness_value 0.3 --init_metallic_value 0.7 --train_ray --use_env_scope --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138
# toycar
CUDA_VISIBLE_DEVICES=0 python train_geo.py -s data/shiny_real/toycar -m outputs/shiny_real/toycar/geo --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 8 --use_predict_normal --use_predict_depth
CUDA_VISIBLE_DEVICES=0 python train_ir.py -s data/shiny_real/toycar -m outputs/shiny_real/toycar/ir --start_checkpoint_geo outputs/shiny_real/toycar/geo/chkpnt20000.pth --eval -r 8 --init_roughness_value 0.3 --init_metallic_value 0.7 --train_ray --use_env_scope --env_scope_center 0.6810 0.8080 4.4550 --env_scope_radius 2.707
