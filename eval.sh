###################################Glossy_Syn#########################################
# angel
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/angel/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/angel/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/angel/ir --albedo_rescale 1.05 1.0 0.9 -e light
# bell
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/bell/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/bell/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/bell/ir -e light
# cat
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/cat/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/cat/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/cat/ir -e light
# horse
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/horse/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/horse/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/horse/ir --albedo_rescale 1.568 1.6 1.44 -e light
# luyu
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/luyu/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/luyu/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/luyu/ir --albedo_rescale 0.784 0.8 0.72 -e light
# potion
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/potion/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/potion/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/potion/ir --albedo_rescale 0.392 0.4 0.36 -e light
# tbell
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/tbell/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/tbell/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/tbell/ir --albedo_rescale 0.83 1.0 1.55 -e light
# teapot
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/glossy_syn/teapot/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/glossy_syn/teapot/ir --eval --skip_train
CUDA_VISIBLE_DEVICES=0 python relighting_wi_gt.py -s data/relight_gt -m outputs/glossy_syn/teapot/ir -e light
################################ShinyBlender##########################################
# ball
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_blender/ball/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_blender/ball/ir --eval --skip_train --white_background
CUDA_VISIBLE_DEVICES=0 python relighting_wo_gt.py -m outputs/shiny_blender/ball/ir --no_lpips -e light
# car
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_blender/car/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_blender/car/ir --eval --skip_train --white_background
CUDA_VISIBLE_DEVICES=0 python relighting_wo_gt.py -m outputs/shiny_blender/car/ir --no_lpips -e light
# coffee
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_blender/coffee/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_blender/coffee/ir --eval --skip_train --white_background
CUDA_VISIBLE_DEVICES=0 python relighting_wo_gt.py -m outputs/shiny_blender/coffee/ir --no_lpips -e light
# helmet
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_blender/helmet/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_blender/helmet/ir --eval --skip_train --white_background
CUDA_VISIBLE_DEVICES=0 python relighting_wo_gt.py -m outputs/shiny_blender/helmet/ir --no_lpips -e light
# teapot
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_blender/teapot/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_blender/teapot/ir --eval --skip_train --white_background
CUDA_VISIBLE_DEVICES=0 python relighting_wo_gt.py -m outputs/shiny_blender/teapot/ir --no_lpips -e light
# toaster
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_blender/toaster/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_blender/toaster/ir --eval --skip_train --white_background
CUDA_VISIBLE_DEVICES=0 python relighting_wo_gt.py -m outputs/shiny_blender/toaster/ir --no_lpips -e light

###################################ShinyReal##############################################
# gardenspheres
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_real/gardenspheres/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_real/gardenspheres/ir --eval  --skip_train
# sedan
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_real/sedan/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_real/sedan/ir --eval  --skip_train
# toycar
CUDA_VISIBLE_DEVICES=0 python eval_geo.py --save_images -m outputs/shiny_real/toycar/geo
CUDA_VISIBLE_DEVICES=0 python eval_ir.py -m outputs/shiny_real/toycar/ir --eval  --skip_train