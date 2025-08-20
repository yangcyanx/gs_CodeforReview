
import torch
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.light_utils import reflect
from utils.general_utils import safe_normalize
from utils.point_utils import depth_to_normal, depths_to_points
from utils.graphics_utils import rotation_between_z, fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
from utils.refl_utils import  get_specular_color_surfel, get_full_color_volume, get_full_color_volume_indirect, get_specular_color_surfel2, reflection, saturate_dot
from utils.sh_utils import eval_sh
from utils.sph_utils import cart2sph
from .ref_gaussian import render_initial, render_surfel, render_volume, render_surfel2
import numpy as np
from utils.system_utils import Timing
import trimesh
import nvdiffrast.torch as dr
import kornia
from torchvision.utils import save_image

def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median / render_alpha, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    render_var = render_depth_median - render_depth_expected.square()
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'render_var': render_var,
    }

def render_ir(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opt=None, iteration=-1, training=False, relight=False, base_color_scale=None, material_only=False):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    base_color = pc.get_base_color
    roughness = pc.get_rough
    metallic = pc.get_metallic 
    
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None
    
    shs = pc.get_features
    colors_precomp = None

    res_fea = pc.get_res_fea
   
    features = torch.cat([base_color, roughness, metallic, res_fea], dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    points = surf_depth.permute(1, 2, 0) * viewpoint_camera.rays_d_hw_unnormalized + viewpoint_camera.camera_center
    
    surf_normal = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    surf_normal[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-8)  
    normal_map = F.normalize(normal_map, dim=-1)

    rendered_base_color, rendered_roughness, rendered_metallic, render_res_fea = rendered_features.split([3, 1, 1, 4], dim=0)
    if base_color_scale is not None:
        rendered_base_color = rendered_base_color * base_color_scale[:, None, None]

    if material_only:
        results = {
            "metallic": rendered_metallic * render_alpha,
            "roughness": rendered_roughness * render_alpha,
            "base_color": rgb_to_srgb(rendered_base_color) * render_alpha,
            "base_color_linear": rendered_base_color * render_alpha,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        }
        return results
    
    if training:
        if opt.train_ray:
            mask_alpha = render_alpha[0] > 0
            mask_sum = mask_alpha.sum()
            
            num_pixels = opt.trace_num_rays // (pipe.diffuse_sample_num + pipe.specular_sample_num + pipe.light_sample_num) #2**15
            if num_pixels > mask_sum:
                ray_ids = torch.arange(mask_sum, device='cuda')
            else:
                ray_ids = torch.multinomial(torch.ones(mask_sum, device=mask_sum.device), num_pixels, replacement=False)

            mask_=mask_alpha[mask_alpha]
            mask_[ray_ids]=False
            mask = torch.zeros_like(mask_alpha)
            mask[mask_alpha]=~mask_
        else:
            mask = render_alpha[0] > 0
    else:
        mask = render_alpha[0] > 0

    rays_d = viewpoint_camera.rays_d_hw
    w_o = -rays_d

    if training:
        render_results = rendering_equation(rendered_base_color.permute(1, 2, 0)[mask], rendered_roughness.permute(1, 2, 0)[mask], rendered_metallic.permute(1, 2, 0)[mask], normal_map[mask], points[mask], w_o[mask], pc, pipe=pipe, training=training, camera_center=viewpoint_camera.camera_center)
    else:
        render_results = rendering_equation_chunk(rendered_base_color.permute(1, 2, 0)[mask], rendered_roughness.permute(1, 2, 0)[mask], rendered_metallic.permute(1, 2, 0)[mask], normal_map[mask], points[mask], w_o[mask], pc, pipe=pipe, training=training, relight=relight, camera_center=viewpoint_camera.camera_center)

    diffuse = render_results['diffuse']
    specular = render_results['specular']
    light_direct = render_results['light_direct']

    if not relight and (iteration == -1 or iteration > pipe.mlp_from_iter): 
        feature_map = render_res_fea.permute(1, 2, 0)[mask]
        feature_map = F.normalize(feature_map, dim=-1)

        feature_map = feature_map.reshape(-1, 1, pc.gsfeat_dim)
        feature_dirc = feature_map.reshape(-1, pc.gsfeat_dim)

        # Sph-Mip 
        normals = render_normal.permute(1,2,0)[mask]
        reflect_dir = F.normalize(reflect(-rays_d[mask], normals), dim=-1) #rays_d[mask] #reflect(-rays_d[mask], normals)
        reflect_dir_xy = (cart2sph(reflect_dir.reshape(-1, 3)[..., [0,1,2]])[..., 1:] / torch.Tensor([[np.pi, 2*np.pi]]).cuda())[..., [1,0]]
        reflect_dir_xyz = torch.stack([reflect_dir_xy[:, None, :]], dim=0,)

        spec_level = rendered_roughness.permute(1, 2, 0)[mask]

        spec_feat = pc.dir_encoding(reflect_dir_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
        spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
        spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)

        wrap_input = (spec_feat_wrap @ feature_map).reshape(-1, pc.sph_dim*pc.gsfeat_dim)
        input_mlp = torch.cat([wrap_input, spec_feat_dirc], -1)
        mlp_output = pc.light_mlp(input_mlp).float()
        res_color = torch.exp(torch.clamp(mlp_output, max=5.0))
       
    else:
        res_color = torch.zeros_like(rendered_image.permute(1,2,0)[mask])

    rendered_diffuse = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_diffuse[mask] = diffuse
    rendered_diffuse = rendered_diffuse.permute(2, 0, 1)
        
    rendered_specular = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_specular[mask] = specular
    rendered_specular = rendered_specular.permute(2, 0, 1)

    rendered_res_color = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_res_color[mask] = res_color
    rendered_res_color = rendered_res_color.permute(2, 0, 1)
    rendered_full = rgb_to_srgb(rendered_diffuse + rendered_specular + rendered_res_color)
    final_image = rendered_full * render_alpha + bg_color[:, None, None] * (1 - render_alpha)
        
    final_image_sh = rgb_to_srgb(rendered_image) * render_alpha + bg_color[:, None, None] * (1 - render_alpha)
    
    direct_lights = rgb_to_srgb(pc.get_envmap(rays_d, mode='pure_env').permute(2,0,1))
    env_only = direct_lights
    
    results = {
        "render": final_image,
        "env_only": env_only,
        "render_sh": final_image_sh,
        "diffuse": rgb_to_srgb(rendered_diffuse)* render_alpha,
        "specular": rgb_to_srgb(rendered_specular)* render_alpha,
        "res_color": rgb_to_srgb(rendered_res_color)* render_alpha,
        "mask": mask,
        "roughness": rendered_roughness * render_alpha,
        "metallic": rendered_metallic * render_alpha,
        "base_color": rgb_to_srgb(rendered_base_color) * render_alpha,
        "base_color_linear": rendered_base_color * render_alpha,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        ## normal, accum alpha, dist, depth map
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "ray_light_direct": light_direct,
    }
    
    if opt is not None and training and opt.train_ray:
        alpha = render_alpha.permute(1,2,0)[mask]
        full = diffuse + specular + res_color#diffuse + specular + res_color * (1 - rendered_metallic.permute(1, 2, 0)[mask])
        full = rgb_to_srgb(full)
        ray_rgb = full * alpha + bg_color[None, :] * (1 - alpha)
        results.update({
            "ray_rgb": ray_rgb,
        })
        
    if not training:
        light = render_results['light']
        visibility = render_results['visibility']
        light_indirect = render_results['light_indirect']

        rendered_visibility = torch.zeros_like(rendered_image[:1]).permute(1, 2, 0)
        rendered_visibility[mask] = visibility
        rendered_visibility = rendered_visibility.permute(2, 0, 1) * render_alpha

        rendered_light_indirect = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_indirect[mask] = light_indirect
        rendered_light_indirect = rendered_light_indirect.permute(2, 0, 1) * render_alpha
        
        rendered_light = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light[mask] = light
        rendered_light = rendered_light.permute(2, 0, 1) * render_alpha
        
        rendered_light_direct = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_direct[mask] = light_direct
        rendered_light_direct = rendered_light_direct.permute(2, 0, 1) * render_alpha
        
        final_image_env = rendered_full * render_alpha + direct_lights * (1 - render_alpha)
        
        results.update({
            "render_env": final_image_env,
            "light_direct": rgb_to_srgb(rendered_light_direct),
            "visibility": rendered_visibility,
            "light": rgb_to_srgb(rendered_light),
            "light_indirect": rgb_to_srgb(rendered_light_indirect),
        })

    return results

def rendering_equation_chunk(base_color, roughness, metallic, normal, position, w_o, pc, pipe, training=False, relight=False, chunk_size=2**20, camera_center=None, image_sh=None, feature_dr=None):
    chunk_size = chunk_size // (pipe.diffuse_sample_num + pipe.specular_sample_num + pipe.light_sample_num)
    if base_color.shape[0] <= chunk_size:
        return rendering_equation(base_color, roughness, metallic, normal, position, w_o, pc, pipe, training, relight=relight, camera_center=camera_center, feature_dr=feature_dr)
    else:
        results = []
        for i in range(0, base_color.shape[0], chunk_size):
            results.append(rendering_equation(base_color[i:i+chunk_size], roughness[i:i+chunk_size], metallic[i:i+chunk_size], normal[i:i+chunk_size], position[i:i+chunk_size], w_o[i:i+chunk_size], pc, pipe, training, relight=relight, camera_center=camera_center))
        return {k: torch.cat([r[k] for r in results], 0) for k in results[0]}

def rendering_equation(base_color, roughness, metallic, normals, position, viewdirs, pc, pipe, training=False, relight=False, camera_center=None):
    B = base_color.shape[0]
    envmap = pc.get_envmap

    F0 = 0.04 * (1 - metallic) + metallic * base_color
    
    w_o = viewdirs
    rays_refl, NdotV = reflection(w_o, normals)
    rays_refl = safe_normalize(rays_refl)
    
    
    if pipe.specular_sample_num > 0 and pipe.diffuse_sample_num == 0:
        incident_dirs = sample_specular_directions(
            rays_refl, roughness.view(-1,1), training, pipe.specular_sample_num
        )  # [pn,sn1,3]
        # specualr sample prob. H_s is halfway-vecter
        H_s = w_o.unsqueeze(1) + incident_dirs  # [pn,sn0,3]
        H_s = F.normalize(H_s, dim=-1)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(w_o.unsqueeze(1), H_s)
        specular_pdfs = (
            distribution_ggx(NoH_s, roughness.unsqueeze(1))
            * NoH_s
            / (4 * VoH_s + 1e-8)
        )  # D * NoH / (4 * VoH)
        incident_areas = 1 / specular_pdfs.clamp_min(1e-8)

    elif pipe.diffuse_sample_num > 0 and pipe.specular_sample_num > 0:
        p_diffuse = pipe.diffuse_sample_num / (pipe.diffuse_sample_num + pipe.specular_sample_num)
        p_specular = pipe.specular_sample_num / (pipe.diffuse_sample_num + pipe.specular_sample_num)

        diffuse_directions = sample_diffuse_directions(normals, training, pipe.diffuse_sample_num)  # [pn,sn0,3]
        NoL_d = saturate_dot(diffuse_directions, normals.unsqueeze(1)).clamp_(1e-8, 1)  # [pn, sn0, 1]
        diffuse_pdfs = NoL_d / np.pi * p_diffuse

        specular_directions = sample_specular_directions(
            rays_refl, roughness.view(-1,1), training, pipe.specular_sample_num
        )  # [pn,sn1,3]

        # specualr sample prob. H_s is halfway-vecter
        H_s = w_o.unsqueeze(1) + specular_directions  # [pn,sn0,3]
        H_s = F.normalize(H_s, dim=-1)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(w_o.unsqueeze(1), H_s)
        specular_pdfs = (
            distribution_ggx(NoH_s, roughness.unsqueeze(1))
            * NoH_s
            / (4 * VoH_s + 1e-8) * p_specular
        )  # D * NoH / (4 * VoH)
       
        NoL_s = saturate_dot(normals.unsqueeze(1), specular_directions).clamp_(1e-8, 1)
        diffuse_pdfs_specular = NoL_s / np.pi
        
        H_d = w_o.unsqueeze(1) + diffuse_directions
        H_d = F.normalize(H_d, dim=-1)
        NoH_d = saturate_dot(normals.unsqueeze(1), H_d)
        VoH_d = saturate_dot(w_o.unsqueeze(1), H_d)
        specular_pdfs_diffuse = (
            distribution_ggx(NoH_d, roughness.unsqueeze(1)) 
            * NoH_d
            / (4 * VoH_d + 1e-8)
        )

        diffuse_weights = diffuse_pdfs * p_diffuse + specular_pdfs_diffuse * p_specular
        specular_weights = diffuse_pdfs_specular * p_diffuse + specular_pdfs * p_specular
        
        incident_dirs = torch.cat([diffuse_directions, specular_directions], dim=1)
        incident_pdfs = torch.cat([diffuse_pdfs, specular_pdfs], dim=1)
        incident_weights = torch.cat([diffuse_pdfs * p_diffuse / (diffuse_weights + 1e-8), specular_pdfs * p_specular / (specular_weights + 1e-8)], dim=1)
        incident_areas = incident_weights / incident_pdfs.clamp_min(1e-8)
       
    else:
        raise NotImplementedError
    
    
    global_incident_lights = envmap(incident_dirs, mode='pure_env')
    
    if relight:
        features = torch.cat([pc.get_base_color, pc.get_rough, pc.get_metallic], dim=1)
        trace_outputs = pc.trace(position.unsqueeze(1)+incident_dirs*pipe.light_t_min, incident_dirs, features=features, camera_center=camera_center)
        trace_alpha = trace_outputs['alpha'][..., None]
        incident_visibility = 1 - trace_alpha
        trace_feature = trace_outputs['feature'] / trace_alpha.clamp_min(1e-8)
        trace_normal = F.normalize(trace_outputs['normal'], dim=-1)
        trace_base_color, trace_roughness, trace_metallic = trace_feature.split([3, 1, 1], dim=-1)
        trace_diffuse = trace_base_color * envmap(trace_normal, mode='diffuse')
        trace_wi = -incident_dirs
        trace_NdotV = (trace_normal * trace_wi).sum(-1, keepdim=True)
        trace_reflected = F.normalize(trace_NdotV * trace_normal * 2 - trace_wi, dim=-1)
        trace_F0 = 0.04 * (1 - trace_metallic) + trace_metallic * trace_base_color

        fg_uv = torch.cat([trace_NdotV, trace_roughness], -1).clamp(0, 1)
        fg = dr.texture(pc.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(*fg_uv.shape)
        trace_specular = envmap(trace_reflected, roughness=trace_roughness, mode='specular') * (trace_F0 * fg[..., 0:1] + fg[..., 1:2])

        local_incident_lights = (trace_diffuse + trace_specular) * trace_alpha
        if pipe.wo_indirect_relight:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    else:
        trace_outputs = pc.trace(position.unsqueeze(1)+incident_dirs*pipe.light_t_min, incident_dirs, camera_center=camera_center)
        incident_visibility = 1 - trace_outputs['alpha'][..., None]
        local_incident_lights = trace_outputs['color']
        if pipe.wo_indirect:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        if pipe.detach_indirect:
            incident_visibility = incident_visibility.detach()
            local_incident_lights = local_incident_lights.detach()

    incident_lights = global_incident_lights * incident_visibility + local_incident_lights
    
    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp_(-1, 1)
    fresnel_d, _, _ = fresnel_schlick_directions(
    F0.unsqueeze(1), w_o.unsqueeze(1), diffuse_directions#incident_dirs#diffuse_directions
    )
    f_d = base_color[:, None] / np.pi * (1 - metallic[:, None]) * (1 - fresnel_d)
   
    fresnel, _H, HoV = fresnel_schlick_directions(
    F0.unsqueeze(1), w_o.unsqueeze(1), specular_directions#incident_dirs#specular_directions  # pn,sn,3
    )
    NoV = saturate_dot(normals, w_o).unsqueeze(1)  # pn,1,3
    NoL = saturate_dot(normals.unsqueeze(1), specular_directions)  # pn,sn,3
    geometry = geometry_func(NoV, NoL, roughness.unsqueeze(1))
    NoH = saturate_dot(normals.unsqueeze(1), _H)
    distribution = distribution_ggx(NoH, roughness.unsqueeze(1))
    f_s = fresnel * distribution * geometry / (4 * NoV * NoL + 1e-8)
    
    diffuse_incident_lights = incident_lights[:, :pipe.diffuse_sample_num]
    specular_incident_lights = incident_lights[:, pipe.diffuse_sample_num:]
    diffuse_n_d_i = n_d_i[:, :pipe.diffuse_sample_num]
    specular_n_d_i = n_d_i[:, pipe.diffuse_sample_num:]
    diffuse_incident_areas = incident_areas[:, :pipe.diffuse_sample_num]
    specular_incident_areas = incident_areas[:, pipe.diffuse_sample_num:]
    
    specular = ((f_s) * specular_incident_lights * specular_n_d_i * specular_incident_areas).mean(dim=-2)
    diffuse = ((f_d) * diffuse_incident_lights * diffuse_n_d_i * diffuse_incident_areas).mean(dim=-2)#base_color#torch.zeros_like(specular)#((f_d) * diffuse_incident_lights * diffuse_n_d_i * diffuse_incident_areas).mean(dim=-2)#torch.zeros_like(specular)
   
    if training:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "light_direct": global_incident_lights.mean(dim=1),
            "visibility": incident_visibility.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
        }
    else:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "visibility": incident_visibility.mean(dim=1),
            "light": incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "light_direct": global_incident_lights.mean(dim=1),
        }
    return results
    
def get_squared_roughness(roughness):
       return roughness**2
        
def distribution_ggx(NoH, roughness):
    """
    the ggx distributed D function of cook-torrance brdf.
    """
    roughness = get_squared_roughness(roughness)
    a = roughness
    a2 = a**2
    NoH2 = NoH**2
    denom = NoH2 * (a2 - 1.0) + 1.0
    return a2 / (np.pi * denom**2 + 1e-8)

def get_orthogonal_directions(directions):
    x, y, z = torch.split(directions, 1, dim=-1)  # pn,1
    otho0 = torch.cat([y, -x, torch.zeros_like(x)], -1)
    otho1 = torch.cat([-z, torch.zeros_like(x), x], -1)
    mask0 = torch.norm(otho0, dim=-1) > torch.norm(otho1, dim=-1)
    mask1 = ~mask0
    otho = torch.zeros_like(directions)
    otho[mask0] = otho0[mask0]
    otho[mask1] = otho1[mask1]
    otho = F.normalize(otho, dim=-1)
    return otho

def sample_sphere(num_samples, begin_elevation=0) -> np.ndarray:
    """sample angles from the sphere
    reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
    """
    ratio = (begin_elevation + 90) / 180
    num_points = int(num_samples // (1 - ratio))
    phi = (np.sqrt(5) - 1.0) / 2.0
    azimuths = []
    elevations = []
    for n in range(num_points - num_samples, num_points):
        z = 2.0 * n / num_points - 1.0
        azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
        elevations.append(np.arcsin(z))
    return np.array(azimuths), np.array(elevations)

def sample_diffuse_directions(normals, is_train, diffuse_sample_num = 256):
    # normals [pn,3]
    z = normals  # pn,3
    x = get_orthogonal_directions(normals)  # pn,3
    y = torch.cross(z, x, dim=-1)  # pn,3
    # y = torch.cross(z, x, dim=-1) # pn,3
    # project onto this tangent space
    # predefined diffuse sample directions
    
    az, el = sample_sphere(diffuse_sample_num, 0)
    az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
    diffuse_direction_samples = np.stack([az, el], -1)
    diffuse_direction_samples = torch.from_numpy(
        diffuse_direction_samples.astype(np.float32)
    ).cuda()  # [dn0,2]

    az, el = torch.split(diffuse_direction_samples, 1, dim=1)  # sn,1
    el, az = el.unsqueeze(0), az.unsqueeze(0)
    az = az * np.pi * 2
    el_sqrt = torch.sqrt(el + 1e-8)
    random_azimuth=True
    if is_train and random_azimuth:
        az = (az + torch.rand(z.shape[0], 1, 1, device=az.device) * np.pi * 2) % (2 * np.pi)
    coeff_z = torch.sqrt(1 - el + 1e-8)
    coeff_x = el_sqrt * torch.cos(az)
    coeff_y = el_sqrt * torch.sin(az)
    directions = (
        coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)
    )  # pn,sn,3
    return directions

def sample_specular_directions(reflections, roughness, is_train, specular_sample_num = 128):
    # roughness [pn,1]
    z = reflections  # pn,3
    x = get_orthogonal_directions(reflections)  # pn,3
    y = torch.cross(z, x, dim=-1)  # pn,3
    roughness = get_squared_roughness(roughness)
    a = roughness  # we assume the predicted roughness is already squared
    az, el = sample_sphere(specular_sample_num, 0)
    az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
    specular_direction_samples = np.stack([az, el], -1)
    specular_direction_samples = torch.from_numpy(
        specular_direction_samples.astype(np.float32)
    ).cuda()  # [dn1,2]
    az, el = torch.split(specular_direction_samples, 1, dim=1)  # sn,1
    phi = np.pi * 2 * az  # sn,1     # phi is actually the original az?
    a, el = a.unsqueeze(1), el.unsqueeze(0)  # [pn,1,1] [1,sn,1]
    cos_theta = torch.sqrt(
        (1.0 - el + 1e-8) / (1.0 + (a**2 - 1.0) * el + 1e-8) + 1e-8
    )  # pn,sn,1
    sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-8)  # pn,sn,1
    phi = phi.unsqueeze(0)  # 1,sn,1
    random_azimuth = True
    if is_train and random_azimuth:
        phi = (phi + torch.rand(z.shape[0], 1, 1, device=phi.device) * np.pi * 2) % (2 * np.pi)
    coeff_x = torch.cos(phi) * sin_theta  # pn,sn,1
    coeff_y = torch.sin(phi) * sin_theta  # pn,sn,1
    coeff_z = cos_theta  # pn,sn,1
    # convert from local coordinate -> world coordinate
    directions = (
        coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)
    )  # pn,sn,3
    return directions

def fresnel_schlick(F0, HoV):
    FMi = ((-5.55473) * HoV - 6.98316) * HoV
    return F0+ (1.0 - F0) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    #return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0) ** 5.0

def fresnel_schlick_directions(F0, view_dirs, directions):
    H = view_dirs + directions  # [pn,sn,3]
    H = F.normalize(H, dim=-1)
    HoV = torch.clamp(
        torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0
    )  # [pn,sn,1]
    fresnel = fresnel_schlick(F0, HoV)  # [pn,sn,1]
    return fresnel, H, HoV

def geometry_schlick_ggx(NoV, roughness):
    # a = roughness  # a = roughness**2: we assume the predicted roughness is already squared
    roughness = get_squared_roughness(roughness)
    correct_schlick = False
    if correct_schlick:
        k = (roughness + 1) ** 2 / 8
    else:
        k = roughness / 2

    num = NoV
    denom = NoV * (1 - k) + k
    return num / (denom + 1e-8)

def geometry_schlick(NoV, NoL, roughness):
    ggx2 = geometry_schlick_ggx(NoV, roughness)
    ggx1 = geometry_schlick_ggx(NoL, roughness)
    return ggx2 * ggx1

def geometry_ggx_smith_correlated(NoV, NoL, roughness):
    def fun(alpha2, cos_theta):
        # cos_theta = torch.clamp(cos_theta,min=1e-8,max=1-1e-8)
        cos_theta2 = cos_theta**2
        tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-8)
        return 0.5 * torch.sqrt(1 + alpha2 * tan_theta2) - 0.5

    # todo: is this roughness squared?
    alpha_sq = roughness**2
    return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))

def geometry_func(NoV, NoL, roughness):
    geometry = geometry_schlick(NoV, NoL, roughness)
    return geometry