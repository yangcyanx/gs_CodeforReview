import torch
import numpy as np
import nvdiffrast.torch as dr
from .general_utils import safe_normalize, flip_align_view
from utils.sh_utils import eval_sh
import kornia

env_rayd1 = None
FG_LUT = torch.from_numpy(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()


pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def sample_camera_rays_unnormalize(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def reflection(w_o, normal):
    NdotV = torch.sum(w_o*normal, dim=-1, keepdim=True)
    w_k = 2*normal*NdotV - w_o
    return w_k, NdotV


def saturate_dot(v0, v1):
    return torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), min=0.0, max=1.0)

def get_specular_color_surfel_mode(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, metallic = None, roughness = None, pc=None, surf_depth=None, feature_dr=None, isIndir=False): #RT W2C
    global FG_LUT
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    mask = (render_alpha>0)[..., 0]

    ref_roughness = pc.sph_enc(rays_refl[mask], roughness[mask])
    feature_dr = feature_dr[mask].view(mask.sum(), -1)
    #feature_dr = torch.concat([local_light, feature_dr], dim=-1)
    # 外积生成中间张量
    K_S = torch.einsum('...d,...c->...dc', feature_dr, ref_roughness).flatten(start_dim=-2)
    #K_S = torch.concat([local_light, ref_roughness, feature], dim=-1)
    direct_light = torch.zeros_like(normal_map)
    direct_light[mask] = pc.forward_shader_mlp(K_S)

    # Query BSDF
    #fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    #fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    ## Compute direct light
    #direct_light = envmap(rays_refl, roughness=roughness)
    #specular_weight = ((0.04 * (1 - metallic) + albedo * metallic) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    
    # visibility
    visibility = torch.ones_like(render_alpha)
    indirect_light = torch.zeros_like(normal_map)
    if pc.trace is not None and isIndir:
        rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T)
        w_o = safe_normalize(-rays_cam)
        # import pdb;pdb.set_trace() 
        rays_refl, _ = reflection(w_o, normal_map)
        rays_refl = safe_normalize(rays_refl)
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        # import pdb;pdb.set_trace()
        #_, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        #visibility[mask] = (depth >= 10).float().unsqueeze(-1)
        intersections = intersections + rays_refl * 0.15
        trace_outputs = pc.trace(intersections[mask], rays_refl[mask], camera_center=rays_o)
        visibility[mask] = 1 - trace_outputs['alpha'][..., None]
        #visibility[mask] = torch.logical_or(trace_outputs['depth'] == 0, trace_outputs['depth'] >= 10).float()[...,None]
        indirect_light[mask] = trace_outputs['color']
        # indirect light
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light #* render_alpha * specular_weight
    else:
        specular_light = direct_light
    
    # Compute specular color
    specular_raw = specular_light * render_alpha

    #specular = specular_raw * specular_weight
    specular = specular_raw
    
    if isIndir:
        extra_dict = {
            "visibility": visibility.permute(2,0,1),
            "indirect_light": indirect_light.permute(2,0,1),
            "direct_light": direct_light.permute(2,0,1),
            "indirect_color": indirect_color.permute(2,0,1)
        } 
    else:
        extra_dict = None
        
    return specular.permute(2,0,1), extra_dict

def get_specular_color_surfel_rt(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, metallic = None, roughness = None, pc=None, surf_depth=None, indirect_light=None): #RT W2C
    global FG_LUT
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    # Compute direct light
    direct_light = envmap(rays_refl, roughness=roughness)
    specular_weight = ((0.04 * (1 - metallic) + albedo * metallic) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    
    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.trace is not None and indirect_light is not None:
        mask = (render_alpha>0)[..., 0]
        rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T)
        w_o = safe_normalize(-rays_cam)
        # import pdb;pdb.set_trace() 
        rays_refl, _ = reflection(w_o, normal_map)
        rays_refl = safe_normalize(rays_refl)
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        # import pdb;pdb.set_trace()
        #_, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        #visibility[mask] = (depth >= 10).float().unsqueeze(-1)
        trace_outputs = pc.trace(intersections[mask], rays_refl[mask], camera_center=rays_o)
        #visibility[mask] = 1 - trace_outputs['alpha'][..., None]
        visibility[mask] = (trace_outputs['depth'] >= 10).float().unsqueeze(-1)
        # indirect light
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light
    
    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight
    

    if indirect_light is not None:
        extra_dict = {
            "visibility": visibility.permute(2,0,1),
            "indirect_light": indirect_light.permute(2,0,1),
            "direct_light": direct_light.permute(2,0,1),
            "indirect_color": indirect_color.permute(2,0,1)
        } 
    else:
        extra_dict = None
        
    return specular.permute(2,0,1), extra_dict

def get_specular_color_surfel(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, metallic = None, roughness = None, pc=None, surf_depth=None, indirect_light=None): #RT W2C
    global FG_LUT
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    # Compute direct light
    direct_light = envmap(rays_refl, roughness=roughness)
    #direct_light = envmap(rays_refl, mode='pure_env')
    specular_weight = ((0.04 * (1 - metallic) + albedo * metallic) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    
    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None and indirect_light is not None:
        mask = (render_alpha>0)[..., 0]
        rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T)
        w_o = safe_normalize(-rays_cam)
        # import pdb;pdb.set_trace() 
        rays_refl, _ = reflection(w_o, normal_map)
        rays_refl = safe_normalize(rays_refl)
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        # import pdb;pdb.set_trace()
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).float().unsqueeze(-1)
    
        # indirect light
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light
    
    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight
    

    if indirect_light is not None:
        extra_dict = {
            "visibility": visibility.permute(2,0,1),
            "indirect_light": indirect_light.permute(2,0,1),
            "direct_light": direct_light.permute(2,0,1),
            "indirect_color": indirect_color.permute(2,0,1)
        } 
    else:
        extra_dict = None
        
    return specular.permute(2,0,1), extra_dict




def get_specular_color_surfel2(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, metallic = None, roughness = None, pc=None, surf_depth=None): #RT W2C
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    direct_light = envmap(rays_refl)
    specular = direct_light
    
    return specular.permute(2,0,1)




def get_full_color_volume(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, metallic = None, roughness = None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) # 计算BSDF参数
    # fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-metallic) * albedo
    # Compute specular
    specular = envmap(rays_refl, roughness=roughness) * ((0.04 * (1 - metallic) + albedo * metallic) * fg[0][..., 0:1] + fg[0][..., 1:2]) 

    return diffuse, specular




def get_full_color_volume_indirect(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, metallic = None, roughness = None, pc=None, indirect_light=None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None:
        mask = (render_alpha>0).squeeze()
        intersections = xyz
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).unsqueeze(1).float()

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-metallic) * albedo
    # Compute specular
    direct_light = envmap(rays_refl, roughness=roughness) 
    specular_weight = ((0.04 * (1 - metallic) + albedo * metallic) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    specular_light = direct_light * visibility + (1 - visibility) * indirect_light
    specular = specular_light * specular_weight

    extra_dict = {
        "visibility": visibility,
        "direct_light": direct_light,
    }

    return diffuse, specular, extra_dict




def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

      Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
      (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

      Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

      Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1) ** m * 2 ** l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2 ** i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

      This function returns a function that computes the integrated directional
      encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

      Args:
        deg_view: number of spherical harmonics degrees to use.

      Returns:
        A function for evaluating integrated directional encoding.

      Raises:
        ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2 ** (deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = np.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    mat = torch.from_numpy(mat.astype(np.float32)).cuda()
    ml_array = torch.from_numpy(ml_array.astype(np.float32)).cuda()

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.concat([z ** i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.concat([(x + 1j * y) ** m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn