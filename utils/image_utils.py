

import torch
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_ray(img1, img2):
    mse = (((img1 - img2)) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']

    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    
    colormap = colormap.reversed()
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)

def visualize_depth_red(
    depth_np: np.ndarray,
    valid_depth_min: float = 0.0,
    valid_depth_max: float = 200.0,
    near: float = 0.0,
    far: float = 1,
    reverse_colormap: bool = False
) -> np.ndarray:
    """
    优化的深度可视化函数（解决深蓝色异常问题）
    参数:
        depth_np: 原始深度图 (H, W) numpy数组
        valid_depth_min: 有效深度最小值（过滤无效区域）
        valid_depth_max: 有效深度最大值（过滤无效区域）
        near: 近景深度（手动设置，None则自动取有效深度最小值）
        far: 远景深度（手动设置，None则自动取有效深度最大值）
        reverse_colormap: 是否反转Turbo颜色映射（近景亮色）
    返回:
        depth_vis: 可视化图像 (H, W, 3) uint8 numpy数组
    """
    # 1. 过滤无效深度区域（解决无效区域导致的深蓝色）
    valid_mask = (depth_np >= valid_depth_min) & (depth_np <= valid_depth_max)
    if not np.any(valid_mask):
        print("警告：无有效深度区域！")
        return np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
    
    # 2. 计算有效深度的范围（解决归一化不合理问题）
    valid_depths = depth_np[valid_mask]
    auto_near = valid_depths.min() if near is None else near
    auto_far = valid_depths.max() if far is None else far
    
    # 3. 应用对数变换（增强对比度）
    eps = np.finfo(np.float32).eps
    curve_fn = lambda x: -np.log(x + eps)  # 小深度→大值，大深度→小值
    depth_curve = curve_fn(depth_np)
    near_curve = curve_fn(auto_near)
    far_curve = curve_fn(auto_far)
    
    # 4. 归一化到[0, 1]（避免极端值压缩）
    # 处理分母为0的情况（near == far）
    if np.isclose(near_curve, far_curve):
        depth_norm = np.zeros_like(depth_curve)
    else:
        depth_norm = (depth_curve - near_curve) / (far_curve - near_curve)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)  # 限制在有效范围
    
    # 5. 应用颜色映射（解决颜色顺序问题）
    colormap = plt.get_cmap('turbo')
    if reverse_colormap:
        colormap = colormap.reversed()  # 反转后：0→橙色（近景），1→深蓝色（远景）
    depth_vis = colormap(depth_norm)[:, :, :3]  # 提取RGB通道（忽略Alpha）
    
    # 6. 填充无效区域为黑色（0值）
    depth_vis[~valid_mask] = 0.0
    
    # 转换为uint8格式（0-255）
    return (depth_vis * 255).astype(np.uint8)