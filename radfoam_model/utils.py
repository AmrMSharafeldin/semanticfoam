import colorsys
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from matplotlib import cm


def tab20_palette_torch(num_classes, device="cpu"):
    cmap = cm.get_cmap("tab20", num_classes)
    arr = (cmap(np.arange(num_classes))[:, :3]).astype(np.float32)
    return torch.tensor(arr, device=device)


def tab20_palette_np(num_classes):
    cmap = cm.get_cmap("tab20", num_classes)
    return (cmap(np.arange(num_classes))[:, :3]).astype(np.float32)


def id_to_rgb(instance_id):
    R = instance_id % 256
    G = (instance_id // 256) % 256
    B = (instance_id // 65536) % 256
    return [R, G, B]


def palette_from_global_ids_txt(path, device="cpu"):
    with open(path, "r") as f:
        lines = f.readlines()
        global_ids = [int(x.strip()) for x in lines if x.strip()]
    rgb_colors = np.array([id_to_rgb(id_) for id_ in global_ids], dtype=np.uint8)
    rgb_colors_norm = (rgb_colors.astype(np.float32) / 255.0)
    palette_torch = torch.tensor(rgb_colors_norm, dtype=torch.float32, device=device)
    palette_np = rgb_colors_norm
    print(f"Loaded {len(rgb_colors)} colors from {path}")
    return palette_torch, palette_np


def id2rgb(id, max_num_obj=256):
    if id < 0 or id >= max_num_obj:
        raise ValueError(f"ID {id} out of range (0–{max_num_obj-1})")
    if id == 0:
        return np.zeros(3, dtype=np.uint8)
    golden_ratio = 1.6180339887
    h = (id * golden_ratio) % 1.0
    s = 0.5 + (id % 2) * 0.5
    l = 0.5
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def visualize_obj(objects, max_num_obj=256):
    objects_np = np.array(objects, dtype=np.int32)
    rgb_mask = np.zeros((*objects_np.shape, 3), dtype=np.uint8)
    for obj_id in np.unique(objects_np):
        rgb_mask[objects_np == obj_id] = id2rgb(int(obj_id), max_num_obj)
    return rgb_mask


def inverse_softplus(x, beta, scale=1):
    # log(exp(scale*x)-1)/scale
    out = x / scale
    mask = x * beta < 20 * scale
    out[mask] = torch.log(torch.exp(beta * out[mask]) - 1 + 1e-10) / beta
    return out


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(-1, img1.shape[-1]).mean(0, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_expon_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=1_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0
        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


def get_cosine_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=10_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0.0
        lr_cos = lr_final + 0.5 * (lr_init - lr_final) * (
            1
            + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps))
        )
        return lr_cos

    return helper


def tetrahedron_circumcenters(points, tets):
    """
    Compute the circumcenters of an array of tetrahedra.

    Parameters:
    points: torch.Tensor of shape [M, 3]
        The coordinates of the points in 3D space.
    tets: torch.Tensor of shape [N, 4]
        The indices of the points that form each tetrahedron.

    Returns:
    circumcenters: torch.Tensor of shape [N, 3]
        The coordinates of the circumcenters.
    """
    # Ensure input is a floating point tensor for numerical stability
    points = points.float()

    # p1, p2, p3, p4: [N, 3]
    p1 = points[tets[:, 0]]  # [N, 3]
    p2 = points[tets[:, 1]]  # [N, 3]
    p3 = points[tets[:, 2]]  # [N, 3]
    p4 = points[tets[:, 3]]  # [N, 3]

    # Shift coordinate system so that p1 is at the origin for numerical stability
    p2_shifted = p2 - p1  # [N, 3]
    p3_shifted = p3 - p1  # [N, 3]
    p4_shifted = p4 - p1  # [N, 3]

    p2_cross_p3 = torch.cross(p2_shifted, p3_shifted, dim=-1)
    p3_cross_p4 = torch.cross(p3_shifted, p4_shifted, dim=-1)
    p4_cross_p2 = torch.cross(p4_shifted, p2_shifted, dim=-1)

    volume = (p2_shifted * p3_cross_p4).sum(dim=-1, keepdim=True) / 6
    numerator = (
        (p2_shifted * p2_shifted).sum(dim=-1, keepdim=True) * p3_cross_p4
        + (p3_shifted * p3_shifted).sum(dim=-1, keepdim=True) * p4_cross_p2
        + (p4_shifted * p4_shifted).sum(dim=-1, keepdim=True) * p2_cross_p3
    )
    circumcenters = numerator / volume / 12 + p1

    centroid = (p1 + p2 + p3 + p4) / 4
    mask = (volume.abs() < 1e-6).squeeze()
    circumcenters[mask] = centroid[mask]

    return circumcenters





def test_render_psnr_joint(model, test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, device, white_background=True):
    rays = test_data_handler.rays
    points, _, _, _ = model.get_trace_data()
    start_points = model.get_starting_point(
        rays[:, 0, 0].to(device), points, model.aabb_tree
    )

    psnr_list = []
    with torch.no_grad():
        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()[0]
            rgb_batch = rgb_batch_fetcher.next()[0]

            rgba_output, _, _, _, _, _ = model(ray_batch, start_points[i])

            opacity = rgba_output[..., -1:]
            if white_background:
                rgb_output = rgba_output[..., :3] + (1 - opacity)
            else:
                rgb_output = rgba_output[..., :3]

            rgb_output = rgb_output.reshape(*rgb_batch.shape).clamp(0, 1)
            img_psnr = psnr(rgb_output, rgb_batch).mean()
            psnr_list.append(img_psnr)

            torch.cuda.synchronize()

    return (sum(psnr_list) / len(psnr_list)).item()

