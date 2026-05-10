import colorsys
import torch
import numpy as np
from matplotlib import cm

def tab20_palette_torch(num_classes, device="cpu"):
    cmap = cm.get_cmap("tab20", num_classes)
    arr = (cmap(np.arange(num_classes))[:, :3]).astype(np.float32)  # (num_classes, 3)
    return torch.tensor(arr, device=device)

def tab20_palette_np(num_classes):
    cmap = cm.get_cmap("tab20", num_classes)
    return (cmap(np.arange(num_classes))[:, :3]).astype(np.float32)


import numpy as np
import torch

def id_to_rgb(instance_id):
    """Decode the integer ID back to RGB color."""
    R = instance_id % 256
    G = (instance_id // 256) % 256
    B = (instance_id // 65536) % 256
    return [R, G, B]


def palette_from_global_ids_txt(path, device="cpu"):
    """
    Read global_ids.txt and build a color palette (torch + numpy versions).
    Each line in the file should contain a single integer ID.
    """
    # Load integer IDs
    with open(path, "r") as f:
        lines = f.readlines()
        global_ids = [int(x.strip()) for x in lines if x.strip()]

    # Convert each encoded ID to RGB
    rgb_colors = np.array([id_to_rgb(id_) for id_ in global_ids], dtype=np.uint8)

    # Normalize for PyTorch (0–1 float) version
    rgb_colors_norm = (rgb_colors.astype(np.float32) / 255.0)

    # Return both torch and numpy palettes
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
