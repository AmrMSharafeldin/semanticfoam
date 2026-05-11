import os
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import lpips

from radfoam_model.utils import id2rgb, visualize_obj

lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(_2d.expand(channel, 1, window_size, window_size).contiguous())


def ssim(img1, img2, window_size=11):
    """Structural similarity index between two [C, H, W] image tensors."""
    channel = img1.size(-3)
    window = _create_window(window_size, channel).type_as(img1)
    if img1.is_cuda:
        window = window.cuda(img1.get_device(), non_blocking=True)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1_mu2 + C1) * (2 * s12 + C2)) /
            ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))).mean()


def save_mask_png(mask_bool, path):
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


def save_rgb_png(rgb_float01, path):
    Image.fromarray(np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_rgb_with_white_bg(rgb_tensor, mask_bool, path):
    H, W = mask_bool.shape
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    out = np.ones((H, W, 3), dtype=np.uint8) * 255
    out[mask_bool] = (rgb * 255).astype(np.uint8)[mask_bool]
    Image.fromarray(out).save(path)


def compute_iou(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_acc(gt, pred):
    if np.sum(gt) == 0:
        return 1.0
    return float(np.logical_and(gt, pred).sum()) / float(np.sum(gt))


def compute_lpips(gt_t, pred_t):
    gt = (gt_t * 2 - 1).unsqueeze(0)
    pred = (pred_t * 2 - 1).unsqueeze(0)
    return float(lpips_loss_fn(gt, pred).item())


def load_masks(mask_root):
    """Load binary PNG masks from a directory tree of {obj_id}/{frame}.png files."""
    frame_dict = {}
    for obj_id in sorted(d for d in os.listdir(mask_root)
                         if os.path.isdir(os.path.join(mask_root, d))):
        obj_dir = os.path.join(mask_root, obj_id)
        for fname in sorted(os.listdir(obj_dir)):
            if not fname.lower().endswith(".png"):
                continue
            frame = os.path.splitext(fname)[0]
            mask = cv2.imread(os.path.join(obj_dir, fname), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                frame_dict.setdefault(frame, {})[obj_id] = mask > 128
    return frame_dict


def filter_masks_to_test(all_masks, test_data):
    basenames = {os.path.splitext(f)[0] for f in test_data.image_names}
    filt = {f: m for f, m in all_masks.items() if f in basenames}
    print(f"[INFO] Loaded {len(all_masks)} frames, matched {len(filt)} to test-set")
    return filt


def _find_in_dataset(frame, data_handler):
    for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
        cand = f"{frame}{ext}"
        if cand in data_handler.image_names:
            return data_handler.image_names.index(cand)
    return None


def forward_predict(model, classifier, data_handler, idx, cls_args, points):
    """Run inference on one image; returns (rgb_CHW, gt_CHW, pred_class_HW_np)."""
    rays = data_handler.rays[idx]
    H, W, _ = rays.shape
    flat = rays.reshape(-1, 6).cuda()
    start = model.get_starting_point(flat[:, :3], points, model.aabb_tree)

    with torch.no_grad():
        out, seg_out, *_ = model(flat, start)
        opacity = out[..., -1:]
        rgb = (out[..., :3] + (1 - opacity)).reshape(H, W, 3).clamp(0, 1)
        seg_feats = seg_out[..., -cls_args.input_dim:]
        preds = []
        for j in range(0, seg_feats.shape[0], 32768):
            probs = torch.softmax(classifier(seg_feats[j:j + 32768]), -1)
            preds.append(probs.argmax(-1))
        pred_class = torch.cat(preds).reshape(H, W)

    return rgb.permute(2, 0, 1), data_handler.rgbs[idx].permute(2, 0, 1), pred_class.cpu().numpy()


def map_objects_to_classes(pred_prompt, prompt_labels, thr=0.5):
    H, W = pred_prompt.shape
    uniq, counts = np.unique(pred_prompt, return_counts=True)
    total = dict(zip(uniq, counts))
    obj_map = {}
    for oid, mask in prompt_labels.items():
        resized = cv2.resize(mask.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        vals, cts = np.unique(pred_prompt[resized], return_counts=True)
        pred_counts = dict(zip(vals, cts))
        dom = [int(cls) for cls, cnt in pred_counts.items()
               if cnt / total.get(cls, 1) > thr]
        if not dom:
            dom = [max(pred_counts, key=pred_counts.get)]
        obj_map[oid] = dom
        print(f"[PROMPT] object {oid} → {dom}")
    return obj_map
