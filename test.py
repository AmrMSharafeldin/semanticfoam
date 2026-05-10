import os
import math
import warnings
import colorsys

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import configargparse
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import lpips

from data_loader import DataHandler
from configs import *
from radfoam_model.utils import psnr
from utils import load_checkpoint
import radfoam
from render_video import (
    build_cameras, compute_object_center,
    render_scene, render_object,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()


# ------------------------------------------------------------------
# SSIM
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------

def id2rgb(id, max_num_obj=256):
    if id < 0 or id >= max_num_obj:
        raise ValueError(f"ID {id} out of range")
    if id == 0:
        return np.zeros(3, dtype=np.uint8)
    h = (id * 1.6180339887) % 1.0
    s = 0.5 + (id % 2) * 0.5
    r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def visualize_obj(objects, max_num_obj=256):
    arr = np.array(objects, dtype=np.int32)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for oid in np.unique(arr):
        rgb[arr == oid] = id2rgb(int(oid), max_num_obj)
    return rgb


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


# ------------------------------------------------------------------
# Stage 1 — test render (PSNR / SSIM / LPIPS)
# ------------------------------------------------------------------

def test_render(model, classifier, out_dir, test_data_handler, sam_test_data_handler,
                ray_batch_fetcher, rgb_batch_fetcher, classifier_args):
    rays = test_data_handler.rays
    points, _, _, _ = model.get_trace_data()
    start_points = model.get_starting_point(rays[:, 0, 0].cuda(), points, model.aabb_tree)

    psnr_list, ssim_list, lpips_list = [], [], []
    sam_fetcher = radfoam.BatchFetcher(sam_test_data_handler.rgbs, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()[0]
            rgb_batch = rgb_batch_fetcher.next()[0]

            output, seg_output, _, _, _, _ = model(ray_batch, start_points[i])
            opacity = output[..., -1:]
            rgb_output = (output[..., :3] + (1 - opacity)).reshape(*rgb_batch.shape).clip(0, 1)

            seg_features = seg_output[..., -classifier_args.input_dim:]
            pred_class_list, pred_conf_list = [], []
            for j in range(0, seg_features.shape[0], 32768):
                chunk = seg_features[j:j + 32768]
                probs = classifier(chunk).softmax(dim=-1)
                pred_class_list.append(probs.argmax(dim=-1))
                pred_conf_list.append(probs.max(dim=-1).values)

            pred_class_np = torch.cat(pred_class_list).reshape(*rgb_batch.shape[:2]).cpu().numpy()
            pred_conf_np = torch.cat(pred_conf_list).reshape(*rgb_batch.shape[:2]).cpu().numpy()
            sam_gt = sam_fetcher.next()[0].squeeze(-1).cpu().numpy()

            view_dir = os.path.join(out_dir, "test", f"view_{i:03d}")
            os.makedirs(view_dir, exist_ok=True)

            if i == 0:
                mask_dir = os.path.join(view_dir, "binary_masks")
                os.makedirs(mask_dir, exist_ok=True)
                for cls_id in np.unique(pred_class_np):
                    Image.fromarray((pred_class_np == cls_id).astype(np.uint8) * 255).save(
                        os.path.join(mask_dir, f"class_{cls_id}.png")
                    )
                Image.fromarray(pred_class_np.astype(np.uint8)).save(
                    os.path.join(mask_dir, "pred_multiclass.png")
                )

            save_rgb_png(rgb_batch.cpu().numpy(), os.path.join(view_dir, "gt_rgb.png"))
            save_rgb_png(rgb_output.cpu().numpy(), os.path.join(view_dir, "pred_rgb.png"))
            save_rgb_png((rgb_output - rgb_batch).abs().cpu().numpy(), os.path.join(view_dir, "error_map.png"))
            save_rgb_png(visualize_obj(sam_gt, classifier_args.num_classes) / 255.0,
                         os.path.join(view_dir, "gt_segmentation.png"))
            save_rgb_png(visualize_obj(pred_class_np, classifier_args.num_classes) / 255.0,
                         os.path.join(view_dir, "pred_segmentation.png"))

            cmap = plt.get_cmap("plasma")
            Image.fromarray((cmap(pred_conf_np)[:, :, :3] * 255).astype(np.uint8)).save(
                os.path.join(view_dir, "confidence_map.png")
            )

            psnr_list.append(psnr(rgb_output, rgb_batch).mean().cpu().item())
            rgb_out_t = rgb_output.permute(2, 0, 1).unsqueeze(0)
            rgb_gt_t = rgb_batch.permute(2, 0, 1).unsqueeze(0)
            ssim_list.append(ssim(rgb_out_t, rgb_gt_t).item())
            lpips_list.append(compute_lpips(
                rgb_batch.permute(2, 0, 1).contiguous(),
                rgb_output.permute(2, 0, 1).contiguous(),
            ))

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_lpips = sum(lpips_list) / len(lpips_list)

    with open(os.path.join(out_dir, "test", "metrics.txt"), "w") as f:
        f.write(f"Average PSNR:  {avg_psnr:.4f}\n")
        f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")

    return avg_psnr


# ------------------------------------------------------------------
# Stage 2 — segmentation evaluation
# ------------------------------------------------------------------

def load_masks(mask_root):
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
    rays = data_handler.rays[idx]
    rgb_gt = data_handler.rgbs[idx]
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

    return rgb.permute(2, 0, 1), rgb_gt.permute(2, 0, 1), pred_class.cpu().numpy()


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


def render_set(model, classifier, train_data, test_data, cls_args, dataset_args, outdir):
    mask_root = os.path.join(dataset_args.data_path, dataset_args.scene,
                             "segmentation_labels", "masks")
    if not os.path.isdir(mask_root):
        print("[ERROR] mask folder missing:", mask_root)
        return {}

    all_masks = load_masks(mask_root)
    test_masks = filter_masks_to_test(all_masks, test_data)
    if not test_masks:
        print("[ERROR] no masks matched test-set")
        return {}

    train_data.reload("train", downsample=min(dataset_args.downsample))
    test_data.reload("test", downsample=min(dataset_args.downsample))

    points, _, _, _ = model.get_trace_data()
    model.update_segmentation_indices(classifier, cls_args)
    model.set_render_mode("rgb")

    prompt_frame = sorted(test_masks.keys())[0]
    prompt_idx = _find_in_dataset(prompt_frame, test_data)
    if prompt_idx is None:
        print("[ERROR] prompt image not in test-set")
        return {}

    print(f"[PROMPT] using frame {prompt_frame}")
    rgb_p, _, pred_prompt = forward_predict(model, classifier, test_data, prompt_idx, cls_args, points)
    obj_map = map_objects_to_classes(pred_prompt, test_masks[prompt_frame])

    root_save = os.path.join(outdir, "segmentation_eval/test", prompt_frame)
    os.makedirs(root_save, exist_ok=True)
    save_rgb_png(rgb_p.permute(1, 2, 0).cpu().numpy(),
                 os.path.join(root_save, f"{prompt_frame}_render_full.png"))

    obj_ious = {k: [] for k in obj_map}
    obj_accs = {k: [] for k in obj_map}

    for frame, labels in test_masks.items():
        idx = _find_in_dataset(frame, test_data)
        if idx is None:
            print("[SKIP] no rgb for", frame)
            continue

        rgb_t, gt_t, pred = forward_predict(model, classifier, test_data, idx, cls_args, points)
        obj_mask_global = (pred != 0)

        frame_dir = os.path.join(outdir, "segmentation_eval/test", frame)
        full_dir, mask_dir, obj_dir = (os.path.join(frame_dir, s)
                                       for s in ["full", "masks", "objects"])
        for d in [full_dir, mask_dir, obj_dir]:
            os.makedirs(d, exist_ok=True)

        save_rgb_png(gt_t.permute(1, 2, 0).cpu().numpy(), os.path.join(full_dir, "gt_full.png"))
        save_rgb_png(rgb_t.permute(1, 2, 0).cpu().numpy(), os.path.join(full_dir, "render_full.png"))

        device = rgb_t.device
        gt_t, rgb_t = gt_t.to(device), rgb_t.to(device)

        H, W = pred.shape
        for oid, gt_mask in labels.items():
            if oid not in obj_map or not obj_map[oid]:
                continue

            cls_ids = obj_map[oid]
            gt_resized = cv2.resize(gt_mask.astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            pred_mask = np.isin(pred, cls_ids)
            gt_resized = gt_resized & obj_mask_global
            pred_mask = pred_mask & obj_mask_global

            save_mask_png(gt_resized, os.path.join(mask_dir, f"{oid}_gt_mask.png"))
            save_mask_png(pred_mask, os.path.join(mask_dir, f"{oid}_pred_mask.png"))

            iou = compute_iou(gt_resized, pred_mask)
            acc = compute_acc(gt_resized, pred_mask)
            obj_ious[oid].append(iou)
            obj_accs[oid].append(acc)
            print(f"[EVAL] {frame} obj {oid} → IoU={iou:.3f}, Acc={acc:.3f}")

            save_rgb_with_white_bg(gt_t, gt_resized & obj_mask_global,
                                   os.path.join(obj_dir, f"{oid}_gt_object.png"))
            save_rgb_with_white_bg(rgb_t, pred_mask & obj_mask_global,
                                   os.path.join(obj_dir, f"{oid}_pred_object.png"))

    avg_iou = {k: float(np.mean(v)) for k, v in obj_ious.items() if v}
    avg_acc = {k: float(np.mean(v)) for k, v in obj_accs.items() if v}
    mean_iou = float(np.mean(list(avg_iou.values()))) if avg_iou else 0.0
    mean_acc = float(np.mean(list(avg_acc.values()))) if avg_acc else 0.0

    scene = dataset_args.scene
    metrics_dir = os.path.join(outdir, "metrics_summary")
    os.makedirs(metrics_dir, exist_ok=True)
    fpath = os.path.join(metrics_dir, f"{scene}_metrics.txt")
    with open(fpath, "w") as f:
        f.write(f"Scene: {scene}\n")
        for k in avg_iou:
            f.write(f"{k}: IoU={avg_iou[k]:.4f}, Acc={avg_acc[k]:.4f}\n")
        f.write(f"\nMean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Acc: {mean_acc:.4f}\n")

    global_csv = os.path.join(metrics_dir, "all_scenes_summary.csv")
    write_header = not os.path.exists(global_csv)
    with open(global_csv, "a") as f:
        if write_header:
            f.write("scene,mean_iou,mean_acc\n")
        f.write(f"{scene},{mean_iou:.4f},{mean_acc:.4f}\n")

    print("\n=== FINAL METRICS ===")
    print("Avg IoU:", avg_iou)
    print("Avg Acc:", avg_acc)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Acc: {mean_acc:.4f}")
    print("=====================\n")

    return {"avg_iou": avg_iou, "avg_acc": avg_acc,
            "mean_iou": mean_iou, "mean_acc": mean_acc,
            "obj_map": obj_map,
            "metrics_path": fpath}


# ------------------------------------------------------------------
# Object extraction
# ------------------------------------------------------------------

def extract_objects(model, classifier, cls_args, obj_map, outdir, conf_thresh=0.9):
    """Classify all scene points and save per-object point assets."""
    objects_dir = os.path.join(outdir, "objects")
    os.makedirs(objects_dir, exist_ok=True)

    _, attributes, _, _ = model.get_trace_data()
    seg_feats = attributes[..., -cls_args.input_dim:]

    with torch.no_grad():
        probs = torch.softmax(classifier(seg_feats), dim=-1)
        pred_classes = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values

    for oid, cls_ids in obj_map.items():
        cls_tensor = torch.tensor(cls_ids, device=pred_classes.device)
        mask = torch.isin(pred_classes, cls_tensor) & (conf >= conf_thresh)
        if not mask.any():
            print(f"[EXTRACT] No points for object {oid} (conf_thresh={conf_thresh}), skipping")
            continue
        save_base = os.path.join(objects_dir,
                                 f"{oid}_class{'_'.join(map(str, cls_ids))}")
        model.save_asset(mask, save_path_base=save_base)
        print(f"[EXTRACT] {oid} → {mask.sum().item()} pts (conf≥{conf_thresh}) → {save_base}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--eval_segmentation", action="store_true",
                        help="Run segmentation evaluation (IoU/Acc) and extract object assets")
    parser.add_argument("--trajectory_type", default=None,
                        choices=["360", "firstcam", "spiral"],
                        help="Render trajectory videos (RGB + seg + per-object)")

    parser.add_argument("--conf_thresh", type=float, default=0.9,
                        help="Minimum softmax confidence to include a point in object extraction")
    # trajectory params
    parser.add_argument("--fps",          type=int,   default=30)
    parser.add_argument("--n_frames",     type=int,   default=200)
    parser.add_argument("--radius",       type=float, default=3.5)
    parser.add_argument("--fov",          type=float, default=0.7)
    parser.add_argument("--height",       type=float, default=0.8)
    parser.add_argument("--forward_push", type=float, default=0.0)
    parser.add_argument("--n_rots",       type=int,   default=2)

    model_params        = ModelParams(parser)
    pipeline_params     = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params      = DatasetParams(parser)
    classifier_params   = ClassifierParams(parser)

    args           = parser.parse_args()
    device         = torch.device("cuda")
    model_args     = model_params.extract(args)
    dataset_args   = dataset_params.extract(args)
    classifier_args = classifier_params.extract(args)
    pipeline_args  = pipeline_params.extract(args)
    optimizer_args = optimization_params.extract(args)

    test_dir       = os.path.dirname(args.config)
    checkpoint_dir = os.path.join(test_dir, "checkpoints")
    os.makedirs(os.path.join(test_dir, "test"), exist_ok=True)

    model, classifier = load_checkpoint(
        model_args, classifier_args, optimizer_args, pipeline_args,
        device, checkpoint_dir, pipeline_args.chpt_iter, eval=True
    )

    test_data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    sam_dataset_args = dataset_args
    sam_dataset_args.dataset = dataset_args.dataset.replace("colmap", "segment")
    sam_test_handler = DataHandler(sam_dataset_args, rays_per_batch=0, device=device)
    sam_test_handler.reload(split="test", downsample=min(dataset_args.downsample))

    ray_fetcher = radfoam.BatchFetcher(test_data_handler.rays, batch_size=1, shuffle=False)
    rgb_fetcher = radfoam.BatchFetcher(test_data_handler.rgbs, batch_size=1, shuffle=False)

    #---- Stage 1: test render ----
    print("\n[STAGE 1] TEST RENDER")
    avg_psnr = test_render(
        model, classifier, test_dir,
        test_data_handler, sam_test_handler,
        ray_fetcher, rgb_fetcher, classifier_args,
    )
    print(f"Test render complete — PSNR: {avg_psnr:.4f} dB\n")

    # ---- Stage 2: segmentation eval + Stage 3: object extraction ----
    obj_map = {}
    if args.eval_segmentation:
        print("[STAGE 2] SEGMENTATION EVALUATION")
        dataset_args.dataset = dataset_args.dataset.replace("segment", "colmap")
        results = render_set(
            model=model,
            classifier=classifier,
            train_data=DataHandler(dataset_args, rays_per_batch=0, device=device),
            test_data=test_data_handler,
            cls_args=classifier_args,
            dataset_args=dataset_args,
            outdir=test_dir,
        )
        obj_map = results.get("obj_map", {}) if results else {}
        if obj_map:
            print("[STAGE 3] OBJECT EXTRACTION")
            extract_objects(model, classifier, classifier_args, obj_map, test_dir,
                            conf_thresh=args.conf_thresh)

    # ---- Stage 4: trajectory videos ----
    if args.trajectory_type:
        ttype = args.trajectory_type
        print(f"\n[STAGE 4] RENDER VIDEOS — trajectory: {ttype}")

        # classify all points once for object centering
        _, feats, _, _ = model.get_trace_data()
        with torch.no_grad():
            seg_all = classifier(feats[..., -classifier_args.input_dim:]).argmax(dim=-1)

        if pipeline_args.video_target_class:
            scene_target = compute_object_center(model, seg_all, pipeline_args.video_target_class)
        else:
            scene_target = model.primal_points.mean(dim=0)

        scene_cams = build_cameras(ttype, model, scene_target, test_data_handler, args)

        video_dir  = os.path.join(test_dir, "videos")
        scene_name = f"scene_{ttype}"
        scene_out  = os.path.join(video_dir, scene_name)
        os.makedirs(scene_out, exist_ok=True)

        print(f"  → full scene RGB + seg")
        render_scene(model, classifier, classifier_args,
                     scene_cams, scene_out, scene_name, args.fps)

        # per-object isolated videos using the same trajectory type
        for oid, cls_ids in obj_map.items():
            obj_center = compute_object_center(model, seg_all, cls_ids)
            obj_cams   = build_cameras(ttype, model, obj_center, test_data_handler, args)
            obj_name   = f"obj_{oid}"
            obj_out    = os.path.join(video_dir, obj_name)
            os.makedirs(obj_out, exist_ok=True)
            print(f"  → object '{oid}' classes {cls_ids}")
            render_object(model, classifier, classifier_args,
                          obj_cams, cls_ids, obj_out, obj_name, args.fps)


if __name__ == "__main__":
    main()
