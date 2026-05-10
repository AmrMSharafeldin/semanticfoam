# # import os
# # import numpy as np
# # import torch
# # import torch.nn.functional as F
# # import configargparse
# # import warnings
# # import cv2
# # import colorsys
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # from plyfile import PlyData
# # from torch import nn

# # from data_loader import DataHandler
# # from configs import *
# # from radfoam_model.scene import RadFoamScene
# # from radfoam_model.utils import psnr
# # from utils import load_checkpoint
# # import radfoam

# # warnings.filterwarnings("ignore")
# # torch.manual_seed(42)
# # np.random.seed(42)


# # # Utility functions

# # def id2rgb(id, max_num_obj=256):
# #     if id < 0 or id >= max_num_obj:
# #         raise ValueError(f"ID {id} out of range (0–{max_num_obj-1})")
# #     if id == 0:
# #         return np.zeros(3, dtype=np.uint8)
# #     golden_ratio = 1.6180339887
# #     h = (id * golden_ratio) % 1.0
# #     s = 0.5 + (id % 2) * 0.5
# #     l = 0.5
# #     r, g, b = colorsys.hls_to_rgb(h, l, s)
# #     return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


# # def visualize_obj(objects, max_num_obj=256):
# #     objects_np = np.array(objects, dtype=np.int32)
# #     rgb_mask = np.zeros((*objects_np.shape, 3), dtype=np.uint8)
# #     for obj_id in np.unique(objects_np):
# #         rgb_mask[objects_np == obj_id] = id2rgb(int(obj_id), max_num_obj)
# #     return rgb_mask


# # def save_mask_png(mask_bool, path):
# #     Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


# # def save_rgb_png(rgb_float01, path):
# #     Image.fromarray(np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)).save(path)


# # # Stage 1 — original test rendering loop (kept intact)

# # def test_render(model, classifier, out_dir, test_data_handler, sam_test_data_handler,
# #                 ray_batch_fetcher, rgb_batch_fetcher, classifier_args, debug=False):
# #     rays = test_data_handler.rays
# #     points, _, _, _ = model.get_trace_data()
# #     start_points = model.get_starting_point(rays[:, 0, 0].cuda(), points, model.aabb_tree)
# #     psnr_list = []
# #     sam_test_fetcher = radfoam.BatchFetcher(sam_test_data_handler.rgbs, batch_size=1, shuffle=False)

# #     with torch.no_grad():
# #         for i in range(rays.shape[0]):
# #             # fetch test batch
# #             ray_batch = ray_batch_fetcher.next()[0]
# #             rgb_batch = rgb_batch_fetcher.next()[0]

# #             # forward pass
# #             output, seg_output, _, _, _, _ = model(ray_batch, start_points[i])
# #             opacity = output[..., -1:]
# #             rgb_output = output[..., :3] + (1 - opacity)
# #             rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

# #             # segmentation features
# #             seg_features = seg_output[..., -classifier_args.input_dim:]
# #             pred_class_list, pred_logits_list = [], []
# #             for j in range(0, seg_features.shape[0], 32768):
# #                 chunk = seg_features[j:j+32768]
# #                 logits_chunk = classifier(chunk)
# #                 probs_chunk = logits_chunk.softmax(dim=-1)
# #                 pred_class_list.append(probs_chunk.argmax(dim=-1))
# #                 pred_logits_list.append(probs_chunk.max(dim=-1).values)

# #             pred_class = torch.cat(pred_class_list, dim=0)
# #             pred_conf = torch.cat(pred_logits_list, dim=0)

# #             pred_class_np = pred_class.reshape(*rgb_batch.shape[:2]).cpu().numpy()
# #             pred_conf_np = pred_conf.reshape(*rgb_batch.shape[:2]).cpu().numpy()
# #             sam_gt = sam_test_fetcher.next()[0].squeeze(-1).cpu().numpy()

# #             pred_seg_rgb = visualize_obj(pred_class_np, classifier_args.num_classes)
# #             gt_seg_rgb = visualize_obj(sam_gt, classifier_args.num_classes)

# #             rgb_output_np = np.uint8(rgb_output.cpu() * 255)
# #             rgb_batch_np = np.uint8(rgb_batch.cpu() * 255)
# #             error = np.uint8((rgb_output - rgb_batch).abs().cpu() * 255)

# #             img_psnr = psnr(rgb_output, rgb_batch).mean().cpu().item()
# #             psnr_list.append(img_psnr)

# #             # visualization
# #             fig, axs = plt.subplots(2, 3, figsize=(18, 8))
# #             axs[0,0].imshow(rgb_batch_np); axs[0,0].set_title("GT RGB")
# #             axs[0,1].imshow(rgb_output_np); axs[0,1].set_title("Predicted RGB")
# #             axs[0,2].imshow(error); axs[0,2].set_title("Error Map")
# #             axs[1,0].imshow(gt_seg_rgb); axs[1,0].set_title("GT Segmentation")
# #             axs[1,1].imshow(pred_seg_rgb); axs[1,1].set_title("Predicted Segmentation")
# #             conf_map = axs[1,2].imshow(pred_conf_np, cmap='viridis', vmin=0, vmax=1)
# #             axs[1,2].set_title("Confidence Map")
# #             fig.colorbar(conf_map, ax=axs[1,2], fraction=0.046, pad=0.04)
# #             for ax_row in axs:
# #                 for ax in ax_row:
# #                     ax.axis("off")

# #             fig.suptitle(f"Test Frame {i} — PSNR: {img_psnr:.2f}", fontsize=16)
# #             fig.tight_layout()

# #             os.makedirs(f"{out_dir}/test", exist_ok=True)
# #             fig.savefig(f"{out_dir}/test/frame_{i:03d}_grid.png", dpi=150)
# #             plt.close(fig)

# #     average_psnr = sum(psnr_list) / len(psnr_list)
# #     with open(f"{out_dir}/test/metrics.txt", "w") as f:
# #         f.write(f"Average PSNR: {average_psnr:.4f}\n")
# #     return average_psnr


# # # Stage 2 — full multi-frame IoU evaluation and asset export

# # def get_all_prompts_and_masks(object_masks_dir):
# #     anno_all = {}
# #     frame_list = sorted([
# #         d for d in os.listdir(object_masks_dir)
# #         if os.path.isdir(os.path.join(object_masks_dir, d))
# #     ])
# #     for frame_name in frame_list:
# #         frame_dir = os.path.join(object_masks_dir, frame_name)
# #         anno = {}
# #         for seg_name in sorted(os.listdir(frame_dir)):
# #             seg_file = os.path.join(frame_dir, seg_name)
# #             label = os.path.splitext(seg_name)[0]
# #             seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
# #             if seg is None:
# #                 print(f"[Warning] Could not read {seg_file}")
# #                 continue
# #             seg = seg > 128
# #             anno[label] = seg
# #         anno_all[frame_name] = anno
# #     print("Frames discovered:", list(anno_all.keys()))
# #     for f, labels in anno_all.items():
# #         print(f"Frame {f} has labels:", list(labels.keys()))
# #     return anno_all


# # def get_bbox_from_mask_bool(mask_bool):
# #     ys, xs = np.where(mask_bool)
# #     if len(xs) == 0 or len(ys) == 0:
# #         return 0, 0, mask_bool.shape[0], mask_bool.shape[1]
# #     min_y, max_y = ys.min(), ys.max() + 1
# #     min_x, max_x = xs.min(), xs.max() + 1
# #     return min_y, min_x, max_y, max_x


# # def resize_mask_to_pred(gt_mask, target_shape):
# #     H, W = target_shape
# #     gt_resized = cv2.resize(gt_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
# #     return gt_resized.astype(bool)


# # def compute_iou(gt_mask_bool, pred_mask_bool):
# #     inter = np.logical_and(gt_mask_bool, pred_mask_bool).sum()
# #     union = np.logical_or(gt_mask_bool, pred_mask_bool).sum()
# #     return float(inter) / float(union) if union > 0 else 0.0


# # def to_uint8_img(arr_float01):
# #     return np.clip(arr_float01 * 255.0, 0, 255).astype(np.uint8)


# # def forward_predict_classes_for_frame(model, classifier, data_handler, idx, classifier_args, points):
# #     frame_rays = data_handler.rays[idx]
# #     frame_rgb = data_handler.rgbs[idx]
# #     H, W, _ = frame_rays.shape

# #     flat_rays = frame_rays.reshape(-1, 6).cuda()
# #     ray_origins = flat_rays[:, :3]
# #     start_points = model.get_starting_point(ray_origins, points, model.aabb_tree)

# #     with torch.no_grad():
# #         output, seg_output, _, _, _, _ = model(flat_rays, start_points)
# #         opacity = output[..., -1:]
# #         rgb_output = output[..., :3] + (1.0 - opacity)
# #         rgb_output = rgb_output.reshape(H, W, 3).clamp(0, 1)
# #         seg_features = seg_output[..., -classifier_args.input_dim:]

# #         preds = []
# #         chunk = 32768
# #         for j in range(0, seg_features.shape[0], chunk):
# #             feats = seg_features[j:j + chunk]
# #             logits = classifier(feats)
# #             probs = torch.softmax(logits, dim=-1)
# #             preds.append(torch.argmax(probs, dim=-1))
# #         pred_class = torch.cat(preds, dim=0).reshape(H, W)

# #     # out_dir = "dummy/binary_masks"
# #     # os.makedirs(out_dir, exist_ok=True)

# #     pred_class_np = pred_class.cpu().numpy()

# #     # # Save rendered RGB image
# #     # rgb_path = f"{out_dir}/frame_{idx:03d}_rendered.png"
# #     # rgb_img = (rgb_output.cpu().numpy() * 255).astype(np.uint8)
# #     # Image.fromarray(rgb_img).save(rgb_path)

# #     # Save per-class binary masks
# #     # for cls_id in np.unique(pred_class_np):
# #     #     mask = (pred_class_np == cls_id).astype(np.uint8) * 255
# #     #     Image.fromarray(mask).save(f"{out_dir}/frame_{idx:03d}_class{cls_id}.png")

# #     # Save multi-class mask
# #     # Image.fromarray(pred_class_np.astype(np.uint8)).save(f"{out_dir}/frame_{idx:03d}_multi.png")

# #     rgb_t = rgb_output.permute(2, 0, 1).contiguous()
# #     frame_rgb_t = frame_rgb.permute(2, 0, 1)
# #     return rgb_t, frame_rgb_t, pred_class_np


# # def map_objects_to_classes_from_prompt(pred_class_np_prompt, prompt_labels, threshold=0.5):
# #     object_to_label = {}
# #     ious_prompt = {}

# #     unique_classes, counts_total = np.unique(pred_class_np_prompt, return_counts=True)
# #     total_per_class = dict(zip(unique_classes, counts_total))
# #     pred_shape = pred_class_np_prompt.shape

# #     for obj_name, gt_mask in prompt_labels.items():
# #         gt_mask_resized = resize_mask_to_pred(gt_mask, pred_shape)
# #         gt_mask_bool = gt_mask_resized.astype(bool)

# #         masked_pred = pred_class_np_prompt[gt_mask_bool]
# #         mask_counts = dict(zip(*np.unique(masked_pred, return_counts=True)))

# #         dominant_classes = []
# #         for cls_id, count in mask_counts.items():
# #             frac = count / total_per_class.get(cls_id, 1)
# #             if frac > threshold:
# #                 dominant_classes.append(int(cls_id))

# #         if not dominant_classes and mask_counts:
# #             dominant_classes = [max(mask_counts, key=mask_counts.get)]

# #         best_cls = dominant_classes[0]
# #         pred_mask = (pred_class_np_prompt == best_cls)
# #         iou = (np.logical_and(pred_mask, gt_mask_bool).sum() /
# #                np.logical_or(pred_mask, gt_mask_bool).sum())
# #         ious_prompt[obj_name] = iou

# #         object_to_label[obj_name] = dominant_classes
# #         print(f"[prompt] {obj_name} dominant {dominant_classes}  (IoU={iou:.3f})")

# #     return object_to_label, ious_prompt


# # def render_set(model, classifier, train_data_handler, test_data_handler,
# #                encode_features, classifier_args, dataset_args, outdir, debug=False):

# #     object_masks_dir = os.path.join(dataset_args.data_path, dataset_args.scene, "segmentations")
# #     all_prompts = get_all_prompts_and_masks(object_masks_dir)
# #     if not all_prompts:
# #         print("no frames found in segmentations")
# #         return {}

# #     train_data_handler.reload(split="train", downsample=min(dataset_args.downsample))
# #     test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

# #     points, _, _, _ = model.get_trace_data()
# #     model.update_segmentation_indices(classifier, classifier_args)
# #     model.set_render_mode("rgb")

# #     frame_names = list(all_prompts.keys())
# #     prompt_frame_base = frame_names[0]
# #     if dataset_args.data_path.endswith("/lerf"):
# #         prompt_image_name = f"{prompt_frame_base}.jpg"
# #     elif dataset_args.data_path.endswith("/llff"):
# #         prompt_image_name = f"{prompt_frame_base}.png"
# #     else:
# #         prompt_image_name = f"{prompt_frame_base}.JPG"
# #     print(f"Prompt frame: {prompt_image_name}")

# #     if prompt_image_name in test_data_handler.image_names:
# #         prompt_handler = test_data_handler
# #         prompt_idx = test_data_handler.image_names.index(prompt_image_name)
# #         prompt_split = "test"
# #     elif prompt_image_name in train_data_handler.image_names:
# #         prompt_handler = train_data_handler
# #         prompt_idx = train_data_handler.image_names.index(prompt_image_name)
# #         prompt_split = "train"
# #     else:
# #         print(f"{prompt_image_name} not found in dataset")
# #         return {}

# #     rgb_prompt_t, gt_prompt_t, pred_class_np_prompt = forward_predict_classes_for_frame(
# #         model, classifier, prompt_handler, prompt_idx, classifier_args, points)

# #     prompt_labels = all_prompts[prompt_frame_base]
# #     object_to_label, ious_prompt = map_objects_to_classes_from_prompt(pred_class_np_prompt, prompt_labels)

# #     save_root = os.path.join(outdir, f"segmentation_eval/{prompt_split}/{prompt_frame_base}")
# #     os.makedirs(save_root, exist_ok=True)
# #     save_rgb_png(rgb_prompt_t.permute(1, 2, 0).cpu().numpy(), os.path.join(save_root, f"{prompt_frame_base}_render.png"))

# #     for obj_name, gt_mask in prompt_labels.items():
# #         gt_resized = resize_mask_to_pred(gt_mask, pred_class_np_prompt.shape)
# #         class_ids = object_to_label.get(obj_name, [])
# #         pred_mask = np.isin(pred_class_np_prompt, class_ids)
# #         save_mask_png(gt_resized, os.path.join(save_root, f"{prompt_frame_base}_{obj_name}_mask_gt.png"))
# #         save_mask_png(pred_mask, os.path.join(save_root, f"{prompt_frame_base}_{obj_name}_mask_pred.png"))

# #     per_object_ious = {k: [] for k in object_to_label.keys()}
# #     psnrs_full, psnrs_crop = [], []

# #     for frame_base, labels in all_prompts.items():
# #         if dataset_args.data_path.endswith("/lerf"):
# #             image_name = f"{frame_base}.jpg"
# #         else:
# #             image_name = f"{frame_base}.JPG"
# #         if image_name in test_data_handler.image_names:
# #             handler, split = test_data_handler, "test"
# #             idx = test_data_handler.image_names.index(image_name)
# #         elif image_name in train_data_handler.image_names:
# #             handler, split = train_data_handler, "train"
# #             idx = train_data_handler.image_names.index(image_name)
# #         else:
# #             continue

# #         rgb_t, gt_t, pred_class_np = forward_predict_classes_for_frame(
# #             model, classifier, handler, idx, classifier_args, points)

   

# #         frame_out = os.path.join(outdir, f"segmentation_eval/{split}/{frame_base}")
# #         crop_dir = os.path.join(frame_out, "crop")
# #         os.makedirs(frame_out, exist_ok=True)
# #         os.makedirs(crop_dir, exist_ok=True)
# #         save_rgb_png(rgb_t.permute(1, 2, 0).cpu().numpy(), os.path.join(frame_out, f"{frame_base}.png"))

# #         for obj_name, gt_mask in labels.items():
# #             if obj_name not in object_to_label or not object_to_label[obj_name]:
# #                 continue
# #             class_ids = object_to_label[obj_name]
# #             gt_resized = resize_mask_to_pred(gt_mask, pred_class_np.shape)
# #             pred_mask = np.isin(pred_class_np, class_ids)

# #             save_mask_png(gt_resized, os.path.join(frame_out, f"{frame_base}_{obj_name}_mask_gt.png"))
# #             save_mask_png(pred_mask, os.path.join(frame_out, f"{frame_base}_{obj_name}_mask_pred.png"))

# #             iou = compute_iou(gt_resized, pred_mask)
# #             per_object_ious[obj_name].append(iou)
# #             print(f"[eval] {frame_base}, {obj_name} -> classes {class_ids}, iou={iou:.3f}")

# #             rgb_np = rgb_t.permute(1, 2, 0).cpu().numpy()
# #             removed_np = rgb_np.copy()
# #             removed_np[pred_mask] = 0.0
# #             save_rgb_png(removed_np, os.path.join(frame_out, f"{frame_base}_{obj_name}_removed.png"))

# #             mask_bool = gt_resized.astype(bool)
# #             min_y, min_x, max_y, max_x = get_bbox_from_mask_bool(mask_bool)
# #             gt_crop_t = gt_t[:, min_y:max_y, min_x:max_x]
# #             rgb_crop_t = rgb_t[:, min_y:max_y, min_x:max_x]

# #             save_rgb_png(gt_crop_t.permute(1, 2, 0).cpu().numpy(),
# #                          os.path.join(crop_dir, f"{frame_base}_{obj_name}_gt.png"))
# #             save_rgb_png(rgb_crop_t.permute(1, 2, 0).cpu().numpy(),
# #                          os.path.join(crop_dir, f"{frame_base}_{obj_name}.png"))

# #             device = rgb_t.device

# #             # ensure all tensors are on same device
# #             gt_t = gt_t.to(device)
# #             rgb_t = rgb_t.to(device)
# #             pred_mask_t = torch.from_numpy(pred_mask).to(device)

# #             mask = pred_mask_t.bool()

# #             # masked images for visualization
# #             masked_rgb = rgb_t * mask
# #             masked_gt = gt_t * mask
# #             save_rgb_png(masked_rgb.permute(1, 2, 0).cpu().numpy(),
# #                          os.path.join(frame_out, f"{frame_base}_{obj_name}_masked_render.png"))
# #             save_rgb_png(masked_gt.permute(1, 2, 0).cpu().numpy(),
# #                          os.path.join(frame_out, f"{frame_base}_{obj_name}_masked_gt.png"))

# #             # PSNR inside mask
# #             rgb_flat = rgb_t.permute(1, 2, 0)[mask]
# #             gt_flat = gt_t.permute(1, 2, 0)[mask]
# #             if rgb_flat.numel() == 0:
# #                 psnr_full = float("nan")
# #             else:
# #                 psnr_full = psnr(rgb_flat.unsqueeze(0), gt_flat.unsqueeze(0)).mean().double().item()

# #             psnr_crop = psnr(
# #                 rgb_crop_t.unsqueeze(0).to(device),
# #                 gt_crop_t.unsqueeze(0).to(device)
# #             ).mean().double().item()

# #             with open(os.path.join(frame_out, f"{frame_base}_{obj_name}_psnr.txt"), "w") as f:
# #                 f.write(f"psnr_full_masked={psnr_full:.4f}\npsnr_crop={psnr_crop:.4f}\n")

# #             psnrs_full.append(psnr_full)
# #             psnrs_crop.append(psnr_crop)

# #     avg_per_object = {k: float(np.mean(v)) for k, v in per_object_ious.items() if v}
# #     mean_iou_all = float(np.mean(list(avg_per_object.values()))) if avg_per_object else 0.0
# #     mean_psnr_full = np.nanmean(psnrs_full) if psnrs_full else 0.0
# #     mean_psnr_crop = np.nanmean(psnrs_crop) if psnrs_crop else 0.0

# #     scene_name = dataset_args.scene
# #     metrics_dir = os.path.join(outdir, "metrics_summary")
# #     os.makedirs(metrics_dir, exist_ok=True)

# #     scene_metrics_path = os.path.join(metrics_dir, f"{scene_name}_metrics.txt")
# #     with open(scene_metrics_path, "w") as f:
# #         f.write(f"Scene: {scene_name}\n")
# #         f.write(f"Split: train/test mix\n\n")
# #         f.write("Prompt object-to-class mapping\n")
# #         for k, v in object_to_label.items():
# #             f.write(f"{k}: classes {v}, prompt_iou={ious_prompt.get(k, 0.0):.4f}\n")

# #         f.write("\nPer-object mean IoU across views\n")
# #         for k, v in avg_per_object.items():
# #             f.write(f"{k}: {v:.4f}\n")

# #         f.write(f"\nMean IoU across all objects: {mean_iou_all:.4f}\n")
# #         f.write(f"Mean PSNR (masked region only): {mean_psnr_full:.4f} dB\n")
# #         f.write(f"Mean PSNR (crop): {mean_psnr_crop:.4f} dB\n")

# #     global_log_path = os.path.join(metrics_dir, "all_scenes_summary.csv")
# #     header = "scene,mean_iou,mean_psnr_masked,mean_psnr_crop\n"
# #     row = f"{scene_name},{mean_iou_all:.4f},{mean_psnr_full:.4f},{mean_psnr_crop:.4f}\n"
# #     write_header = not os.path.exists(global_log_path)
# #     with open(global_log_path, "a") as f:
# #         if write_header:
# #             f.write(header)
# #         f.write(row)

# #     print("Per-object mean IoU:", avg_per_object)
# #     print(f"Mean IoU: {mean_iou_all:.4f}")
# #     print(f"Mean PSNR full masked: {mean_psnr_full:.4f}")
# #     print(f"Mean PSNR crop: {mean_psnr_crop:.4f}")
# #     print(f"Scene metrics saved to {scene_metrics_path}")
# #     print(f"Global summary updated at {global_log_path}")

# #     return {
# #         "object_to_label": object_to_label,
# #         "prompt_ious": ious_prompt,
# #         "per_object_mean_ious": avg_per_object,
# #         "mean_iou_all": mean_iou_all,
# #         "metrics_path": metrics_dir,
# #         "mean_psnr_full": mean_psnr_full,
# #         "mean_psnr_crop": mean_psnr_crop,
# #     }


# # def main():
# #     parser = configargparse.ArgParser()
# #     parser.add_argument("-c", "--config", is_config_file=True, help="Path to config.yaml")

# #     model_params = ModelParams(parser)
# #     pipeline_params = PipelineParams(parser)
# #     optimization_params = OptimizationParams(parser)
# #     dataset_params = DatasetParams(parser)
# #     classifier_params = ClassifierParams(parser)
# #     args = parser.parse_args()

# #     device = torch.device("cuda")
# #     model_args = model_params.extract(args)
# #     dataset_args = dataset_params.extract(args)
# #     classifier_args = classifier_params.extract(args)
# #     pipeline_args = pipeline_params.extract(args)
# #     optimizer_args = optimization_params.extract(args)

    

# #     test_dir = os.path.dirname(args.config)
# #     checkpoint_dir = os.path.join(test_dir, "checkpoints")
# #     os.makedirs(os.path.join(test_dir, "test"), exist_ok=True)

# #     chpt_iter = pipeline_args.chpt_iter


# #     model, classifier = load_checkpoint(
# #         model_args,
# #         classifier_args,
# #         optimizer_args,
# #         pipeline_args,
# #         device,
# #         checkpoint_dir,
# #         chpt_iter,
# #         eval=True
# #     )


# #     # data
# #     test_data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
# #     test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))
# #     sam_dataset_args = dataset_args
# #     sam_dataset_args.dataset = dataset_args.dataset.replace("colmap", "segment")
# #     sam_test_handler = DataHandler(sam_dataset_args, rays_per_batch=0, device=device)
# #     sam_test_handler.reload(split="test", downsample=min(dataset_args.downsample))
# #     ray_fetcher = radfoam.BatchFetcher(test_data_handler.rays, batch_size=1, shuffle=False)
# #     rgb_fetcher = radfoam.BatchFetcher(test_data_handler.rgbs, batch_size=1, shuffle=False)

# #     # stage 1
# #     print("\nSTAGE 1: TEST RENDER")
# #     psnr_val = test_render(model, classifier, test_dir, test_data_handler,
# #                            sam_test_handler, ray_fetcher, rgb_fetcher, classifier_args)
# #     print(f"Test render complete — PSNR: {psnr_val:.4f} dB\n")

# #     # stage 2
# #     print("STAGE 2: FULL EVALUATION")
# #     dataset_args.dataset = dataset_args.dataset.replace("segment", "colmap")
# #     results = render_set(
# #         model=model,
# #         classifier=classifier,
# #         train_data_handler=DataHandler(dataset_args, rays_per_batch=0, device=device),
# #         test_data_handler=test_data_handler,
# #         encode_features=lambda x: x,
# #         classifier_args=classifier_args,
# #         dataset_args=dataset_args,
# #         outdir=test_dir,
# #         debug=False,
# #     )

# #     # optional assets export based on discovered objects
# #     objects_dir = os.path.join(test_dir, "objects")
# #     os.makedirs(objects_dir, exist_ok=True)

# #     object_masks_dir = os.path.join(dataset_args.data_path, dataset_args.scene, "segmentations")
# #     prompt_masks = get_all_prompts_and_masks(object_masks_dir)
# #     if prompt_masks:
# #         prompt_frame_base = list(prompt_masks.keys())[0]
# #     else:
# #         prompt_frame_base = None

# #     if results and "object_to_label" in results:
# #         for obj_name, class_id in results["object_to_label"].items():
# #             if class_id is None:
# #                 print(f"[WARN] No class ID for {obj_name}, skipping.")
# #                 continue
# #             normalized_name = obj_name.replace(" ", "_")
# #             asset_name = f"{normalized_name}_class{class_id}"
# #             save_base = os.path.join(objects_dir, asset_name)
# #             print(f"Exporting asset for {obj_name} (class {class_id}) -> {save_base}")

# #             points, attributes, _, _ = model.get_trace_data()
# #             seg_features = attributes[..., -classifier_args.input_dim:]

# #             logits_pts = classifier(seg_features)
# #             probs = F.softmax(logits_pts, dim=-1)
# #             pred_classes_pts = probs.argmax(dim=-1)

# #             class_ids = torch.tensor(class_id, device=pred_classes_pts.device)
# #             refined_mask = torch.isin(pred_classes_pts, class_ids)

# #             model.save_asset(refined_mask, save_path_base=save_base)
# #             print(f"Saved asset for {obj_name} (class {class_id}) -> {save_base}")


# # if __name__ == "__main__":
# #     main()


import os
import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warnings
import cv2
import colorsys
import matplotlib.pyplot as plt
from PIL import Image
from plyfile import PlyData
from torch import nn

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
from utils import load_checkpoint
import radfoam

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

import lpips
lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()






import math
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel).pin_memory()

    if img1.is_cuda:
        window = window.cuda(img1.get_device(), non_blocking=True)

    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)



def compute_lpips_full(gt_t, pred_t):
    """
    gt_t, pred_t: tensors shaped [3, H, W], values in [0,1]
    """
    # LPIPS expects BCHW and [-1,1] range
    gt = (gt_t * 2 - 1).unsqueeze(0)
    pred = (pred_t * 2 - 1).unsqueeze(0)
    return float(lpips_loss_fn(gt, pred).item())


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

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


def save_mask_png(mask_bool, path):
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


def save_rgb_png(rgb_float01, path):
    Image.fromarray(np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)).save(path)


def compute_acc(gt_mask_bool, pred_mask_bool):
    """Pixel accuracy = intersection / ground-truth area."""
    if np.sum(gt_mask_bool) == 0:
        return 1.0
    inter = np.logical_and(gt_mask_bool, pred_mask_bool).sum()
    return float(inter) / float(np.sum(gt_mask_bool))


# -------------------------------------------------
# Stage 1 — test rendering (unchanged)
# -------------------------------------------------

def test_render(model, classifier, out_dir, test_data_handler, sam_test_data_handler,
                ray_batch_fetcher, rgb_batch_fetcher, classifier_args, debug=False):
    rays = test_data_handler.rays
    points, _, _, _ = model.get_trace_data()
    start_points = model.get_starting_point(rays[:, 0, 0].cuda(), points, model.aabb_tree)
    psnr_list = []
    ssim_list = []
    lpips_list = []

    
    sam_test_fetcher = radfoam.BatchFetcher(sam_test_data_handler.rgbs, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()[0]
            rgb_batch = rgb_batch_fetcher.next()[0]

            output, seg_output, _, _, _, _ = model(ray_batch, start_points[i])
            opacity = output[..., -1:]
            rgb_output = output[..., :3] + (1 - opacity)
            rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

            seg_features = seg_output[..., -classifier_args.input_dim:]
            pred_class_list, pred_logits_list = [], []
            for j in range(0, seg_features.shape[0], 32768):
                chunk = seg_features[j:j+32768]
                logits_chunk = classifier(chunk)
                probs_chunk = logits_chunk.softmax(dim=-1)
                pred_class_list.append(probs_chunk.argmax(dim=-1))
                pred_logits_list.append(probs_chunk.max(dim=-1).values)

            pred_class = torch.cat(pred_class_list, dim=0)
            pred_conf = torch.cat(pred_logits_list, dim=0)

            pred_class_np = pred_class.reshape(*rgb_batch.shape[:2]).cpu().numpy()

            # ============================================================
            # SAVE BINARY MASKS FOR FIRST VIEW ONLY
            # ============================================================
            if i == 0:
                mask_debug_dir = os.path.join(out_dir, "test", "view_000", "binary_masks")
                os.makedirs(mask_debug_dir, exist_ok=True)

                unique_ids = np.unique(pred_class_np)
                print(f"[DEBUG] Unique predicted class IDs in view 0: {unique_ids}")

                for cls_id in unique_ids:
                    mask = (pred_class_np == cls_id).astype(np.uint8) * 255
                    Image.fromarray(mask).save(os.path.join(mask_debug_dir, f"class_{cls_id}.png"))

                # also save the raw multi-class map
                Image.fromarray(pred_class_np.astype(np.uint8)).save(
                    os.path.join(mask_debug_dir, "pred_multiclass.png")
                )
                print(f"[DEBUG] Saved all binary masks for view 0 → {mask_debug_dir}")

            pred_conf_np = pred_conf.reshape(*rgb_batch.shape[:2]).cpu().numpy()
            sam_gt = sam_test_fetcher.next()[0].squeeze(-1).cpu().numpy()

            pred_seg_rgb = visualize_obj(pred_class_np, classifier_args.num_classes)
            gt_seg_rgb = visualize_obj(sam_gt, classifier_args.num_classes)

            rgb_output_np = np.uint8(rgb_output.cpu() * 255)
            rgb_batch_np = np.uint8(rgb_batch.cpu() * 255)
            error = np.uint8((rgb_output - rgb_batch).abs().cpu() * 255)

            img_psnr = psnr(rgb_output, rgb_batch).mean().cpu().item()
            psnr_list.append(img_psnr)

            view_dir = os.path.join(out_dir, "test", f"view_{i:03d}")
            os.makedirs(view_dir, exist_ok=True)

            save_rgb_png(rgb_batch_np / 255.0, os.path.join(view_dir, "gt_rgb.png"))
            save_rgb_png(rgb_output_np / 255.0, os.path.join(view_dir, "pred_rgb.png"))
            save_rgb_png(error / 255.0, os.path.join(view_dir, "error_map.png"))
            save_rgb_png(gt_seg_rgb / 255.0, os.path.join(view_dir, "gt_segmentation.png"))
            save_rgb_png(pred_seg_rgb / 255.0, os.path.join(view_dir, "pred_segmentation.png"))

            cmap = plt.get_cmap("plasma")
            conf_colored = (cmap(pred_conf_np)[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(conf_colored).save(os.path.join(view_dir, "confidence_map.png"))

            rgb_output_t = rgb_output.permute(2, 0, 1).unsqueeze(0)     # [1,3,H,W]
            rgb_gt_t     = rgb_batch.permute(2, 0, 1).unsqueeze(0)      # [1,3,H,W]
            img_ssim = ssim(rgb_output_t, rgb_gt_t).item()
            ssim_list.append(img_ssim)


            gt_lp    = rgb_batch.permute(2,0,1).contiguous()
            pred_lp  = rgb_output.permute(2,0,1).contiguous()
            img_lpips = compute_lpips_full(gt_lp, pred_lp)

            lpips_list.append(img_lpips)

    average_psnr = sum(psnr_list) / len(psnr_list)
    with open(f"{out_dir}/test/metrics.txt", "w") as f:
        f.write(f"Average PSNR: {average_psnr:.4f}\n")
    

    average_ssim = sum(ssim_list) / len(ssim_list)
    average_lpips = sum(lpips_list) / len(lpips_list)

    with open(f"{out_dir}/test/metrics.txt", "w") as f:
        f.write(f"Average PSNR: {average_psnr:.4f}\n")
        f.write(f"Average SSIM: {average_ssim:.4f}\n")
        f.write(f"Average LPIPS: {average_lpips:.4f}\n")

    
    
    return average_psnr




# -------------------------------------------------
# Stage 2 — full segmentation evaluation
# -------------------------------------------------

def get_all_prompts_and_masks(object_masks_dir):
    anno_all = {}
    frame_list = sorted([
        d for d in os.listdir(object_masks_dir)
        if os.path.isdir(os.path.join(object_masks_dir, d))
    ])
    for frame_name in frame_list:
        frame_dir = os.path.join(object_masks_dir, frame_name)
        anno = {}
        for seg_name in sorted(os.listdir(frame_dir)):
            seg_file = os.path.join(frame_dir, seg_name)
            label = os.path.splitext(seg_name)[0]
            seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
            if seg is None:
                print(f"[Warning] Could not read {seg_file}")
                continue
            seg = seg > 128
            anno[label] = seg
        anno_all[frame_name] = anno
    print("Frames discovered:", list(anno_all.keys()))
    for f, labels in anno_all.items():
        print(f"Frame {f} has labels:", list(labels.keys()))
    return anno_all


def resize_mask_to_pred(gt_mask, target_shape):
    H, W = target_shape
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return gt_resized.astype(bool)


def compute_iou(gt_mask_bool, pred_mask_bool):
    inter = np.logical_and(gt_mask_bool, pred_mask_bool).sum()
    union = np.logical_or(gt_mask_bool, pred_mask_bool).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# -------------------------------------------------
# Stage 2 — render_set modified with accuracy
# -------------------------------------------------


def forward_predict_classes_for_frame(model, classifier, data_handler, idx, classifier_args, points):
    """Render a single frame and return rendered RGB, ground truth RGB, and predicted class map."""
    frame_rays = data_handler.rays[idx]
    frame_rgb = data_handler.rgbs[idx]
    H, W, _ = frame_rays.shape

    flat_rays = frame_rays.reshape(-1, 6).cuda()
    ray_origins = flat_rays[:, :3]
    start_points = model.get_starting_point(ray_origins, points, model.aabb_tree)

    with torch.no_grad():
        output, seg_output, _, _, _, _ = model(flat_rays, start_points)
        opacity = output[..., -1:]
        rgb_output = output[..., :3] + (1.0 - opacity)
        rgb_output = rgb_output.reshape(H, W, 3).clamp(0, 1)

        seg_features = seg_output[..., -classifier_args.input_dim:]
        preds = []
        chunk = 32768
        for j in range(0, seg_features.shape[0], chunk):
            feats = seg_features[j:j + chunk]
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=-1)
            preds.append(torch.argmax(probs, dim=-1))
        pred_class = torch.cat(preds, dim=0).reshape(H, W)

    rgb_t = rgb_output.permute(2, 0, 1).contiguous()
    frame_rgb_t = frame_rgb.permute(2, 0, 1)
    pred_class_np = pred_class.cpu().numpy()

    return rgb_t, frame_rgb_t, pred_class_np


def map_objects_to_classes_from_prompt(pred_class_np_prompt, prompt_labels, threshold=0.5):
    """Maps each prompt object to predicted class IDs and computes IoU."""
    object_to_label = {}
    ious_prompt = {}

    unique_classes, counts_total = np.unique(pred_class_np_prompt, return_counts=True)
    total_per_class = dict(zip(unique_classes, counts_total))
    pred_shape = pred_class_np_prompt.shape

    for obj_name, gt_mask in prompt_labels.items():
        gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask_bool = gt_mask_resized.astype(bool)

        masked_pred = pred_class_np_prompt[gt_mask_bool]
        mask_counts = dict(zip(*np.unique(masked_pred, return_counts=True)))

        dominant_classes = []
        for cls_id, count in mask_counts.items():
            frac = count / total_per_class.get(cls_id, 1)
            if frac > threshold:
                dominant_classes.append(int(cls_id))

        if not dominant_classes and mask_counts:
            dominant_classes = [max(mask_counts, key=mask_counts.get)]

        best_cls = dominant_classes[0]
        pred_mask = (pred_class_np_prompt == best_cls)
        iou = (np.logical_and(pred_mask, gt_mask_bool).sum() /
               np.logical_or(pred_mask, gt_mask_bool).sum())
        ious_prompt[obj_name] = iou

        object_to_label[obj_name] = dominant_classes
        print(f"[prompt] {obj_name} dominant {dominant_classes} (IoU={iou:.3f})")

    return object_to_label, ious_prompt

def render_set(model, classifier, train_data_handler, test_data_handler,
               encode_features, classifier_args, dataset_args, outdir, debug=False):

    object_masks_dir = os.path.join(dataset_args.data_path, dataset_args.scene, "segmentations")
    all_prompts = get_all_prompts_and_masks(object_masks_dir)
    if not all_prompts:
        print("no frames found in segmentations")
        return {}

    train_data_handler.reload(split="train", downsample=min(dataset_args.downsample))
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    points, _, _, _ = model.get_trace_data()
    model.update_segmentation_indices(classifier, classifier_args)
    model.set_render_mode("rgb")

    prompt_frame_base = list(all_prompts.keys())[0]
    prompt_image_name = f"{prompt_frame_base}.png"

    if prompt_image_name in test_data_handler.image_names:
        prompt_handler = test_data_handler
        prompt_idx = test_data_handler.image_names.index(prompt_image_name)
    elif prompt_image_name in train_data_handler.image_names:
        prompt_handler = train_data_handler
        prompt_idx = train_data_handler.image_names.index(prompt_image_name)
    else:
        print(f"{prompt_image_name} not found in dataset")
        return {}

    rgb_prompt_t, gt_prompt_t, pred_class_np_prompt = forward_predict_classes_for_frame(
        model, classifier, prompt_handler, prompt_idx, classifier_args, points
    )

    prompt_labels = all_prompts[prompt_frame_base]
    object_to_label, ious_prompt = map_objects_to_classes_from_prompt(pred_class_np_prompt, prompt_labels)

    save_root = os.path.join(outdir, f"segmentation_eval/test/{prompt_frame_base}")
    os.makedirs(save_root, exist_ok=True)
    save_rgb_png(rgb_prompt_t.permute(1, 2, 0).cpu().numpy(),
                 os.path.join(save_root, f"{prompt_frame_base}_render_full.png"))

    per_object_ious = {k: [] for k in object_to_label.keys()}
    per_object_accs = {k: [] for k in object_to_label.keys()}
    psnrs_masked_crop = []

    for frame_base, labels in all_prompts.items():
        image_name = f"{frame_base}.png"
        if image_name in test_data_handler.image_names:
            handler = test_data_handler
            idx = test_data_handler.image_names.index(image_name)
        elif image_name in train_data_handler.image_names:
            handler = train_data_handler
            idx = train_data_handler.image_names.index(image_name)
        else:
            continue

        rgb_t, gt_t, pred_class_np = forward_predict_classes_for_frame(
            model, classifier, handler, idx, classifier_args, points
        )

        frame_out = os.path.join(outdir, f"segmentation_eval/test/{frame_base}")
        os.makedirs(frame_out, exist_ok=True)
        save_rgb_png(gt_t.permute(1, 2, 0).cpu().numpy(),
                     os.path.join(frame_out, f"{frame_base}_gt_full.png"))
        save_rgb_png(rgb_t.permute(1, 2, 0).cpu().numpy(),
                     os.path.join(frame_out, f"{frame_base}_render_full.png"))

        device = rgb_t.device
        gt_t, rgb_t = gt_t.to(device), rgb_t.to(device)

        for obj_name, gt_mask in labels.items():
            if obj_name not in object_to_label or not object_to_label[obj_name]:
                continue

            class_ids = object_to_label[obj_name]
            gt_resized = resize_mask_to_pred(gt_mask, pred_class_np.shape)
            pred_mask = np.isin(pred_class_np, class_ids)

            save_mask_png(gt_resized, os.path.join(frame_out, f"{frame_base}_{obj_name}_mask_gt.png"))
            save_mask_png(pred_mask, os.path.join(frame_out, f"{frame_base}_{obj_name}_mask_pred.png"))

            iou = compute_iou(gt_resized, pred_mask)
            acc = compute_acc(gt_resized, pred_mask)
            per_object_ious[obj_name].append(iou)
            per_object_accs[obj_name].append(acc)
            print(f"[eval] {frame_base}, {obj_name} -> IoU={iou:.3f}, Acc={acc:.3f}")

            gt_mask_t = torch.from_numpy(gt_resized).to(device).bool()
            pred_mask_t = torch.from_numpy(pred_mask).to(device).bool()

            masked_gt_full = gt_t * gt_mask_t
            masked_rgb_full = rgb_t * pred_mask_t

            save_rgb_png(masked_gt_full.permute(1, 2, 0).cpu().numpy(),
                         os.path.join(frame_out, f"{frame_base}_{obj_name}_gt_object.png"))
            save_rgb_png(masked_rgb_full.permute(1, 2, 0).cpu().numpy(),
                         os.path.join(frame_out, f"{frame_base}_{obj_name}_render_object.png"))

            rgb_flat = rgb_t.permute(1, 2, 0)[gt_mask_t]
            gt_flat = gt_t.permute(1, 2, 0)[gt_mask_t]
            psnr_masked = float("nan") if rgb_flat.numel() == 0 else psnr(
                rgb_flat.unsqueeze(0), gt_flat.unsqueeze(0)
            ).mean().double().item()
            psnrs_masked_crop.append(psnr_masked)

    # === Aggregate results ===
    avg_iou = {k: float(np.mean(v)) for k, v in per_object_ious.items() if v}
    avg_acc = {k: float(np.mean(v)) for k, v in per_object_accs.items() if v}

    mean_iou_all = float(np.mean(list(avg_iou.values()))) if avg_iou else 0.0
    mean_acc_all = float(np.mean(list(avg_acc.values()))) if avg_acc else 0.0
    mean_psnr_masked = np.nanmean(psnrs_masked_crop) if psnrs_masked_crop else 0.0

    scene_name = dataset_args.scene
    metrics_dir = os.path.join(outdir, "metrics_summary")
    os.makedirs(metrics_dir, exist_ok=True)

    scene_metrics_path = os.path.join(metrics_dir, f"{scene_name}_metrics.txt")
    with open(scene_metrics_path, "w") as f:
        f.write(f"Scene: {scene_name}\n\n")
        for k, v in avg_iou.items():
            f.write(f"{k}: IoU={v:.4f}, Acc={avg_acc.get(k, 0):.4f}\n")
        f.write(f"\nMean IoU: {mean_iou_all:.4f}\n")
        f.write(f"Mean Accuracy: {mean_acc_all:.4f}\n")
        f.write(f"Mean PSNR (masked): {mean_psnr_masked:.4f} dB\n")

    global_log_path = os.path.join(metrics_dir, "all_scenes_summary.csv")
    header = "scene,mean_iou,mean_acc,mean_psnr_masked\n"
    row = f"{scene_name},{mean_iou_all:.4f},{mean_acc_all:.4f},{mean_psnr_masked:.4f}\n"
    write_header = not os.path.exists(global_log_path)
    with open(global_log_path, "a") as f:
        if write_header:
            f.write(header)
        f.write(row)

    print("Per-object IoU:", avg_iou)
    print("Per-object Acc:", avg_acc)
    print(f"Mean IoU: {mean_iou_all:.4f}")
    print(f"Mean Acc: {mean_acc_all:.4f}")
    print(f"Mean PSNR (masked): {mean_psnr_masked:.4f} dB")
    print(f"Scene metrics saved to {scene_metrics_path}")
    print(f"Global summary updated at {global_log_path}")

    return {
        "per_object_mean_ious": avg_iou,
        "per_object_mean_accs": avg_acc,
        "mean_iou_all": mean_iou_all,
        "mean_acc_all": mean_acc_all,
        "mean_psnr_masked": mean_psnr_masked,
        "metrics_path": metrics_dir,
    }


# -------------------------------------------------
# main (unchanged)
# -------------------------------------------------

def main():
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config.yaml")

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    classifier_params = ClassifierParams(parser)
    args = parser.parse_args()

    device = torch.device("cuda")
    model_args = model_params.extract(args)
    dataset_args = dataset_params.extract(args)
    classifier_args = classifier_params.extract(args)
    pipeline_args = pipeline_params.extract(args)
    optimizer_args = optimization_params.extract(args)

    test_dir = os.path.dirname(args.config)
    checkpoint_dir = os.path.join(test_dir, "checkpoints")
    os.makedirs(os.path.join(test_dir, "test"), exist_ok=True)

    chpt_iter = pipeline_args.chpt_iter
    model, classifier = load_checkpoint(
        model_args, classifier_args, optimizer_args, pipeline_args,
        device, checkpoint_dir, chpt_iter, eval=True
    )

    test_data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))
    sam_dataset_args = dataset_args
    sam_dataset_args.dataset = dataset_args.dataset.replace("colmap", "segment")
    sam_test_handler = DataHandler(sam_dataset_args, rays_per_batch=0, device=device)
    sam_test_handler.reload(split="test", downsample=min(dataset_args.downsample))
    ray_fetcher = radfoam.BatchFetcher(test_data_handler.rays, batch_size=1, shuffle=False)
    rgb_fetcher = radfoam.BatchFetcher(test_data_handler.rgbs, batch_size=1, shuffle=False)

    print("\nSTAGE 1: TEST RENDER")
    psnr_val = test_render(model, classifier, test_dir, test_data_handler,
                           sam_test_handler, ray_fetcher, rgb_fetcher, classifier_args)
    print(f"Test render complete — PSNR: {psnr_val:.4f} dB\n")

    print("STAGE 2: FULL EVALUATION")
    dataset_args.dataset = dataset_args.dataset.replace("segment", "colmap")
    render_set(
        model=model,
        classifier=classifier,
        train_data_handler=DataHandler(dataset_args, rays_per_batch=0, device=device),
        test_data_handler=test_data_handler,
        encode_features=lambda x: x,
        classifier_args=classifier_args,
        dataset_args=dataset_args,
        outdir=test_dir,
        debug=False,
    )


if __name__ == "__main__":
    main()




