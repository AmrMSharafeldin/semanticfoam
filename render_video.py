import os
import math
import json
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image

import radfoam
from radfoam_model.utils import visualize_obj
from radfoam_model.scene import RadFoamScene
 

def compute_object_center(model, seg_all, cls_ids):
    """Return the bbox midpoint of scene points whose class is in cls_ids (MAD outlier-filtered)."""
    if not isinstance(cls_ids, (list, tuple)):
        cls_ids = [cls_ids]

    center_scene = model.primal_points.mean(dim=0)
    mask = torch.isin(seg_all, torch.tensor(cls_ids, device=seg_all.device))

    if not mask.any():
        return center_scene.detach()

    pts = model.primal_points[mask]

    d = torch.norm(pts - pts.mean(dim=0), dim=1)
    med = torch.median(d)
    mad = torch.median(torch.abs(d - med)) + 1e-6
    pts = pts[d < (med + 3.0 * mad)]

    mins = pts.min(dim=0).values
    maxs = pts.max(dim=0).values
    return ((mins + maxs) * 0.5).detach()


def make_video(folder, name="video.mp4", fps=30):
    """Write all PNGs in folder to an h264 mp4 via ffmpeg."""
    frames = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    if not frames:
        return

    first = cv2.imread(os.path.join(folder, frames[0]))
    H, W = first.shape[:2]

    raw   = os.path.join(folder, "raw.mp4")
    final = os.path.join(folder, name)

    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        writer.write(cv2.imread(os.path.join(folder, f)))
    writer.release()

    subprocess.run(
        ["ffmpeg", "-y", "-i", raw, "-vcodec", "libx264", "-pix_fmt", "yuv420p", final],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    os.remove(raw)
    print(f"[VIDEO] Saved -> {final}")


def export_cameras_to_json(cameras, save_path):
    """Serialize camera trajectory to a JSON file with world-view transforms."""
    frames = []
    for i, cam in enumerate(cameras):
        right   = cam["right"].detach().cpu().numpy()
        up      = cam["up"].detach().cpu().numpy()
        forward = cam["forward"].detach().cpu().numpy()
        pos     = cam["position"].detach().cpu().numpy()

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = np.stack([right, -up, forward], axis=1)
        M[:3,  3] = pos

        frames.append({"id": i, "world_view_transform": M.tolist(), "fov": float(cam["fov"])})

    with open(save_path, "w") as f:
        json.dump(frames, f, indent=2)
    print(f"[EXPORT] Saved cameras -> {save_path}")


def _get_first_camera(data_handler, device):
    """Extract center, right, up, forward from the first image ray grid."""
    rays = data_handler.rays[5].to(device)
    H, W, _ = rays.shape

    center  = rays[H // 2, W // 2, :3]
    forward = rays[H // 2, W // 2, 3:]
    forward = forward / torch.norm(forward)

    right = rays[H // 2, W // 2 + 10, 3:] - rays[H // 2, W // 2 - 10, 3:]
    right = right / torch.norm(right)

    up = torch.cross(right, forward, dim=0)
    up = up / torch.norm(up)

    return center, right, up, forward


def _orbit_basis(global_up, device):
    """Two orthonormal vectors spanning the plane perpendicular to global_up."""
    tmp = torch.tensor([0.0, 0.0, 1.0], device=device)
    if torch.abs(torch.dot(global_up, tmp)) > 0.99:
        tmp = torch.tensor([1.0, 0.0, 0.0], device=device)

    gx = torch.cross(global_up, tmp)
    gx = gx / torch.norm(gx)
    gy = torch.cross(gx, global_up)
    gy = gy / torch.norm(gy)
    return gx, gy


def _make_cam(pos, target, global_up, fov, H, W):
    f = target - pos
    f = f / torch.norm(f)

    rvec = torch.cross(f, global_up)
    rvec = rvec / torch.norm(rvec)

    u = torch.cross(rvec, f)
    u = u / torch.norm(u)

    return {"position": pos, "forward": f, "right": rvec, "up": u, "fov": fov, "H": H, "W": W}


def build_cameras(ttype, model, target, data_handler, args):
    device = model.device
    W, H   = data_handler.img_wh
    target = target.to(device)
    up     = data_handler.viewer_up.float().to(device)

    if ttype == "firstcam":
        center, right, fc_up, forward = _get_first_camera(data_handler, device)
        cams = []
        for i in range(args.n_frames):
            theta = -2 * math.pi * i / args.n_frames
            offset = right * (args.radius * math.cos(theta)) + fc_up * (args.radius * math.sin(theta))
            pos = center + offset + forward * args.forward_push
            cams.append(_make_cam(pos, target, fc_up, args.fov, H, W))
        return cams

    gx, gy = _orbit_basis(up, device)

    if ttype == "360":
        cams = []
        for i in range(args.n_frames):
            a = -2.0 * math.pi * i / args.n_frames * args.n_rots
            pos = (target
                   + gx * (args.radius * math.cos(a))
                   + gy * (args.radius * math.sin(a))
                   + up * args.height)
            cams.append(_make_cam(pos, target, up, args.fov, H, W))
        return cams

    if ttype == "spiral":
        cams = []
        for i in range(args.n_frames):
            t = i / args.n_frames
            a = -2.0 * math.pi * t * args.n_rots
            r = args.radius * (0.5 + 0.5 * math.cos(2 * math.pi * t))
            h = args.height * math.sin(2 * math.pi * t)
            pos = (target
                   + gx * (r * math.cos(a))
                   + gy * (r * math.sin(a))
                   + up * h)
            cams.append(_make_cam(pos, target, up, args.fov, H, W))
        return cams

    raise ValueError(f"Unknown trajectory type: {ttype!r}")


def _cam_to_rays(cam, device):
    H, W   = cam["H"], cam["W"]
    aspect = W / H

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )

    px = x * math.tan(cam["fov"] / 2) * aspect
    py = -y * math.tan(cam["fov"] / 2)
    pz = torch.ones_like(px)

    dirs = (px.unsqueeze(-1) * cam["right"]
            + py.unsqueeze(-1) * cam["up"]
            + pz.unsqueeze(-1) * cam["forward"])
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    orig = cam["position"].expand_as(dirs)
    return torch.cat([orig, dirs], dim=-1).reshape(-1, 6), H, W


def render_scene(model, classifier, classifier_args, cameras, out_dir, name, fps):
    """Render full-scene RGB + segmentation videos."""
    device = model.device

    export_cameras_to_json(cameras, os.path.join(out_dir, f"{name}.json"))

    points, _, _, _ = model.get_trace_data()
    positions    = torch.stack([c["position"] for c in cameras]).to(device)
    start_points = radfoam.nn(points, model.aabb_tree, positions)

    rgb_dir = os.path.join(out_dir, name + "_rgb")
    seg_dir = os.path.join(out_dir, name + "_seg")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    for i, cam in enumerate(cameras):
        print(f"[RENDER] {name}  {i + 1}/{len(cameras)}")
        rays, H, W = _cam_to_rays(cam, device)

        with torch.no_grad():
            out, seg_out, *_ = model(rays, start_points[i])

            rgb = (out[..., :3] + (1 - out[..., -1:])).reshape(H, W, 3).clamp(0, 1)
            rgb_u8 = (rgb.cpu().numpy() * 255).astype(np.uint8)

            seg_feats = seg_out[..., -classifier_args.input_dim:]
            pred = classifier(seg_feats).argmax(dim=-1).reshape(H, W).cpu().numpy()
            seg_rgb = visualize_obj(pred, classifier_args.num_classes)

        Image.fromarray(rgb_u8).save(os.path.join(rgb_dir, f"{i:04d}.png"))
        Image.fromarray(seg_rgb).save(os.path.join(seg_dir, f"{i:04d}.png"))

    make_video(rgb_dir, f"{name}_rgb.mp4", fps)
    make_video(seg_dir, f"{name}_seg.mp4", fps)


def render_object(model, classifier, classifier_args, cameras, cls_ids, out_dir, name, fps):
    """Render per-object video with white background for non-target pixels."""
    device = model.device

    export_cameras_to_json(cameras, os.path.join(out_dir, f"{name}.json"))

    points, _, _, _ = model.get_trace_data()
    positions    = torch.stack([c["position"] for c in cameras]).to(device)
    start_points = radfoam.nn(points, model.aabb_tree, positions)

    obj_dir = os.path.join(out_dir, name + "_obj")
    os.makedirs(obj_dir, exist_ok=True)

    cls_tensor = torch.tensor(cls_ids, device=device)

    for i, cam in enumerate(cameras):
        print(f"[RENDER] {name}  {i + 1}/{len(cameras)}")
        rays, H, W = _cam_to_rays(cam, device)

        with torch.no_grad():
            out, seg_out, *_ = model(rays, start_points[i])

            opacity = out[..., -1:].clamp(0, 1)
            rgb = (out[..., :3] * opacity + (1 - opacity)).reshape(H, W, 3)

            seg_feats = seg_out[..., -classifier_args.input_dim:]
            pred = classifier(seg_feats).argmax(dim=-1).reshape(H, W)

            obj_mask = torch.isin(pred, cls_tensor).float().unsqueeze(-1)
            rgb = rgb * obj_mask + (1 - obj_mask)

        img = (rgb.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(obj_dir, f"{i:04d}.png"))

    make_video(obj_dir, f"{name}_obj.mp4", fps)



def run_application(
    model,
    classifier,
    classifier_args,
    pipeline_args,
    target_class,
    application_mode="density",
    model_args=None,
):

    with torch.no_grad():
        points, attributes, _, _ = model.get_trace_data()

        seg_features = attributes[..., -classifier_args.input_dim:]
        logits_pts = classifier(seg_features)
        probs = torch.softmax(logits_pts, dim=-1)

        pred_classes_pts = probs.argmax(dim=-1)

        mask = torch.isin(
            pred_classes_pts,
            torch.tensor(target_class, device=pred_classes_pts.device)
        )

        if not mask.any():
            print("[APP] No matching points")
            return False

        print(f"[APP] Matched {mask.sum().item()} points")

        if application_mode == "density":
            model.shutdown_density(mask)

        elif application_mode == "remove":
            model.remove_points(mask, density_thresh=0.01)

        elif application_mode == "duplicate":
            model.duplicate_points(mask)

        elif application_mode == "move":
            model.move_points(mask)

        elif application_mode == "insert":
            model.remove_points(mask, density_thresh=0.01)

            import_base = pipeline_args.import_asset_path
            loaded_scene = RadFoamScene(args=model_args, device=model.device)
            loaded_scene.load_pt(import_base)

            model.import_asset(
                loaded_scene,
                translation=pipeline_args.import_translation,
                scale_factor=pipeline_args.import_scale,
                rotation_degrees=pipeline_args.import_rotation,
            )

        else:
            print("[APP] Unknown mode")
            return False

    return True