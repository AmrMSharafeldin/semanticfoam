import os
import warnings

import numpy as np
import torch
import configargparse

from data_loader import DataHandler
from configs import ModelParams, PipelineParams, OptimizationParams, DatasetParams, ClassifierParams
from utils import load_checkpoint
from render_video import (
    build_cameras, compute_object_center,
    render_scene, render_object, run_application,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--trajectory_type", default="360",
                        choices=["360", "firstcam", "spiral"],
                        help="Camera trajectory type for rendered videos")
    parser.add_argument("--video_target_class", type=int, nargs="+", default=None,
                        help="Class IDs whose centroid is used as orbit centre for scene videos")
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

    args            = parser.parse_args()
    device          = torch.device("cuda")
    model_args      = model_params.extract(args)
    dataset_args    = dataset_params.extract(args)
    classifier_args = classifier_params.extract(args)
    pipeline_args   = pipeline_params.extract(args)
    optimizer_args  = optimization_params.extract(args)

    base_dir  = os.path.dirname(args.config)
    ckpt_dir  = os.path.join(base_dir, "checkpoints")
    video_dir = os.path.join(base_dir, "videos_edited")

    model, classifier = load_checkpoint(
        model_args, classifier_args, optimizer_args,
        pipeline_args, device, ckpt_dir,
        pipeline_args.chpt_iter, eval=True,
    )

    train_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
    train_handler.reload(split="train", downsample=min(dataset_args.downsample))

    # ---- Apply scene edit ----
    if pipeline_args.target_class != [-1]:
        print(f"[EDIT] mode={pipeline_args.application_mode}  classes={pipeline_args.target_class}")
        run_application(
            model=model,
            classifier=classifier,
            classifier_args=classifier_args,
            pipeline_args=pipeline_args,
            target_class=pipeline_args.target_class,
            application_mode=pipeline_args.application_mode,
            model_args=model_args,
        )
    else:
        print("[EDIT] No target_class set — rendering without edit")

    # classify all points once for object centering
    _, feats, _, _ = model.get_trace_data()
    with torch.no_grad():
        seg_all = classifier(feats[..., -classifier_args.input_dim:]).argmax(dim=-1)

    ttype = args.trajectory_type

    # ---- Full-scene RGB-only video ----
    if args.video_target_class:
        scene_target = compute_object_center(model, seg_all, args.video_target_class)
    else:
        scene_target = model.primal_points.mean(dim=0)

    scene_cams = build_cameras(ttype, model, scene_target, train_handler, args)
    scene_name = f"scene_{ttype}"
    scene_out  = os.path.join(video_dir, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    print(f"\n[VIDEO] Full scene — {scene_name}")
    render_scene(model, classifier, classifier_args,
                 scene_cams, scene_out, scene_name, args.fps, rgb_only=True)

    # ---- Per-object RGB videos ----
    obj_groups = []
    if args.video_target_class:
        obj_groups.append(args.video_target_class)
    elif pipeline_args.target_class != [-1]:
        obj_groups.append(pipeline_args.target_class)

    for cls_ids in obj_groups:
        obj_center = compute_object_center(model, seg_all, cls_ids)
        obj_cams   = build_cameras(ttype, model, obj_center, train_handler, args)
        obj_name   = f"obj_{'_'.join(map(str, cls_ids))}_{ttype}"
        obj_out    = os.path.join(video_dir, obj_name)
        os.makedirs(obj_out, exist_ok=True)

        print(f"\n[VIDEO] Object {cls_ids} — {obj_name}")
        render_object(model, classifier, classifier_args,
                      obj_cams, cls_ids, obj_out, obj_name, args.fps)


if __name__ == "__main__":
    main()
