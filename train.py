import os
import uuid
import yaml
import gc
import warnings
import configargparse
import numpy as np
import torch
import tqdm
from torch import nn
from configs import ModelParams, PipelineParams, OptimizationParams, DatasetParams, ClassifierParams
from utils import setup_device_and_logging, setup_dataloaders, setup_model
from radfoam_model.losses import normalized_cross_entropy_loss
from radfoam_model.utils import test_render_psnr_joint

warnings.filterwarnings("ignore")

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def save_checkpoint(model, classifier, classifier_optimizer, out_dir, i=None):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    suffix = f"_{i+1}" if i is not None else ""
    model_path   = os.path.join(ckpt_dir, f"model{suffix}.pt")
    ply_path     = os.path.join(ckpt_dir, f"scene{suffix}.ply")
    cls_path     = os.path.join(ckpt_dir, f"classifier{suffix}.pt")
    model_opt_path = os.path.join(ckpt_dir, f"model_opt{suffix}.pt")
    cls_opt_path   = os.path.join(ckpt_dir, f"cls_opt{suffix}.pt")

    model.save_ply(ply_path)
    model.save_pt(model_path)
    torch.save(model.optimizer.state_dict(), model_opt_path)
    if classifier is not None:
        torch.save(classifier.state_dict(), cls_path)
        torch.save(classifier_optimizer.state_dict(), cls_opt_path)

    tag = f"[Checkpoint @ iter {i+1}]" if i is not None else "[Final]"
    print(f"{tag} Saved -> {model_path}, {ply_path}, {cls_path}")


def _densification_step(
    i,
    model,
    train_data_handler,
    pipeline_args,
    optimizer_args,
    iters_since_update,
    triangulation_update_period,
    iters_since_densification,
    next_densification_after,
):
    if iters_since_update >= triangulation_update_period:
        model.update_triangulation(incremental=True)
        iters_since_update = 0
        if triangulation_update_period < 100:
            triangulation_update_period += 2

    iters_since_update += 1
    if i + 1 >= pipeline_args.densify_from:
        iters_since_densification += 1

    if (
        iters_since_densification == next_densification_after
        and model.primal_points.shape[0] < 0.9 * model.num_final_points
    ):
        point_error, point_contribution = model.collect_error_map(
            train_data_handler, pipeline_args.white_background
        )
        model.prune_and_densify(point_error, point_contribution, pipeline_args.densify_factor)
        model.update_triangulation(incremental=False)
        triangulation_update_period = 1
        gc.collect()
        iters_since_densification = 0
        next_densification_after = max(
            int(
                (pipeline_args.densify_factor - 1)
                * model.primal_points.shape[0]
                * (pipeline_args.densify_until - pipeline_args.densify_from)
                / (model.num_final_points - model.num_init_points)
            ),
            100,
        )

    if i == optimizer_args.freeze_points:
        model.update_triangulation(incremental=False)

    return (
        iters_since_update,
        triangulation_update_period,
        iters_since_densification,
        next_densification_after,
    )


def train(args, pipeline_args, model_args, optimizer_args, dataset_args, classifier_args, out_dir):
    device, out_dir, writer = setup_device_and_logging(args, pipeline_args, dataset_args, out_dir)
    print("using device:", device)
    print("output dir:", out_dir)

    (
        train_data_handler,
        test_data_handler,
        test_ray_batch_fetcher,
        test_rgb_batch_fetcher,
        sam_train_data_handler,
        sam_test_data_handler,
        iter2downsample,
    ) = setup_dataloaders(dataset_args, device)

    classifier = nn.Linear(classifier_args.input_dim, classifier_args.num_classes).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_args.classifier_lr)

    model = setup_model(
        model_args, dataset_args, device, train_data_handler, classifier, out_dir,
        chpt_iter=pipeline_args.chpt_iter,
    )
    model.declare_optimizer(
        optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    if pipeline_args.chpt_iter != "full":
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        model_opt_path = os.path.join(ckpt_dir, f"model_opt_{pipeline_args.chpt_iter}.pt")
        cls_opt_path   = os.path.join(ckpt_dir, f"cls_opt_{pipeline_args.chpt_iter}.pt")
        if os.path.exists(model_opt_path):
            model.optimizer.load_state_dict(torch.load(model_opt_path, map_location=device))
            print(f"Restored model optimizer state from {model_opt_path}")
        if os.path.exists(cls_opt_path):
            classifier_optimizer.load_state_dict(torch.load(cls_opt_path, map_location=device))
            print(f"Restored classifier optimizer state from {cls_opt_path}")

    rgb_loss = nn.SmoothL1Loss(reduction="none")
    w_seg = pipeline_args.segmentation_loss_weight

    def run_loop(viewer):
        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch = next(data_iterator)
        sam_data_iterator = sam_train_data_handler.get_iter()
        sam_ray_batch, sam_rgb_batch = next(sam_data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        torch.cuda.synchronize()
        print("Training (joint color + segmentation)")

        with tqdm.trange(pipeline_args.iterations) as train_bar:
            for i in train_bar:
                if viewer is not None:
                    model.update_viewer(viewer)
                    viewer.step(i)

                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    print(f"Downsampling to {downsample}x at iter {i}")
                    train_data_handler.reload(split="train", downsample=downsample)
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch = next(data_iterator)

                if i == model_args.unfreeze_seg_iter:
                    for param in classifier.parameters():
                        param.requires_grad = True
                    model.identity_encoding.requires_grad = True

                depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=device)
                    .sort(dim=-1, descending=True)
                    .values
                )
                rgba_output, seg_output, depth, _, _, _ = model(
                    ray_batch, depth_quantiles=depth_quantiles
                )

                # Color branch
                opacity = rgba_output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = rgba_output[..., :3] + (1 - opacity)
                else:
                    rgb_output = rgba_output[..., :3]
                color_loss = rgb_loss(rgb_batch, rgb_output)
                opacity_loss = ((1 - opacity) ** 2).mean()
                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = ((depth[..., 0] - depth[..., 1]).abs() * valid_depth_mask).mean()
                w_depth = pipeline_args.quantile_weight * min(2 * i / pipeline_args.iterations, 1)

                # Segmentation branch
                seg_features = seg_output[..., -classifier_args.input_dim:]
                logits = classifier(seg_features)

                seg_loss = normalized_cross_entropy_loss(
                    logits, sam_rgb_batch.squeeze(-1),
                    sam_rgb_batch.max().item() + 1,
                )

                # TV regularization weight (decays after densification ends)
                w_tv = pipeline_args.tv_lambda
                if i >= pipeline_args.densify_until:
                    w_tv *= 0.99 ** (i / 1000)

                if pipeline_args.use_TV:
                    tv_loss = w_tv * model.total_variation_loss().mean()
                else:
                    tv_loss = torch.tensor(0.0, device=device)

                loss = (
                    color_loss.mean()
                    + opacity_loss
                    + w_depth * quant_loss
                    + w_seg * seg_loss
                    + tv_loss
                )

                model.optimizer.zero_grad(set_to_none=True)
                classifier_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                ray_batch, rgb_batch = next(data_iterator)
                sam_ray_batch, sam_rgb_batch = next(sam_data_iterator)
                model.optimizer.step()
                classifier_optimizer.step()
                model.update_learning_rate(i)

                train_bar.set_postfix(
                    color=f"{color_loss.mean().item():.5f}",
                    seg=f"{seg_loss.item():.5f}",
                    tv=f"{tv_loss.item():.5f}",
                )

                if i % 100 == 99 and pipeline_args.debug:
                    writer.add_scalar("train/rgb_loss", color_loss.mean(), i)
                    writer.add_scalar("train/seg_loss", seg_loss.item(), i)
                    writer.add_scalar("train/tv_loss", tv_loss.item(), i)
                    writer.add_scalar("train/num_points", model.primal_points.shape[0], i)
                    writer.add_scalar("lr/points_lr", model.xyz_scheduler_args(i), i)
                    writer.add_scalar("lr/density_lr", model.den_scheduler_args(i), i)
                    writer.add_scalar("lr/attr_lr", model.attr_dc_scheduler_args(i), i)
                    test_psnr = test_render_psnr_joint(
                        model=model,
                        test_data_handler=test_data_handler,
                        ray_batch_fetcher=test_ray_batch_fetcher,
                        rgb_batch_fetcher=test_rgb_batch_fetcher,
                        device=device,
                        white_background=pipeline_args.white_background,
                    )
                    writer.add_scalar("test/psnr", test_psnr, i)

                (
                    iters_since_update,
                    triangulation_update_period,
                    iters_since_densification,
                    next_densification_after,
                ) = _densification_step(
                    i, model, train_data_handler, pipeline_args, optimizer_args,
                    iters_since_update, triangulation_update_period,
                    iters_since_densification, next_densification_after,
                )

                if pipeline_args.save_every and (i + 1) % pipeline_args.save_every == 0:
                    save_checkpoint(model, classifier, classifier_optimizer, out_dir, i)

        save_checkpoint(model, classifier, classifier_optimizer, out_dir)
        del data_iterator, sam_data_iterator
        writer.close()

    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
    }

    if pipeline_args.viewer:
        model.show(run_loop, iterations=pipeline_args.iterations, **viewer_options)
    else:
        run_loop(viewer=None)


def main():
    parser = configargparse.ArgParser(
        default_config_files=["configs/mipnerf360_outdoor_config.yaml"]
    )
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")

    model_params        = ModelParams(parser)
    pipeline_params     = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params      = DatasetParams(parser)
    classifier_params   = ClassifierParams(parser)

    args = parser.parse_args()

    model_args      = model_params.extract(args)
    pipeline_args   = pipeline_params.extract(args)
    optimizer_args  = optimization_params.extract(args)
    dataset_args    = dataset_params.extract(args)
    classifier_args = classifier_params.extract(args)

    experiment_name = (
        pipeline_args.experiment_name
        or f"{dataset_args.scene}@{str(uuid.uuid4())[:8]}"
    )
    out_dir    = f"output/{experiment_name}"
    model_ckpt = f"{out_dir}/model.pt"

    if not os.path.exists(model_ckpt):
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        yaml.add_representer(
            list,
            lambda dumper, data: dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            ),
        )
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

        train(args, pipeline_args, model_args, optimizer_args, dataset_args, classifier_args, out_dir)


if __name__ == "__main__":
    print("Current GPU:", torch.cuda.current_device())
    main()
    gc.collect()
    torch.cuda.empty_cache()
