import torch
import configargparse

from data_loader import DataHandler
from configs import ModelParams, PipelineParams, OptimizationParams, DatasetParams, ClassifierParams
from utils import load_checkpoint
import os


def main():
    parser = configargparse.ArgParser(default_config_files=["configs/mipnerf360_outdoor_config.yaml"])
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")

    model_params        = ModelParams(parser)
    pipeline_params     = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params      = DatasetParams(parser)
    classifier_params   = ClassifierParams(parser)

    args            = parser.parse_args()
    device          = torch.device("cuda")
    model_args      = model_params.extract(args)
    pipeline_args   = pipeline_params.extract(args)
    optimizer_args  = optimization_params.extract(args)
    dataset_args    = dataset_params.extract(args)
    classifier_args = classifier_params.extract(args)

    out_dir  = f"output/{pipeline_args.experiment_name or dataset_args.scene}"
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    model, _ = load_checkpoint(
        model_args, classifier_args, optimizer_args, pipeline_args,
        device, ckpt_dir, pipeline_args.chpt_iter, eval=True,
    )

    test_data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    viewer_options = {
        "camera_pos":     test_data_handler.viewer_pos,
        "camera_up":      test_data_handler.viewer_up,
        "camera_forward": test_data_handler.viewer_forward,
    }

    model.show(lambda viewer: model.update_viewer(viewer), **viewer_options)


if __name__ == "__main__":
    main()
