import os
import uuid
import yaml
import copy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import radfoam
from data_loader import DataHandler
from radfoam_model.scene import RadFoamScene


def setup_model(model_args, dataset_args, device, train_data_handler, classifier, out_dir,
                chpt_iter="full"):
    model = RadFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )

    if chpt_iter != "full":
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        ckpt_model = os.path.join(ckpt_dir, f"model_{chpt_iter}.pt")
        ckpt_classifier = os.path.join(ckpt_dir, f"classifier_{chpt_iter}.pt")
        if os.path.exists(ckpt_model) and os.path.exists(ckpt_classifier):
            print(f"Resuming from {ckpt_model}, {ckpt_classifier}")
            model.load_pt(ckpt_model)
            classifier.load_state_dict(torch.load(ckpt_classifier, map_location=device))
        else:
            raise FileNotFoundError(f"Checkpoint not found for iter {chpt_iter}.")

    # Geometry trainable; classifier and identity encoding start frozen (unfrozen later in loop)
    for param in model.parameters():
        param.requires_grad = True
    for param in classifier.parameters():
        param.requires_grad = False
    if hasattr(model, "identity_encoding"):
        model.identity_encoding.requires_grad = False

    return model


def setup_device_and_logging(args, pipeline_args, dataset_args, out_dir):
    device = torch.device(args.device)
    experiment_name = pipeline_args.experiment_name or f"{dataset_args.scene}@{str(uuid.uuid4())[:8]}"
    out_dir = f"output/{experiment_name}"
    os.makedirs(f"{out_dir}/test", exist_ok=True)
    writer = SummaryWriter(out_dir, purge_step=0)
    if pipeline_args.debug:
        yaml.add_representer(
            list,
            lambda dumper, data: dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            ),
        )
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)
    return device, out_dir, writer


def setup_dataloaders(dataset_args, device):
    iter2downsample = dict(
        zip(dataset_args.downsample_iterations, dataset_args.downsample)
    )

    train_data_handler = DataHandler(
        dataset_args, rays_per_batch=dataset_args.radiance_batch_size, device=device
    )
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    test_data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    test_ray_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_rgb_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rgbs, batch_size=1, shuffle=False
    )

    segmentation_dataset_args = copy.deepcopy(dataset_args)
    segmentation_dataset_args.dataset = dataset_args.dataset.replace("colmap", "segment")

    sam_test_data_handler = DataHandler(segmentation_dataset_args, rays_per_batch=0, device=device)
    sam_test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    sam_train_data_handler = DataHandler(
        segmentation_dataset_args,
        rays_per_batch=dataset_args.segmentation_batch_size,
        device=device,
    )
    sam_train_data_handler.reload(split="train", downsample=downsample)

    return (
        train_data_handler,
        test_data_handler,
        test_ray_batch_fetcher,
        test_rgb_batch_fetcher,
        sam_train_data_handler,
        sam_test_data_handler,
        iter2downsample,
    )


def load_checkpoint(model_args, classifier_args, optimizer_args, pipeline_args, device,
                    checkpoint_dir, chpt_iter="final", eval=True):
    if chpt_iter in ("final", "full"):
        model_name = "model.pt"
        cls_name   = "classifier.pt"
    else:
        model_name = f"model_{chpt_iter}.pt"
        cls_name   = f"classifier_{chpt_iter}.pt"

    model_path = os.path.join(checkpoint_dir, model_name)
    cls_path   = os.path.join(checkpoint_dir, cls_name)

    model = RadFoamScene(args=model_args, device=device)
    if os.path.exists(model_path):
        print(f"Loading model checkpoint: {model_path}")
        model.load_pt(model_path)
    else:
        print(f"No model checkpoint found at {model_path}, initializing from scratch.")

    classifier = nn.Linear(classifier_args.input_dim, classifier_args.num_classes).to(device)
    if os.path.exists(cls_path):
        print(f"Loading classifier checkpoint: {cls_path}")
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
    else:
        print(f"No classifier checkpoint found at {cls_path}, initializing new classifier.")

    if eval:
        model.eval()
        classifier.eval()
    else:
        model.train()
        classifier.train()

    model.declare_optimizer(
        optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    return model, classifier
