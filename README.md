# Semantic Foam: Unifying Spatial and Semantic Scene Decomposition

## Amr Sharafeldin, Shrisudhan Govindarajan, Thomas Walker, Aryan Mikaeili, Daniel Rebain, Kwang Moo Yi, Andrea Tagliasacchi

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-5F8D7A?style=for-the-badge&logo=githubpages&logoColor=white)](http://semanticfoam.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-8FAF9D?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.26262)

</div>

This repository contains the official implementation of **Semantic Foam: Unifying Spatial and Semantic Scene Decomposition**. Semantic Foam extends Radiant Foam with an explicit semantic feature field defined over a volumetric Voronoi decomposition, enabling high-quality object-level segmentation and editing directly within real-time radiance fields.

By leveraging the implicit surface formulation of Radiant Foam alongside spatial regularization over Voronoi neighborhoods, Semantic Foam produces coherent object masks without requiring convex hull post-processing. The method supports semantic segmentation, object extraction, insertion, removal, and novel view synthesis of edited scenes.

The repository includes scripts for training, evaluation, scene editing , and visualization, alongside a real-time viewer that can be used to visualize trained models, or optionally to observe the progression of models as they train.

Everything in this repository is still under active development and subject to change.

> Warning: this is an organic, free-range research codebase, and should be treated with the appropriate care when integrating it into any other software.

---



## Getting Started

Start by cloning the repository:

```bash
git clone https://github.com/AmrMSharafeldin/semanticfoam.git
cd semanticfoam
```

You will need a Linux environment with:

- Python 3.10+
- CUDA 12.x
- A CUDA-compatible GPU with Compute Capability 7.0 or higher

After creating your Python environment, install PyTorch matching your CUDA version:

```bash
conda create -n semanticfoam python=3.11 -y
conda activate semanticfoam

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

We have tested our method on the following datasets:

- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/?utm_source=chatgpt.com)
- [NVOS](https://jason718.github.io/nvos/?utm_source=chatgpt.com)
- [LERF](https://www.lerf.io/?utm_source=chatgpt.com)

---

### Expected Directory Structure

```text
dataset_root/
├── images/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
│
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
│
├── object_mask/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
│
├── segmentation_labels/
│   └── masks/
│       ├── object_1/
│       │   ├── 0000.png
│       │   ├── 0001.png
│       │   └── ...
│       │
│       ├── object_2/
│       │   ├── 0000.png
│       │   ├── 0001.png
│       │   └── ...
│       │
│       └── ...
│
└── config.yaml
```

---

### Folder Description

| Folder | Description |
| --- | --- |
| `images/` | Input RGB training images |
| `sparse/0/` | COLMAP sparse reconstruction outputs |
| `object_mask/` | Ground truth segmentation maps |
| `segmentation_labels/` | Per-object binary masks used for IoU and accuracy evaluation |
| `config.yaml` | Training configuration file |

---

### Notes

- Segmentation maps are generated using [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA?utm_source=chatgpt.com).  
  A preprocessing script will be released soon.

- COLMAP reconstructions can be generated using:

```bash
python prepare_colmap_data.py --data_dir data/your_own_data
```

- Per-object binary masks were generated using [SAM-UI](https://github.com/mtaktash/sam-ui?utm_source=chatgpt.com).


## Training

Training is launched with:

```bash
python train.py -c configs/<config_file>.yaml
```


# Testing and Evaluation

Use the following script to evaluate a trained checkpoint, render test views, compute rendering metrics, evaluate segmentation quality, extract segmented assets, and generate trajectory videos.

---

## Basic Usage

```bash
python test.py -c output/<checkpoint_directory>/config.yaml
```

Example:

```bash
python test.py -c output/garden_semanticfoam/config.yaml
```

---

## What `test.py` Does

The evaluation pipeline is divided into multiple stages:

### 1. Test Rendering

Renders all test views and computes:

- PSNR
- SSIM
- LPIPS

Also saves:

- RGB renders
- Error maps
- Segmentation predictions
- Confidence maps

---

### 2. Segmentation Evaluation *(Optional)*

Computes:

- IoU
- Pixel Accuracy

Also saves:

- Predicted masks
- Ground-truth masks
- Extracted object crops

---

### 3. Object Extraction *(Optional)*

Extracts segmented scene assets and saves each object as:

- `.pt`
- `.ply`

---

### 4. Trajectory Video Rendering *(Optional)*

Generates:

- RGB trajectory videos
- Segmentation videos
- Per-object isolated renders

---

# Output Structure

After evaluation, outputs are written inside the checkpoint directory:

```text
output/<checkpoint_directory>/
├── test/
│   ├── metrics.txt
│   ├── view_000/
│   │   ├── gt_rgb.png
│   │   ├── pred_rgb.png
│   │   ├── error_map.png
│   │   ├── gt_segmentation.png
│   │   ├── pred_segmentation.png
│   │   └── confidence_map.png
│
├── segmentation_eval/
│   └── test/
│
├── metrics_summary/
│   ├── <scene>_metrics.txt
│   └── all_scenes_summary.csv
│
├── objects/
│   ├── <object>.pt
│   └── <object>.ply
│
└── videos/
    ├── scene_360/
    └── obj_<object_name>/
```

---

# Common Usage Examples

## Basic Evaluation

```bash
python test.py -c output/garden/config.yaml
```

---

## Segmentation Evaluation + Object Extraction

```bash
python test.py \
    -c output/garden/config.yaml \
    --eval_segmentation
```

---

## Render 360° Trajectory Videos

```bash
python test.py \
    -c output/garden/config.yaml \
    --trajectory_type 360
```

---

## Render First Camera Orbit

```bash
python test.py \
    -c output/garden/config.yaml \
    --trajectory_type firstcam
```

---

## Render Spiral Trajectory

```bash
python test.py \
    -c output/garden/config.yaml \
    --trajectory_type spiral
```

---

## Full Evaluation Pipeline

```bash
python test.py \
    -c output/garden/config.yaml \
    --eval_segmentation \
    --trajectory_type 360
```

---

# Important Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--eval_segmentation` | `False` | Run segmentation evaluation and compute IoU / Accuracy metrics. Also extracts object assets. |
| `--trajectory_type` | `None` | Render trajectory videos. Options: `360`, `firstcam`, `spiral`. |
| `--conf_thresh` | `0.9` | Softmax confidence threshold used during object extraction. |
| `--fps` | `30` | Output video frame rate. |
| `--n_frames` | `200` | Number of frames in rendered trajectories. |
| `--radius` | `3.5` | Orbit radius for trajectory rendering. |
| `--fov` | `0.7` | Camera field of view used during trajectory rendering. |
| `--height` | `0.8` | Camera height offset for trajectories. |
| `--forward_push` | `0.0` | Push camera forward during first-camera trajectories. |
| `--n_rots` | `2` | Number of rotations for orbit trajectories. |

---

# Trajectory Types

| Type | Description |
|---|---|
| `360` | Global orbit around the scene or selected object center. |
| `firstcam` | Orbit relative to the first training camera pose. |
| `spiral` | Smooth spiral trajectory around the scene. |

---

# Rendering Metrics

The script computes the following rendering metrics:

| Metric | Description |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity |
| LPIPS | Learned Perceptual Image Patch Similarity |

---

# Segmentation Metrics

For segmentation evaluation:

| Metric | Description |
|---|---|
| IoU | Intersection-over-Union |
| Accuracy | Pixel-wise segmentation accuracy |

---

# Notes

- Segmentation evaluation requires:

```text
segmentation_labels/masks/
```

to exist inside the dataset directory.



- Video rendering automatically generates:
  - full-scene videos
  - segmentation videos
  - isolated per-object renders

- All renders and metrics are saved automatically into the experiment output directory.

## Citation

```bibtex
@inproceedings{semanticfoam2026,
  title     = {Semantic Foam: Unifying Spatial and Semantic Scene Decomposition},
  author    = {Sharafeldin, Amr and Govindarajan, Shrisudhan and Walker, Thomas and
               Mikaeili, Aryan and Rebain, Daniel and Yi, Kwang Moo and
               Tagliasacchi, Andrea},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

