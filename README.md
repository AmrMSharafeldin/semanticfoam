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

