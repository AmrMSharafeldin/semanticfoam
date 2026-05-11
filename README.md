# Semantic Foam: Unifying Spatial and Semantic Scene Decomposition

## Amr Sharafeldin, Shrisudhan Govindarajan, Thomas Walker, Aryan Mikaeili, Daniel Rebain, Kwang Moo Yi, Andrea Tagliasacchi

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=githubpages&logoColor=white)](http://semanticfoam.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.26262)

</div>

This repository contains the official implementation of **Semantic Foam: Unifying Spatial and Semantic Scene Decomposition**. Semantic Foam extends Radiant Foam with an explicit semantic feature field defined over a volumetric Voronoi decomposition, enabling high-quality object-level segmentation and editing directly within real-time radiance fields.

By leveraging the implicit surface formulation of Radiant Foam alongside spatial regularization over Voronoi neighborhoods, Semantic Foam produces coherent object masks without requiring convex hull post-processing. The method supports semantic segmentation, object extraction, insertion, removal, and novel view synthesis of edited scenes.

The repository includes scripts for training, evaluation, scene editing , and visualization, alongside a real-time viewer that can be used to visualize trained models, or optionally to observe the progression of models as they train.

Everything in this repository is still under active development and subject to change.

> Warning: this is an organic, free-range research codebase, and should be treated with the appropriate care when integrating it into any other software.

---



# Getting Started

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



# Citation

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

