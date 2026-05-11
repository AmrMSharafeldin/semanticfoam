# Semantic Foam: Semantic Decomposition for Real-Time Radiance Fields

## Amr Sharafeldin, Shrisudhan Govindarajan, Thomas Walker, Aryan Mikaeili, Daniel Rebain, Kwang Moo Yi, Andrea Tagliasacchi

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=githubpages&logoColor=white)](http://semanticfoam.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](#)

</div>

This repository contains the official implementation of **Semantic Foam: Semantic Decomposition for Real-Time Radiance Fields**. Semantic Foam extends Radiant Foam with an explicit semantic feature field defined over a volumetric Voronoi decomposition, enabling high-quality object-level segmentation and editing directly within real-time radiance fields.

By leveraging the implicit surface formulation of Radiant Foam alongside spatial regularization over Voronoi neighborhoods, Semantic Foam produces coherent non-convex object masks without requiring convex hull post-processing. The method supports semantic segmentation, object extraction, insertion, removal, and novel view synthesis of edited scenes.

The repository includes scripts for training, evaluation, semantic editing, rendering, and visualization, alongside an interactive viewer for inspecting trained scenes and semantic decompositions.

Everything in this repository is still under active development and subject to change.

> Warning: this is an organic, free-range research codebase, and should be treated with the appropriate care when integrating it into any other software.

---



# Getting Started

Start by cloning the repository:

```bash
git clone https://github.com/<your-repo>/semanticfoam.git
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
@article{sharafeldin2026semanticfoam,
  title   = {Semantic Foam: Semantic Decomposition for Real-Time Radiance Fields},
  author  = {Sharafeldin, Amr and Govindarajan, Shrisudhan and Walker, Thomas and
             Mikaeili, Aryan and Rebain, Daniel and Yi, Kwang Moo and
             Tagliasacchi, Andrea},
  journal = {arXiv},
  year    = {2026},
}
```

---

