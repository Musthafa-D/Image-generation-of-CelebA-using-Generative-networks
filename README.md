# CelebA Image Generation Using Generative Networks (PyTorch)

## Project Overview

This project implements **image generation** on the CelebA face attribute dataset using generative neural networks in PyTorch.  
CelebA contains over 200,000 celebrity face images labeled with multiple attributes. In this work, only the image content is used to train a generative model that learns to produce realistic face images from random noise.

Generative models are widely used for tasks such as:
- Artistic content generation
- Data augmentation
- Understanding latent representations

---

## Dataset

- **CelebA (Celeb Faces Attributes)**
- ~200,000 images (cropped faces)
- High diversity in appearance, pose, background
- Used here for **conditional and uncoditional image generation**

The dataset must be downloaded separately due to size constraints.

---

## What This Project Includes

- PyTorch-based data loading and preprocessing
- Generative model architecture (GAN / similar)
- Training loop with adversarial optimization
- Image generation and visualization

---
