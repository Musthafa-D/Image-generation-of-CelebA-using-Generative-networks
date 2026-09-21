# CelebA Image Generation with GANs

Adversarial image-generation experiments in PyTorch, combining generator/discriminator training with classifier-based evaluation and interpretation. The project explores the images produced by a generative model and the behaviour of the networks used to assess them.

The study configuration selects an unconditional GAN on CelebA. The code also contains a conditional variant using facial attributes, so results should identify the model and conditioning settings used.

## Approach

- Map random latent vectors to synthetic images through a generator.
- Train a discriminator to distinguish generated and dataset images.
- Explore linear and NLRL discriminator output heads.
- Configure learning rates, latent dimensions and network parameters.
- Inspect training curves, generated samples and attribution/representation plots.

## Code organisation

| File | Role |
|---|---|
| `networks.py` | GAN/conditional GAN and auxiliary network definitions |
| `data_loader.py` | Framework-based dataset preparation |
| `learner.py` | Adversarial training and evaluation |
| `optuna_hyp.py` | Optuna experiments |
| `main.py` | Study launcher |
| `dummy_main.py` | Fixed-run launcher |
| `metrics.py` and `plots.py` | Analysis and visualisation |
| `StudySummary.ipynb` | Saved study inspection |

## Environment and use

The code uses PyTorch and the external `ccbdl` framework for configuration, data loading, experiment storage and parts of the learning workflow. A compatible installation of that framework is required; it is not bundled here. Other dependencies include torchvision, NumPy, Matplotlib, Optuna and Captum, with additional analysis libraries used by individual modules.

Use the original compatible environment, prepare the dataset at the configured location, and run from the repository root so relative paths resolve correctly. The archive does not include a complete dependency lock file. Supply datasets and optional pretrained models separately where referenced.

The fixed-run launcher reads `dummy_config.yaml`. Inspect its model type, image representation, discriminator head and learning-rate settings. The learner also loads classifiers for auxiliary metrics; supply the checkpoints expected by the selected path.

After preparation, the fixed-run entry point is:

```bash
python dummy_main.py
```

For studies, `main.py` reads `config.yaml`. Interpret outputs using the configuration that produced them, rather than the study name alone.

## Evaluation

Generator and discriminator losses describe training, but a lower generator loss does not necessarily mean better or more diverse images. Generated grids and class coverage provide complementary evidence.

Classifier-based Fréchet calculations use custom features. Label them accordingly rather than comparing them directly with standard Inception-feature FID scores. Where PSNR/SSIM are used, make the meaning of the real/generated pairing explicit.

No headline quality score is asserted here. Assess outputs, checkpoints and experiment records together before reporting a numerical comparison.

## Project focus

The repository combines adversarial learning, configurable output heads and model interpretation. It retains the original `dummy_*` naming for fixed runs and relies on external framework components rather than implementing every dependency independently.
