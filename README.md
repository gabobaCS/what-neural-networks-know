# What Neural Networks Know: Uncovering Latent Knowledge in Deep Vision Models
<p align="center">
<img src="figures/experiment-pipeline-v3.png" width="90%" />
</p>

Official implementation of the paper **"What Neural Networks Know: Uncovering Latent Knowledge in Deep Vision Models"**.  
We develop **PEEK**, a fully data-free probing method that uses logit-maximization–based prototype synthesis to generate class-level representations, which are then compared in feature space to infer latent semantic knowledge.


## Overview
An integral part of the paper is the generation of class prototypes. We use DeepDream-based prototypes for neural probing and we employ DeepInversion prototypes to construct synthetic reference sets that significantly strengthen OOD detection.


### DeepDream
DeepDream synthesizes an image $\hat{x}$ by maximizing a chosen network activation (e.g., a class logit) while suppressing high-frequency artifacts:

$$
\max_{\hat{x}}\; L(\hat{x}) - R(\hat{x}), 
$$

<p align="center">
  <img src="figures/selected_imagenet_samples-1.png" width="15%" />
  <img src="figures/selected_deepdream_samples-1.png" width="15%" />
</p>

The resulting images lack discernible visual features, yet the network still interprets them in a way that preserves their underlying semantic meaning.

### DeepInversion
DeepInversion extends DeepDream by enforcing that synthesized images match the feature statistics stored in batch-normalization layers, encouraging realistic intermediate activations:

$$
R_{\text{feature}}(\hat{x}) =
\sum_{l}
\left\|
\mu_l(\hat{x}) - \mathbb{E}[\mu_l(x)]
\right\|_2^2
+
\left\|
\sigma_l^2(\hat{x}) - \mathbb{E}[\sigma_l^2(x)]
\right\|_2^2 .
$$

<p align="center">
  <img src="figures/selected_imagenet_samples-1.png" width="15%" />
  <img src="figures/selected_deepinversion_samples-1.png" width="15%" />
</p>



## Installation

```bash
pip install -r requirements.txt
```

## Usage

### OOD Detection Methods

All OOD detection methods use YAML configuration files for dataset and model specifications.

**Maximum Softmax Probability (MSP):**
```bash
python -m wnnnk_core.ood_methods.msp --config path/to/config.yaml
```

**Energy-based Detection:**
```bash
python -m wnnnk_core.ood_methods.energy --config path/to/config.yaml --temperature 1.0
```

**ODIN:**
```bash
python -m wnnnk_core.ood_methods.odin
```

**KNN-based Methods:**
```bash
# Using ImageNet as reference bank
python -m wnnnk_core.ood_methods.knn --config path/to/config.yaml --k 7 --samples_per_class 10

# Using deep inversion features
python -m wnnnk_core.ood_methods.deep_inversion_knn --config path/to/config.yaml --k 7
```

### Configuration Format

Example YAML configuration:
```yaml
model_name: "resnet50"
id_dataset:
  name: "ImageNet"
  logits: "path/to/id_logits.pt"
  avgpool: "path/to/id_activations.pt"
ood_datasets:
  - name: "iNaturalist"
    logits: "path/to/ood_logits.pt"
    avgpool: "path/to/ood_activations.pt"
  - name: "SUN"
    logits: "path/to/ood_logits.pt"
    avgpool: "path/to/ood_activations.pt"
```
