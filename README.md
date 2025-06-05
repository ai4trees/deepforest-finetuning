# DeepForest Finetuning

A Python package for fine-tuning the [DeepForest](https://github.com/weecology/DeepForest) model on custom data for tree detection in aerial imagery. This project provides a streamlined workflow for preprocessing training data, fine-tuning the DeepForest model, making predictions, and evaluating results.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
  - [Using Conda](#using-conda)
  - [Using Docker](#using-docker)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Fine-tuning](#2-model-fine-tuning)
  - [3. Making Predictions](#3-making-predictions)
  - [4. Evaluating Results](#4-evaluating-results)
- [Configuration Files](#configuration-files)
- [Example Usage](#example-usage)
- [Advanced Features](#advanced-features)
- [License](#license)

## Overview

DeepForest is a deep learning model designed for detecting trees in aerial RGB imagery. This package extends DeepForest by providing a framework to fine-tune the model on your own datasets. Key features include:

- Data preprocessing for various input formats
- Automatic label projection from 3D point clouds to 2D orthophotos
- Image rescaling and tiling
- Model fine-tuning with multiple random seeds for robust evaluation
- Prediction on new images with customizable tiling
- Evaluation metrics calculation (precision, recall, F1 score)
- Support for experiment tracking with [Weights & Biases](https://wandb.ai/)

## Installation

### Using Conda

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/deepforest-finetuning.git
   cd deepforest-finetuning
   ```

2. Create and activate a conda environment from the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   conda activate deepforest-env
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Using Docker

A Dockerfile is provided for containerized usage:

```bash
docker build -t deepforest-finetuning .
docker run --gpus all -it -v /path/to/your/data:/data deepforest-finetuning /bin/bash
```

## Project Structure

```
deepforest-finetuning/
├── configs/              # Configuration files for different workflows
│   ├── baseline/         # Configurations for the baseline model
│   ├── finetuning/       # Configurations for fine-tuning
│   └── preprocessing/    # Configurations for data preprocessing
├── scripts/              # Executable scripts for main workflows
│   ├── evaluate.py       # Script for model evaluation
│   ├── finetuning.py     # Script for fine-tuning
│   ├── prediction.py     # Script for making predictions
│   └── preprocessing.py  # Script for data preprocessing
└── src/                  # Source code
    └── deepforest_finetuning/
        ├── config/       # Configuration dataclasses
        ├── evaluation/   # Model evaluation logic
        ├── prediction/   # Prediction logic
        ├── preprocessing/# Data preprocessing logic
        ├── training/     # Model training and fine-tuning logic
        └── utils/        # Utility functions
```

## Workflow

### 1. Data Preprocessing

The package supports multiple preprocessing steps:

#### a. Projecting Labels from Point Clouds

If you have 3D point cloud data with tree positions, you can project them to 2D bounding boxes:

```bash
python scripts/preprocessing.py configs/preprocessing/project_point_cloud_labels.toml
```

Required configuration:
```toml
base_dir = "/path/to/your/data"
point_cloud_paths = ["pointcloud1.las", "pointcloud2.las"]
image_paths = ["image1.tif", "image2.tif"]
label_json_output_paths = ["labels1.json", "labels2.json"]
```

#### b. Preprocessing Manually Corrected Labels

Convert manually created or corrected labels to the format required by DeepForest:

```bash
python scripts/preprocessing.py configs/preprocessing/preprocess_manually_corrected_labels.toml
```

#### c. Filtering Labels

Filter out unwanted labels based on size, position, etc.:

```bash
python scripts/preprocessing.py configs/preprocessing/filter_labels.toml
```

#### d. Image Rescaling

Rescale images to different resolutions:

```bash
python scripts/preprocessing.py configs/preprocessing/rescale_images.toml
```

### 2. Model Fine-tuning

Fine-tune the DeepForest model on your custom dataset:

```bash
python scripts/finetuning.py configs/finetuning/finetuning_5_cm_manual_labeling_small.toml
```

Example configuration:
```toml
base_dir = "/path/to/your/data"
tmp_dir = "./tmp"
patch_size = 640
patch_overlap = 0.2
image_folder = "images"
train_annotation_files = ["annotations"]
test_annotation_files = ["test_annotations"]
epochs = 20
seeds = [0, 1, 2, 3, 4]
learning_rate = 0.0001
checkpoint_dir = "checkpoints"
early_stopping_patience = 2
save_top_k = 1
target_metric = "val_f1"
```

This will:
1. Split images into patches
2. Create training and test datasets
3. Fine-tune the model for the specified number of epochs
4. Run with multiple random seeds for robust evaluation
5. Save checkpoints and logs

### 3. Making Predictions

Make predictions with the fine-tuned model:

```bash
python scripts/prediction.py configs/finetuning/predict_finetuned_5_cm.toml
```

Example configuration:
```toml
checkpoint_path = "/path/to/checkpoint.pt"
image_files = ["/path/to/image.tif"]
predict_tile = true
patch_size = 1000
patch_overlap = 0.2

[prediction_export]
output_folder = "/path/to/predictions"
output_file_name = "predictions.csv"
```

### 4. Evaluating Results

Evaluate predictions against ground truth:

```bash
python scripts/evaluate.py configs/evaluation/evaluate_finetuned_5_cm.toml
```

Example configuration:
```toml
prediction_file = "/path/to/predictions.csv"
label_file = "/path/to/ground_truth.csv"
iou_threshold = 0.4
output_file = "/path/to/evaluation_results.csv"
```

## Configuration Files

All workflows are configured using TOML files:

- **Preprocessing configs**: Define data paths and parameters for preprocessing steps
- **Training configs**: Specify training hyperparameters, data paths, and evaluation settings
- **Prediction configs**: Set model checkpoint, input images, and output format
- **Evaluation configs**: Define prediction and ground truth file paths, and evaluation metrics

## Example Usage

### Complete Fine-tuning Pipeline Example

1. Preprocess your data (e.g., project point cloud labels, rescale images)
2. Fine-tune the model:
   ```bash
   python scripts/finetuning.py configs/finetuning/my_finetuning_config.toml
   ```
3. Make predictions with the fine-tuned model:
   ```bash
   python scripts/prediction.py configs/prediction/my_prediction_config.toml
   ```
4. Evaluate the results:
   ```bash
   python scripts/evaluate.py configs/evaluation/my_evaluation_config.toml
   ```

### Using Different Image Resolutions

The package supports working with images at various resolutions. The config directories contain subdirectories for different resolutions (e.g., `2_5_cm`, `5_cm`, `7_5_cm`, `10_cm`).

## Advanced Features

### Experiment Tracking with Weights & Biases

The fine-tuning process supports integration with Weights & Biases for experiment tracking:

```toml
# Add to your fine-tuning config
use_wandb = true
wandb_project = "deepforest-finetuning"
wandb_entity = "your-wandb-username"
```

This will log metrics, hyperparameters, and validation results to your W&B account.

### Multiple Random Seeds

To ensure robust evaluation, you can run fine-tuning with multiple random seeds:

```toml
seeds = [0, 1, 2, 3, 4]
```

This will train separate models with different weight initializations and report the average performance.

### Early Stopping and Model Checkpointing

To prevent overfitting, you can enable early stopping and control model checkpointing:

```toml
early_stopping_patience = 2  # Stop training if performance doesn't improve for this many epochs
save_top_k = 1  # Save the top k best models based on the target metric
target_metric = "val_f1"  # Metric to monitor for early stopping and checkpointing
```

The `mode` (min/max) is automatically inferred from the metric name. Metrics containing "loss" use "min" mode (lower is better), all others use "max" mode (higher is better).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
