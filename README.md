## Fine-Tuning DeepForest for Forest Tree Detection in High-Resolution UAV Imagery

A Python package for fine-tuning the [DeepForest](https://github.com/weecology/DeepForest) model on custom data. DeepForest is a deep learning model for detecting trees in aerial RGB imagery. This package extends DeepForest by providing a workflow to fine-tune the model on your own datasets. Key features include:

- Data preprocessing for various input formats
- Automatic label projection from 3D point clouds to 2D orthophotos
- Image rescaling and tiling
- Model fine-tuning with multiple random seeds for robust evaluation
- Prediction on new images with customizable tiling
- Evaluation metrics calculation (precision, recall, F1 score)

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
docker run --gpus all --rm -it -v /path/to/your/data:/workspace/data/ deepforest-finetuning
```

## Workflow

### 1. Data Preprocessing

The package supports multiple preprocessing steps:

#### a. Projecting Labels from Point Clouds

If you have 3D point cloud data with pointwise tree instance labels in addition to 2D aerial images, you can project them to 2D bounding boxes:

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

#### b. Filtering Labels

Filters labels using non-maximum suppression based on overlap and size:

```bash
python scripts/preprocessing.py configs/preprocessing/filter_labels.toml
```

Required configuration:
```toml
base_dir = "/path/to/your/data"
input_label_folder = "labels"
output_label_folder = "labels_filtered"
iou_threshold = 0.5
```

#### c. Image Rescaling

Rescale images and corresponding labels to different resolutions:

```bash
python scripts/preprocessing.py configs/preprocessing/rescale_images.toml
```

Required configuration:
```toml
base_dir = "/path/to/your/data"
# input_images can either be a list of individual file paths or string specifying a folder path
input_images = ["image1.tif", "image2.tif"]
# if no labels are available, input_label_folders can be left empty
input_label_folders = ["labels"]
# there must be one output folder for each target resolution
output_folders = ["rescaled_2_5_cm", "rescaled_5_cm"]
target_resolutions = [0.025, 0.05]
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
target_metric = "test_f1"
```

This will:
1. Split images into patches and load training and test datasets
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

All workflows are configured using TOML files. Example configurations are provided in the `configs` folder.

## Other Features

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
target_metric = "test_f1"  # Metric to monitor for early stopping and checkpointing
```

The `mode` (min/max) is automatically inferred from the metric name. Metrics containing "loss" use "min" mode (lower is better), all others use "max" mode (higher is better).

### How to Cite

If you use our code, please consider citing our paper:

```
@article{Burmeister_FineTuning_DeepForest_2025,
author = {Burmeister, Josafat-Mattias and Zabbarov, Julian and Reder, Stefan and Richter, Rico and Mund, Jan-Peter and Döllner, Jürgen},
doi = {n/a},
journal = {ISPRS Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
pages = {n/a},
title = {{Fine-Tuning DeepForest for Forest Tree Detection in High-Resolution UAV Imagery}},
volume = {n/a},
year = {2025}
}
```
