# MRI Superresolution Project

This project upscales 1.5 Tesla MRI scans to appear as if they were taken with higher-resolution scanners (3T or 7T) using Deep Learning models implemented in PyTorch.

## Features

- **Dataset Extraction**: Extract full-resolution and downsampled (simulated 1.5T) datasets from NIfTI files
- **Training**: Train models using paired high-resolution and simulated low-resolution MRI slices with validation, early stopping, and learning rate scheduling
- **Inference**: Perform super-resolution on new MRI images with quality metrics (PSNR, SSIM) and visual comparison
- **Models Available**:
  - **CNNSuperRes**: A simple CNN with residual connections
  - **EDSRSuperRes**: An enhanced deep super-resolution model with residual blocks and optional upscaling
  - **UNetSuperRes**: A U-Net architecture with skip connections, particularly effective for medical imaging

## Prerequisites

- Python 3.8+
- Required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- For Windows users, install curses support:
  ```bash
  pip install windows-curses
  ```

## Dataset

The MRI data used in this project is from:

- **Title**: raw_HC001-HC005.tar.gz
- **Created by**: Jessica Royer
- **Published on**: OSF by Center for Open Science
- **Publication Year**: 2021
- **Direct URL**: https://osf.io/4vfas
- **Note**: The raw data is part of a larger dataset available at https://osf.io/z8mu5

## Project Structure

```
mri_superresolution/
├── __init__.py
├── launch.py                # Launcher for curses-based UI
├── models/
│   ├── __init__.py
│   ├── cnn_model.py         # Simple CNN Model
│   ├── edsr_model.py        # EDSR Model
│   └── unet_model.py        # U-Net Model
├── scripts/
│   ├── __init__.py
│   ├── downsample_extract.py # Simulate 1.5T data and extract slices
│   ├── extract_full_res.py   # Extract full-resolution slices
│   ├── infer.py              # Inference script
│   └── train.py              # Training script
└── utils/
    ├── __init__.py
    ├── dataset.py            # Dataset management and augmentation
    ├── losses.py             # Custom loss functions and metrics
    └── preprocessing.py      # Image preprocessing utilities
```

## Usage

You can either use the command-line interface or the curses-based launcher.

### Option 1: Using the Launcher

To start the curses-based launcher:

```bash
python launch.py
```

The launcher provides the following menu options:

1. **Extract Full-Resolution Dataset**: Extracts high-resolution slices
2. **Extract Downsampled Dataset**: Simulates 1.5T images and extracts slices
3. **Train Model**: Trains a model (Simple CNN, EDSR, or U-Net)
4. **Infer Image**: Runs inference on a low-resolution image
5. **Exit**: Exits the launcher

### Option 2: Using Command-Line Interface

You can also run scripts directly using the command line.

#### 1. Extract Full-Resolution Dataset

```bash
python scripts/extract_full_res.py --datasets_dir ./datasets --output_dir ./training_data
```

#### 2. Extract Downsampled Dataset

```bash
python scripts/downsample_extract.py --datasets_dir ./datasets --output_dir ./training_data_1.5T
```

#### 3. Train Model

To train the Simple CNN model:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T --model_type simple --validation_split 0.2 --ssim_weight 0.5
```

To train the EDSR model:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T --model_type edsr --scale 2 --validation_split 0.2 --ssim_weight 0.5
```

To train the U-Net model:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T --model_type unet --base_filters 64 --validation_split 0.2 --ssim_weight 0.5
```

#### 4. Infer Image

To run inference using the Simple CNN model:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png --model_type simple --show_comparison
```

To run inference using the EDSR model:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png --model_type edsr --scale 2 --show_comparison
```

To run inference using the U-Net model:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png --model_type unet --show_comparison
```

To evaluate against a ground truth image:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png --target_image ./target.png --model_type simple --show_comparison
```

## Models Overview

### 1. CNNSuperRes

- A basic CNN architecture with residual connections
- Fast and suitable for real-time applications
- Lightweight with minimal parameters

### 2. EDSRSuperRes

- Advanced architecture with multiple residual blocks
- Optional upsampling using PixelShuffle
- High-quality super-resolution suitable for medical imaging

### 3. UNetSuperRes

- U-Net architecture with encoder-decoder structure and skip connections
- Particularly effective for medical imaging tasks
- Captures both local and global features through multi-scale processing

## Loss Function and Metrics

The project uses:

- **CombinedLoss**: A weighted combination of:
  - L1 Loss (Mean Absolute Error) for pixel-wise accuracy
  - SSIM (Structural Similarity Index) for perceptual quality
- **Evaluation Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)

## Training Improvements

The training process now includes:

- **Validation Split**: A portion of the dataset is reserved for validation
- **Early Stopping**: Training stops when validation loss stops improving
- **Learning Rate Scheduling**: Learning rate is reduced when validation loss plateaus
- **Model Checkpointing**: The best model based on validation loss is saved
- **Mixed Precision Training**: Uses PyTorch's AMP for faster training
- **SSIM Loss**: Combines L1 loss with SSIM for better perceptual quality

## Inference Improvements

The inference process now includes:

- **Quality Metrics**: PSNR and SSIM metrics when a target image is provided
- **Visual Comparison**: Side-by-side comparison of input, output, and target images
- **Best Model Selection**: Automatically uses the best model checkpoint if available

## Augmentation Techniques

- Horizontal flipping
- Small random rotations (-5 to 5 degrees)
- Brightness and contrast adjustments

## Checkpoints

Trained model checkpoints are saved in the `./checkpoints` directory:

- `cnn.pth`: Checkpoint for the Simple CNN model
- `edsr.pth`: Checkpoint for the EDSR model
- `unet.pth`: Checkpoint for the U-Net model
- `best_*.pth`: Best model checkpoints based on validation loss

## Contribution

Contributions are welcome! If you find a bug or want to add a feature, feel free to submit a pull request.

## Citation

If you use this project or its components, please cite the dataset:

```
@misc{Royer2021MRIData,
  author = {Jessica Royer},
  title = {raw_HC001-HC005.tar.gz},
  year = {2021},
  url = {https://osf.io/4vfas},
  publisher = {OSF},
  organization = {Center for Open Science}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
