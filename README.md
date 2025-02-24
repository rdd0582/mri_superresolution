# MRI Superresolution Project

This project upscales 1.5 Tesla MRI scans to appear as if they were taken with higher-resolution scanners (3T or 7T) using Deep Learning models implemented in PyTorch.

## Features

- **Dataset Extraction**: Extract full-resolution and downsampled (simulated 1.5T) datasets from NIfTI files
- **Training**: Train models using paired high-resolution and simulated low-resolution MRI slices
- **Inference**: Perform super-resolution on new MRI images
- **Models Available**:
  - **CNNSuperRes**: A simple CNN with residual connections
  - **EDSRSuperRes**: An enhanced deep super-resolution model with residual blocks and optional upscaling

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
mri_cnn_project/
├── __init__.py
├── launch.py                # Launcher for curses-based UI
├── models/
│   ├── __init__.py
│   ├── cnn_model.py         # Simple CNN Model
│   └── edsr_model.py        # EDSR Model
├── scripts/
│   ├── __init__.py
│   ├── downsample_extract.py # Simulate 1.5T data and extract slices
│   ├── extract_full_res.py   # Extract full-resolution slices
│   ├── infer.py              # Inference script
│   └── train.py              # Training script
└── utils/
    ├── __init__.py
    ├── dataset.py            # Dataset management and augmentation
    ├── losses.py             # Custom loss function (L1 loss)
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
3. **Train Model**: Trains either the Simple CNN or EDSR model
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
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T --model_type simple
```

To train the EDSR model:
```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T --model_type edsr --scale 2
```

#### 4. Infer Image

To run inference using the Simple CNN model:
```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png --model_type simple
```

To run inference using the EDSR model:
```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png --model_type edsr --scale 2
```

## Models Overview

### 1. CNNSuperRes
- A basic CNN architecture with residual connections
- Fast and suitable for real-time applications

### 2. EDSRSuperRes
- Advanced architecture with multiple residual blocks
- Optional upsampling using PixelShuffle
- High-quality super-resolution suitable for medical imaging

## Loss Function

The project uses CombinedLoss which includes:
- L1 Loss (Mean Absolute Error)
- SSIM (Structural Similarity Index) is not currently used but can be easily re-enabled

## Augmentation Techniques

- Horizontal flipping
- Small random rotations (-5 to 5 degrees)
- Brightness and contrast adjustments

## Checkpoints

Trained model checkpoints are saved in the `./checkpoints` directory:
- `cnn.pth`: Checkpoint for the Simple CNN model
- `edsr.pth`: Checkpoint for the EDSR model

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