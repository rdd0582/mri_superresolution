# MRI Superresolution Project

This project upscales 1.5 Tesla MRI scans to appear as if they were taken with higher-resolution scanners (3T or 7T) using Deep Learning models implemented in PyTorch.

## Features

- **Dataset Processing**: Extract full-resolution and downsampled (simulated 1.5T) datasets from NIfTI files
- **Training**: Train models with validation, early stopping, learning rate scheduling, and combined loss functions
- **Inference**: Perform super-resolution on new MRI images with quality metrics (SSIM) and visual comparison
- **Models Available**:
  - **CNNSuperRes**: Lightweight CNN with configurable residual blocks
  - **EDSRSuperRes**: Enhanced Deep Super-Resolution network with scale flexibility
  - **UNetSuperRes**: U-Net architecture optimized for medical imaging

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

### RTX GPU Optimization Requirements

For using the RTX GPU optimizations:
- NVIDIA RTX GPU (20xx series or later for best performance)
- CUDA 11.0+
- PyTorch 1.7+ for AMP (Automatic Mixed Precision)
- PyTorch 2.0+ for torch.compile
- Latest NVIDIA drivers

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
├── launch.py                # Interactive UI for all operations
├── models/
│   ├── __init__.py
│   ├── cnn_model.py         # Simple CNN with residual blocks
│   ├── edsr_model.py        # Enhanced Deep Super-Resolution
│   └── unet_model.py        # U-Net architecture
├── scripts/
│   ├── __init__.py
│   ├── downsample_extract.py # Process and simulate 1.5T data
│   ├── extract_full_res.py   # Extract high-resolution slices
│   ├── infer.py              # Inference with metrics and visualization
│   └── train.py              # Training with validation and monitoring
├── utils/
│   ├── __init__.py
│   ├── dataset.py            # Dataset management and augmentation
│   ├── losses.py             # Custom loss functions and metrics
│   └── preprocessing.py      # Image preprocessing utilities
├── checkpoints/              # Saved model weights
├── training_data/            # Full-resolution image slices
└── training_data_1.5T/       # Downsampled image slices
```

## Usage

### Interactive Launcher

The easiest way to use this project is through the interactive launcher:

```bash
python launch.py
```

The launcher provides a menu-driven interface with the following options:

1. **Extract Full-Resolution Dataset**: Process high-resolution NIfTI files
2. **Extract Downsampled Dataset**: Generate simulated 1.5T images
3. **Train Model**: Train any of the three model types with customizable parameters
4. **Infer Image**: Run super-resolution on low-resolution images
5. **Exit**: Quit the launcher

### GPU Acceleration for RTX GPUs

This project includes optimizations specifically for NVIDIA RTX GPUs:

1. **Automatic Mixed Precision (AMP)**: Uses half-precision (FP16) operations where possible to leverage Tensor Cores on RTX GPUs, resulting in 2-3x speedup.

2. **Model Compilation**: Uses PyTorch's compilation feature to further optimize execution through operation fusion and dispatch optimization.

To enable these optimizations:

```bash
# Training with RTX optimizations
python scripts/train.py --model_type unet --use_amp --use_compile

# Inference with RTX optimizations
python scripts/infer.py --model_type unet --input_image ./input.png --use_amp --use_compile
```

See [RTX Optimization Guide](docs/RTX_OPTIMIZATION.md) for more details.

### Command-Line Interface

You can also run the scripts directly with command-line arguments.

#### Data Preparation

Extract full-resolution dataset:

```bash
python scripts/extract_full_res.py --datasets_dir ./datasets --output_dir ./training_data
```

Create downsampled (simulated 1.5T) dataset:

```bash
python scripts/downsample_extract.py --datasets_dir ./datasets --output_dir ./training_data_1.5T
```

#### Training

Train the CNN model:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T \
  --model_type simple --num_blocks 8 --validation_split 0.2 --ssim_weight 0.5 \
  --batch_size 16 --epochs 50 --patience 10
```

Train the EDSR model:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T \
  --model_type edsr --scale 1 --num_res_blocks 16 --validation_split 0.2 \
  --ssim_weight 0.5 --batch_size 16 --epochs 50 --patience 10
```

Train the U-Net model:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T \
  --model_type unet --base_filters 64 --validation_split 0.2 --ssim_weight 0.5 \
  --batch_size 16 --epochs 50 --patience 10
```

Train with RTX GPU optimizations:

```bash
python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T \
  --model_type unet --base_filters 64 --validation_split 0.2 --ssim_weight 0.5 \
  --batch_size 16 --epochs 50 --patience 10 --use_amp --use_compile
```

#### Inference

Run inference with the CNN model:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png \
  --model_type simple --num_blocks 8 --show_comparison
```

Run inference with the EDSR model:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png \
  --model_type edsr --scale 1 --num_res_blocks 16 --show_comparison
```

Run inference with the U-Net model:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png \
  --model_type unet --base_filters 64 --show_comparison
```

Run inference with RTX GPU optimizations:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png \
  --model_type unet --base_filters 64 --show_comparison --use_amp --use_compile
```

To evaluate against a ground truth image:

```bash
python scripts/infer.py --input_image ./input.png --output_image ./output.png \
  --target_image ./target.png --model_type simple --show_comparison
```

## Model Details

### CNNSuperRes

- Configurable number of residual blocks
- Efficient architecture with fewer parameters
- Direct image-to-image mapping without upsampling

### EDSRSuperRes

- Based on Enhanced Deep Super-Resolution Network
- Configurable upscaling factor and number of residual blocks
- Optimized for detail preservation in medical images

### UNetSuperRes

- Encoder-decoder architecture with skip connections
- Configurable base filter count for capacity adjustment
- Effective at capturing local and global features

## Training Features

- **Combined Loss Function**: Weighted combination of L1 loss and SSIM
- **Validation Split**: Automatic train/validation dataset separation
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Scheduling**: Reduces learning rate when progress plateaus
- **Model Checkpointing**: Saves the best model based on validation metrics
- **Training Progress**: Real-time visualization of training metrics

## Inference Features

- **Quality Metrics**: SSIM calculations against target images
- **Visual Comparison**: Side-by-side comparison of input, output, and target
- **Automatic Model Loading**: Uses the best saved checkpoint

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
