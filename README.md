# MRI Super-Resolution Tool

This project provides a pipeline for Magnetic Resonance Imaging (MRI) super-resolution, focusing on enhancing the quality of lower-resolution MRI scans using a deep learning model. It includes scripts for data preparation (extracting paired high-resolution and simulated low-resolution slices), model training, and inference, along with a convenient terminal-based user interface (TUI).

## Features

*   **Data Preparation:**
    *   Extracts 2D slices from 3D/4D NIfTI files (`.nii`, `.nii.gz`).
    *   Supports BIDS (Brain Imaging Data Structure) filename conventions for subject identification.
    *   Generates paired datasets: high-resolution (HR) slices and corresponding simulated low-resolution (LR) slices.
    *   **Low-Resolution Simulation:** Uses k-space manipulation (cropping) and Rician noise addition for realistic simulation of lower field strength (e.g., 1.5T from 3T). An older method using Gaussian blur + noise is also available.
    *   **Preprocessing:** Includes robust normalization (percentile clipping), resizing (letterboxing, cropping, stretching), padding, and histogram equalization options.
*   **Model:**
    *   Implements a **U-Net** based architecture (`UNetSuperRes`) specifically for image quality enhancement (preserves input resolution).
    *   Uses Kaiming He initialization for convolutional layers.
*   **Training:**
    *   Trains the U-Net model on paired HR/LR slices.
    *   **Loss Function:** Uses a `CombinedLoss` incorporating L1 loss, Structural Similarity Index Measure (SSIM) loss, and optional VGG-based Perceptual Loss. Weights for each component are adjustable.
    *   Supports data augmentation (flips, rotations, brightness/contrast adjustments, noise).
    *   Validation loop with early stopping based on validation loss (`ReduceLROnPlateau` scheduler).
    *   Checkpoint saving (best model based on validation loss, final model).
    *   Optional TensorBoard logging for monitoring training progress.
    *   Automatic Mixed Precision (AMP) support for faster training on compatible GPUs.
    *   Optimized data loading using `DataLoader` with configurable workers.
*   **Inference:**
    *   Runs the trained model on new (low-resolution) input images.
    *   Preprocesses input images consistently with training data.
    *   Saves the enhanced output image.
    *   Optionally compares the output with a ground truth (high-resolution) image.
    *   Calculates evaluation metrics: SSIM, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
    *   Visualizes results: Input, Output, Ground Truth (optional), and Absolute Difference map (optional).
*   **Terminal UI (TUI):**
    *   A curses-based interface (`scripts/ui.py`) provides easy access to:
        *   Paired Slice Extraction
        *   Model Training
        *   Inference
    *   Allows configuration of parameters for each step directly in the UI.
    *   Displays progress and status messages.
*   **Utilities:**
    *   Scripts to test different SSIM weightings during training (`scripts/test_ssim_weights.py`) and compare the results (`scripts/compare_ssim_detailed.py`).
    *   Script to analyze and visualize the spatial resolution distribution in datasets (`utils/visualise_res.py`).
*   **Logging:** Comprehensive logging for UI actions (`ui.log`), training (`training.log`), and inference (`inference.log`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd mri_superresolution
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    venv\\Scripts\\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, you might need to create it based on the imports in the Python files. Key libraries include `torch`, `torchvision`, `numpy`, `nibabel`, `opencv-python`, `matplotlib`, `pandas`, `scipy`, `Pillow`, `colorama` (for Windows TUI colors), `windows-curses` (Windows only).)*

## Directory Structure

```
mri_superresolution/
│
├── datasets/             # Input NIfTI files (organized, e.g., by BIDS)
├── training_data/        # Output directory for extracted HR slices
├── training_data_1.5T/   # Output directory for simulated LR slices
├── checkpoints/          # Saved model checkpoints during training
│   ├── samples/          # Example image comparisons saved during training
├── logs/                 # Default directory for training logs (if not in checkpoints)
├── models/               # Model definitions (e.g., unet_model.py)
│   └── __init__.py
├── scripts/              # Executable scripts (UI, train, infer, extract)
│   ├── __init__.py
│   ├── ui.py             # Terminal User Interface
│   ├── train.py          # Training script
│   ├── infer.py          # Inference script
│   ├── extract_paired_slices.py # Data extraction/simulation script
│   ├── compare_ssim_detailed.py # Utility script
│   └── test_ssim_weights.py     # Utility script
├── utils/                # Utility functions (preprocessing, dataset, losses, etc.)
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── losses.py
│   ├── extraction_utils.py
│   └── visualise_res.py
├── README.md             # This file
├── requirements.txt      # Project dependencies
├── ui.log                # Log file for the TUI
├── training.log          # Log file for training runs
└── inference.log         # Log file for inference runs
```

## Usage

You can run the different steps of the pipeline either using the individual scripts or the integrated Terminal UI.

### 1. Terminal UI (Recommended)

The easiest way to use the tool is via the TUI:

```bash
python scripts/ui.py
```

Navigate using arrow keys and Enter. The UI guides you through:
*   **Extract Paired Slices:** Configure input/output directories, number of slices, simulation parameters, and run the extraction.
*   **Train Super-Resolution Model:** Select model type (currently UNet), configure training parameters (data directories, epochs, batch size, learning rate, loss weights, etc.), select checkpoint directory, and start training.
*   **Infer on Image:** Select the input image, the trained model checkpoint, output path, and optional target image for comparison.

### 2. Individual Scripts

#### a) Data Preparation (`extract_paired_slices.py`)

This script processes NIfTI files from an input directory (`--datasets_dir`), extracts slices, preprocesses them, and saves paired HR and LR images to specified output directories.

```bash
python scripts/extract_paired_slices.py \
    --datasets_dir ./datasets \
    --hr_output_dir ./training_data \
    --lr_output_dir ./training_data_1.5T \
    --n_slices 10 \
    --target_size 256 256 \
    --noise_std 5 \
    --kspace_crop_factor 0.5 \
    --use_kspace_simulation
```

*   Key arguments:
    *   `--datasets_dir`: Root directory containing NIfTI files (expects `anat` subfolders).
    *   `--hr_output_dir`: Where to save high-resolution PNGs.
    *   `--lr_output_dir`: Where to save simulated low-resolution PNGs.
    *   `--n_slices`: Number of slices to extract per volume.
    *   `--lower_percent`, `--upper_percent`: Range of slices to consider (e.g., 0.2 to 0.8 excludes noisy end slices).
    *   `--target_size`: Resize slices to this `width height` (e.g., 256 256). Uses letterboxing by default.
    *   `--use_kspace_simulation`: Flag to use k-space cropping for LR simulation (recommended).
    *   `--kspace_crop_factor`: Amount of k-space center to keep (e.g., 0.5 means 50%).
    *   `--noise_std`: Standard deviation for simulated Rician noise (relative to 0-255 range).
    *   `--blur_sigma`: Sigma for Gaussian blur if *not* using k-space simulation.

#### b) Training (`train.py`)

This script trains the U-Net model using the paired HR/LR datasets.

```bash
python scripts/train.py \
    --full_res_dir ./training_data \
    --low_res_dir ./training_data_1.5T \
    --model_type unet \
    --checkpoint_dir ./checkpoints/unet_run1 \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --ssim_weight 0.7 \
    --perceptual_weight 0.0 \
    --num_workers 4 \
    --use_amp \
    --use_tensorboard
```

*   Key arguments:
    *   `--full_res_dir`, `--low_res_dir`: Paths to the prepared HR and LR datasets.
    *   `--model_type`: Currently only `unet` is supported.
    *   `--base_filters`: Number of filters in the first U-Net layer (default: 32).
    *   `--checkpoint_dir`: Directory to save model checkpoints and sample images.
    *   `--epochs`: Number of training epochs.
    *   `--batch_size`: Number of samples per batch.
    *   `--learning_rate`: Initial learning rate for the Adam optimizer.
    *   `--weight_decay`: L2 regularization factor.
    *   `--ssim_weight`: Weight for the (1 - SSIM) loss component (0 to 1).
    *   `--perceptual_weight`: Weight for the Perceptual loss component (0 to 1). L1 weight is `1 - ssim_weight - perceptual_weight`.
    *   `--vgg_layer_idx`: VGG layer index for Perceptual loss (default: 35 for VGG19 `relu5_4`).
    *   `--perceptual_loss_type`: Type of loss for perceptual features (`l1` or `mse`, default: `l1`).
    *   `--validation_split`: Fraction of data to use for validation (default: 0.2).
    *   `--patience`: Epochs to wait for validation loss improvement before reducing LR (default: 10).
    *   `--num_workers`: Number of parallel data loading processes.
    *   `--seed`: Random seed for reproducibility.
    *   `--augmentation`: Flag to enable data augmentation.
    *   `--use_tensorboard`: Flag to enable TensorBoard logging (requires `tensorboard` installed).
    *   `--use_amp`: Flag to enable Automatic Mixed Precision.
    *   `--cpu`: Flag to force CPU usage even if CUDA is available.
    *   `--checkpoint_file`: Path to a specific checkpoint file to resume training from.
    *   `--log_dir`: Directory for TensorBoard logs (defaults to inside `checkpoint_dir`).

#### c) Inference (`infer.py`)

This script applies a trained model to a single input image.

```bash
python scripts/infer.py \
    --input_image ./path/to/low_res_input.png \
    --output_image ./output/super_res_output.png \
    --checkpoint_path ./checkpoints/unet_run1/best_model_unet.pth \
    --model_type unet \
    --target_image ./path/to/high_res_ground_truth.png \
    --show_comparison \
    --show_diff
```

*   Key arguments:
    *   `--input_image`: Path to the low-resolution input PNG image.
    *   `--output_image`: Path to save the super-resolved output PNG image.
    *   `--checkpoint_path`: Path to the trained model checkpoint (`.pth` file). Can also be a directory, in which case it looks for `best_model_*.pth` or `final_model_*.pth`.
    *   `--model_type`: Model type corresponding to the checkpoint (e.g., `unet`).
    *   `--base_filters`: Base filters used for the loaded UNet model (if not standard 64, specify).
    *   `--target_image`: (Optional) Path to the corresponding high-resolution ground truth image for comparison and metrics calculation.
    *   `--show_comparison`: Flag to display a plot comparing Input, Output, and Target (if provided).
    *   `--show_diff`: Flag to include an absolute difference map in the comparison plot (requires `--target_image`).
    *   `--cpu`: Flag to force CPU usage.
    *   `--use_amp`: Flag to use AMP during inference (may improve speed on some GPUs).

## Requirements

*   Python 3.7+
*   PyTorch (check `requirements.txt` for version)
*   Torchvision
*   NumPy
*   Nibabel (for NIfTI I/O)
*   OpenCV (`opencv-python`)
*   Matplotlib
*   Pandas
*   SciPy
*   Pillow (PIL Fork)
*   Colorama (for Windows TUI colors)
*   `windows-curses` (Required for TUI **only on Windows**)
*   TensorBoard (optional, for logging)
*   CUDA-capable GPU (recommended for training and faster inference)

See `requirements.txt` for specific versions.

## License

MIT License - see the LICENSE file for details (assuming one exists).
