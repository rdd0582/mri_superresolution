# MRI Super-Resolution Tool

## Description

This project provides a suite of tools for training and evaluating a U-Net based deep learning model for Magnetic Resonance Imaging (MRI) super-resolution. The primary goal is to enhance the quality of low-resolution MRI slices by learning a mapping to their high-resolution counterparts.

The project includes utilities for:
*   Extracting paired high-resolution (HR) and simulated low-resolution (LR) slices from NIfTI datasets.
*   Training a U-Net model with configurable loss functions (L1, SSIM, Perceptual) and hyperparameters.
*   Running inference (super-resolution) on new images using a trained model.
*   Evaluating model performance using metrics like SSIM and PSNR.
*   Comparing model results against traditional interpolation methods.
*   A text-based User Interface (UI) to manage these tasks.

## Features

*   **Data Preparation:** Extracts paired 2D slices from 3D/4D NIfTI volumes (`scripts/extract_paired_slices.py`). Simulates low-resolution data using k-space cropping and Rician noise modeling.
*   **Model Training:** Trains a U-Net model (`models/unet_model.py`) using the extracted paired slices (`scripts/train.py`). Supports configurable loss components (L1, SSIM, VGG Perceptual Loss), learning rate scheduling, early stopping, and data augmentation.
*   **Inference:** Applies a trained model to enhance a low-resolution input image (`scripts/infer.py`).
*   **Evaluation & Comparison:**
    *   Evaluates a trained model on a test dataset (`scripts/test_model.py`).
    *   Compares the AI model's output against standard interpolation methods (Bilinear, Bicubic) (`scripts/test_comparison.py`).
    *   Utilities to test and compare models trained with different loss weightings (`scripts/test_ssim_weights.py`, `scripts/compare_ssim_detailed.py`).
*   **Visualization:** Generates visual comparisons of input, output, and target images, including difference maps (`scripts/infer.py`, `scripts/test_model.py`, `scripts/test_comparison.py`). Includes a utility to visualize dataset resolutions (`utils/visualise_res.py`).
*   **User Interface:** Provides a convenient text-based UI (`scripts/ui.py`) to run the extraction, training, and inference pipelines without manually invoking scripts with command-line arguments.

## Directory Structure

```
/
|-- models/                 # Contains model definitions (e.g., unet_model.py)
|-- scripts/                # Executable Python scripts for UI, training, inference, testing, etc.
|-- utils/                  # Utility functions for preprocessing, losses, dataset handling, etc.
|-- datasets/               # (Expected) Directory to store input NIfTI datasets (e.g., ./datasets/set1/sub-01/anat/T1w.nii.gz)
|-- training_data/          # (Default) Output directory for extracted high-resolution slices
|-- training_data_1.5T/     # (Default) Output directory for extracted low-resolution slices
|-- checkpoints/            # (Default) Directory to save trained model checkpoints
|-- logs/                   # (Default) Directory for TensorBoard logs
|-- test_results/           # (Default) Output directory for test scripts
|-- requirements.txt        # Python package dependencies
|-- README.txt              # This file
|-- *.log                   # Log files generated by scripts (e.g., ui.log, training.log)
```

## Setup & Installation

1.  **Prerequisites:** Python 3.8+ is recommended.
2.  **Clone Repository:** `git clone <repository_url>` (if applicable)
3.  **Install Dependencies:** Navigate to the project root directory in your terminal and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  **PyTorch with CUDA (Optional):** If you have an NVIDIA GPU and want CUDA acceleration, you might need a specific PyTorch version. The `requirements.txt` file contains a commented-out example for CUDA 11.8. Uninstall any existing PyTorch CPU version first (`pip uninstall torch torchvision torchaudio`) and then install the GPU version, e.g.:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    Adjust the CUDA version (`cu118`, `cu121`, etc.) based on your system's CUDA toolkit installation. Check the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your setup.

## Usage

### Using the Text-based UI (Recommended)

The easiest way to use the tool is through the provided UI:

```bash
python scripts/ui.py
```

This will launch an interactive terminal interface where you can:
*   Configure paths for datasets and outputs.
*   Run the **Extract Paired Slices** process.
*   Configure training parameters (model type, hyperparameters, loss weights, etc.).
*   Run the **Train Super-Resolution Model** process.
*   Select a trained checkpoint.
*   Configure inference parameters.
*   Run **Infer on Image** to enhance a specific low-resolution image.

Navigate the UI using the arrow keys (Up/Down), Enter to select or modify options, and 'Q' to quit.

### Using Individual Scripts (Advanced)

Each script in the `scripts/` directory can also be run directly from the command line with arguments. Use the `-h` or `--help` flag to see available options for each script.

**Examples:**

*   **Extract Data:**
    ```bash
    python scripts/extract_paired_slices.py --datasets_dir ./datasets --hr_output_dir ./training_data --lr_output_dir ./training_data_1.5T --n_slices 20
    ```
*   **Train Model:**
    ```bash
    python scripts/train.py --full_res_dir ./training_data --low_res_dir ./training_data_1.5T --model_type unet --epochs 50 --batch_size 4 --ssim_weight 0.4 --perceptual_weight 0.1 --use_amp
    ```
*   **Run Inference:**
    ```bash
    python scripts/infer.py --input ./path/to/low_res.png --output ./output_enhanced.png --model_type unet --checkpoint_dir ./checkpoints --show_comparison
    ```

## Configuration

Hyperparameters, file paths, and other settings can be configured either:
*   Through the options presented in the Text UI (`scripts/ui.py`).
*   Via command-line arguments when running individual scripts (`scripts/*.py`). See `--help` for each script. 