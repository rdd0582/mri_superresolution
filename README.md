# MRI Superresolution Project

This project leverages paired full-resolution and downsampled MRI images to train a **Convolutional Neural Network (CNN)** that enhances low-quality (simulated 1.5T) scans into high-quality (3T) images.

## Features
- Extracts 2D slices from 3D **NIfTI** MRI images.
- Downsamples MRI scans to simulate lower resolution.
- Trains a **CNN model** to enhance low-resolution images.
- Performs inference to generate super-resolved MRI scans.

---

## Project Structure
```
project_root/
├── dataset/                   # Original 3T NIfTI images.
├── training_data/             # PNG slices extracted from full-resolution images.
├── training_data_1.5T/        # PNG slices extracted from downsampled images.
├── models/                    # Contains the CNN model (see `cnn_model.py`).
├── scripts/                   # Main scripts for extraction, downsampling, training, and inference.
│   ├── extract_full_res.py    # Extract slices from full-resolution NIfTI files.
│   ├── downsample_extract.py  # Downsample NIfTI files and extract slices from them.
│   ├── train.py               # Train the CNN model using paired data.
│   └── infer.py               # Run inference with the trained model.
├── utils/                     # Dataset loader for paired images.
├── launch.py                  # Launcher script to trigger different actions.
└── requirements.txt           # Required Python packages.
```

---

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mri-superresolution.git
cd mri-superresolution
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- Python **3.7+**
- PyTorch & torchvision
- nibabel
- matplotlib
- numpy
- Pillow
- argparse

---

## Usage
A single **launcher script (`launch.py`)** provides easy access to different functions.

### Extract Full-Resolution Slices
```bash
python launch.py --action extract_full
```
This script:
- Traverses the `./dataset/` directory for **NIfTI** files.
- Extracts **10** equally spaced slices from the center of each 3D volume (default).
- Saves the slices as **PNG images** in `./training_data/`.

### Downsample and Extract Slices
```bash
python launch.py --action downsample
```
This script:
- Downsamples **3D NIfTI** volumes to simulate **1.5T MRI scans**.
- Extracts slices from the downsampled images.
- Saves PNG images in `./training_data_1.5T/`.

### Train the Model
```bash
python launch.py --action train
```
This script:
- Loads paired PNG images from `./training_data/` (high-res) and `./training_data_1.5T/` (low-res).
- Trains a **CNN model** using **MSE loss** and the **Adam optimizer**.
- Saves the trained model’s checkpoint in `./checkpoints/`.

#### Customize Training Parameters
```bash
python launch.py --action train --epochs 20 --batch_size 16 --learning_rate 0.001
```

### Run Inference
```bash
python launch.py --action infer --input_image <path_to_low_res_png>
```
This script:
- Loads the trained model from `./checkpoints/cnn_superres.pth`.
- Processes the low-resolution input image.
- Generates a super-resolved image (`output.png`).

#### Customize Inference Parameters
```bash
python launch.py --action infer --model_path ./checkpoints/cnn_superres.pth --input_image ./training_data_1.5T/sample.png --output_image result.png
```

---

## Customization

### Modify the Model Architecture
Edit `models/cnn_model.py` to experiment with different **CNN architectures**.

### Adjust Data Handling
Modify `utils/dataset.py` to customize how paired images are **loaded** or **augmented**.

### Change Slicing Parameters
Both extraction scripts allow customization of:
- **Number of slices** (`n_slices`)
- **Volume range** (`lower_percent`, `upper_percent`)

---

## License
[AGPL v3]

---

## Contact
For any questions or feedback, please reach out to rupertd909@gmail.com.

