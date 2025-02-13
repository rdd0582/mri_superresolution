MRI Superresolution Project
This project leverages paired full-resolution and downsampled MRI images to train a Convolutional Neural Network (CNN) that enhances low-quality (simulated 1.5T) scans into high-quality (3T) images.

The repository includes utilities to extract image slices from 3D NIfTI data, downsample images to simulate lower resolution, train a CNN model, and run inference with the trained model.

Project Structure
graphql
Copy
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
├── utils/                     # Contains the dataset loader for paired images.
├── launch.py                  # Launcher script to trigger different actions.
└── requirements.txt           # Required Python packages.
Installation
Install the necessary Python packages. For example:

bash
Copy
pip install -r requirements.txt
The requirements typically include:

Python 3.7+
PyTorch & torchvision
nibabel
matplotlib
numpy
Pillow
argparse
Usage
A single launcher script (launch.py) is provided to conveniently access different actions in the project. The available actions are:

extract_full: Extracts PNG slices from full-resolution NIfTI images.
downsample: Downsamples the full-resolution NIfTI images (to simulate 1.5T scans) and extracts corresponding PNG slices.
train: Trains the CNN superresolution model using paired data (full-resolution and downsampled images).
infer: Performs inference using the trained model on a low-resolution input image.
Running via the Launcher
Use the launcher to perform the desired action. The launcher will pass any additional command-line arguments directly to the underlying script.

Extract Full-Resolution Slices:

bash
Copy
python launch.py --action extract_full
This script (scripts/extract_full_res.py) will:

Traverse the ./dataset directory for NIfTI files.
Extract a specified number of equally spaced slices (default is 10) from the central portion of each 3D volume.
Save the slices as PNG images in the ./training_data directory.
Downsample and Extract Slices:

bash
Copy
python launch.py --action downsample
This script (scripts/downsample_extract.py) will:

Traverse the ./dataset directory for NIfTI files.
Downsample each 3D volume to simulate a 1.5T scan (using a target voxel size, e.g., (2.0, 2.0, 2.0)).
Extract a specified number of slices from the downsampled images.
Save the resulting PNG images in the ./training_data_1.5T directory.
Train the Model:

bash
Copy
python launch.py --action train
This script (scripts/train.py) will:

Load paired PNG images from ./training_data (full-resolution) and ./training_data_1.5T (downsampled).
Train a CNN model (defined in models/cnn_model.py) using an MSE loss and the Adam optimizer.
Save the trained model’s state dictionary (default filename cnn_superres.pth) in the ./checkpoints directory.
You can pass additional arguments (like --epochs, --batch_size, etc.) directly through the launcher. For example:

bash
Copy
python launch.py --action train --epochs 20 --batch_size 16 --learning_rate 0.001
Run Inference:

bash
Copy
python launch.py --action infer --input_image <path_to_low_res_png>
This script (scripts/infer.py) will:

Load the trained model from a specified checkpoint (default: ./checkpoints/cnn_superres.pth).
Process the provided low-resolution input PNG image.
Generate and save a high-resolution output image (default: output.png).
Display the output image using matplotlib.
You can also specify custom paths for the model checkpoint or output image, for example:

bash
Copy
python launch.py --action infer --model_path ./checkpoints/cnn_superres.pth --input_image ./training_data_1.5T/sample.png --output_image result.png
Customization
Model Architecture:
Modify models/cnn_model.py to experiment with different network designs.

Dataset Handling:
Adjust utils/dataset.py to change how paired images are loaded or augmented.

Slicing Parameters:
Both extraction scripts allow customization of the number of slices and the portion of the volume to consider (via n_slices, lower_percent, and upper_percent).

License
[Insert your license here]

Contact
For any questions or feedback, please reach out to [your contact info].