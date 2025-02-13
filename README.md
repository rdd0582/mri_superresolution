# MRI Superresolution Project

This project uses paired full-resolution and downsampled MRI images to train a CNN that enhances low-quality (1.5T–like) scans into high-quality (3T) images.

**Project Structure:**

- **dataset/**: Contains the original 3T NIfTI images.
- **training_data/**: PNG slices extracted from full-resolution images.
- **training_data_1.5T/**: PNG slices extracted from downsampled images.
- **models/**: Contains the CNN model.
- **scripts/**: Includes extraction, downsampling, training, and inference scripts.
- **utils/**: Contains the dataset loader for paired images.
- **launch.py**: A simple launcher for various actions.
- **requirements.txt**: Required Python packages.

**Usage:**

1. **Extract full-resolution slices:**  
   `python launch.py --action extract_full`

2. **Extract downsampled slices:**  
   `python launch.py --action downsample`

3. **Train the model:**  
   `python launch.py --action train`

4. **Run inference:**  
   `python launch.py --action infer --input_image <path_to_low_res_png>`
