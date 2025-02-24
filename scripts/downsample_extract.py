#!/usr/bin/env python
import os
import sys

# Update sys.path so that the project root is included before importing other modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import nibabel as nib
import numpy as np
import cv2  # For image resizing and saving
from scipy.ndimage import gaussian_filter
from utils.preprocessing import preprocess_slice  # Reuse the preprocessing function

def simulate_15T_data(data, noise_std=5, blur_sigma=0.5):
    """
    Simulate a 1.5T image from high-quality data by applying Gaussian blur and adding
    Rician-like noise.

    The noise_std and blur_sigma defaults have been further reduced to better resemble
    a 1.5T scan.
    """
    # Apply Gaussian blur to mimic a smoother appearance
    blurred = gaussian_filter(data, sigma=blur_sigma)
    
    # Create Rician-like noise by combining two independent Gaussian noise fields.
    noise1 = np.random.normal(0, noise_std, data.shape)
    noise2 = np.random.normal(0, noise_std, data.shape)
    simulated = np.sqrt((blurred + noise1)**2 + noise2**2)
    return simulated

def generate_filename(subject, slice_idx, timepoint=None):
    """
    Generate a filename with the following format:
      SubjectName[_T{timepoint}]_s{slice_idx:03d}.png
    (No modality suffix is added so that paired images have identical names.)
    """
    if timepoint is not None:
        return f"{subject}_T{timepoint}_s{slice_idx:03d}.png"
    else:
        return f"{subject}_s{slice_idx:03d}.png"

def extract_slices_3d(data, subject, output_dir, timepoint=None,
                      n_slices=10, lower_percent=0.2, upper_percent=0.8, target_size=None):
    """
    Extract n_slices equally spaced from the central portion of a 3D volume,
    preprocess (robust normalization, letterbox resize preserving full resolution),
    and save each slice.
    """
    num_slices = data.shape[2]
    lower_index = int(lower_percent * num_slices)
    upper_index = int(upper_percent * num_slices)
    slice_indices = np.linspace(lower_index, upper_index, n_slices, dtype=int)

    for idx in slice_indices:
        slice_data = data[:, :, idx]
        processed_slice = preprocess_slice(slice_data, target_size=target_size)
        filename = generate_filename(subject, idx, timepoint)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_slice)
        print(f"Saved: {output_path}")

def extract_slices(nifti_file, output_dir, n_slices=10,
                   lower_percent=0.2, upper_percent=0.8, target_size=None,
                   noise_std=5, blur_sigma=0.5):
    """
    Load a NIfTI file, simulate a 1.5T appearance, and extract slices.
    Supports both 3D and 4D data.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    subject = os.path.splitext(os.path.basename(nifti_file))[0]

    if data.ndim == 3:
        sim_data = simulate_15T_data(data, noise_std=noise_std, blur_sigma=blur_sigma)
        extract_slices_3d(sim_data, subject, output_dir,
                          n_slices=n_slices, lower_percent=lower_percent,
                          upper_percent=upper_percent, target_size=target_size)
    elif data.ndim == 4:
        num_timepoints = data.shape[3]
        print(f"Processing 4D file with {num_timepoints} time points: {nifti_file}")
        for t in range(num_timepoints):
            data_3d = data[:, :, :, t]
            sim_data_3d = simulate_15T_data(data_3d, noise_std=noise_std, blur_sigma=blur_sigma)
            extract_slices_3d(sim_data_3d, subject, output_dir, timepoint=t,
                              n_slices=n_slices, lower_percent=lower_percent,
                              upper_percent=upper_percent, target_size=target_size)
    else:
        print(f"Unexpected data dimensionality for {nifti_file}: {data.ndim}D")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulate 1.5T images from high-resolution NIfTI scans and extract slices."
    )
    parser.add_argument('--datasets_dir', type=str, default='./datasets', 
                        help='Directory containing dataset subfolders')
    parser.add_argument('--output_dir', type=str, default='./training_data_1.5T', 
                        help='Output directory for simulated 1.5T slices')
    parser.add_argument('--n_slices', type=int, default=10, 
                        help='Number of slices to extract per volume')
    parser.add_argument('--lower_percent', type=float, default=0.2, 
                        help='Lower percentile for slice selection')
    parser.add_argument('--upper_percent', type=float, default=0.8, 
                        help='Upper percentile for slice selection')
    # Set target_size default to None for full-resolution extraction.
    parser.add_argument('--target_size', type=int, nargs=2, default=None, 
                        help='Target size for resizing slices (width height), set to None for full resolution')
    # Further reduced defaults for noise and blur
    parser.add_argument('--noise_std', type=float, default=5, 
                        help='Standard deviation for noise')
    parser.add_argument('--blur_sigma', type=float, default=0.5, 
                        help='Sigma for Gaussian blur')

    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for set_name in os.listdir(datasets_dir):
        set_path = os.path.join(datasets_dir, set_name)
        if os.path.isdir(set_path):
            print(f"Processing dataset: {set_name}")
            for root, dirs, files in os.walk(set_path):
                # Only process directories named "anat"
                if os.path.basename(root).lower() != "anat":
                    continue
                for file in files:
                    if file.endswith('.nii') or file.endswith('.nii.gz'):
                        nifti_path = os.path.join(root, file)
                        print(f"Processing {nifti_path}")
                        try:
                            extract_slices(nifti_path, output_dir,
                                           n_slices=args.n_slices,
                                           target_size=args.target_size,
                                           lower_percent=args.lower_percent,
                                           upper_percent=args.upper_percent,
                                           noise_std=args.noise_std,
                                           blur_sigma=args.blur_sigma)
                        except Exception as e:
                            print(f"Error processing {nifti_path}: {e}")
