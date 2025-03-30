#!/usr/bin/env python
import os
import sys

# Update sys.path so that the project root is included before importing other modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import nibabel as nib
import numpy as np
import cv2  # Still needed if any cv2-specific operations are required
from utils.preprocessing import preprocess_slice, InterpolationMethod, ResizeMethod  # Import common preprocessing functions
from utils.extraction_utils import generate_bids_identifier, generate_filename, extract_slices_3d

import argparse

def preprocess_high_res_slice(slice_data, target_size=(320, 240), 
                           apply_simulation=False, noise_std=5, blur_sigma=0.5):
    """
    Wrapper function for preprocessing slices.
    Can generate either high-resolution or simulated low-resolution slices.
    """
    processed_slice = preprocess_slice(
        slice_data, 
        target_size=target_size,
        interpolation=InterpolationMethod.CUBIC,
        resize_method=ResizeMethod.LETTERBOX,
        apply_simulation=apply_simulation,
        noise_std=noise_std,
        blur_sigma=blur_sigma
    )
    
    # Return float [0,1] image (conversion to uint8 happens in extract_slices_3d)
    return processed_slice

def extract_slices(nifti_file, hr_output_dir, lr_output_dir,
                   n_slices=10, lower_percent=0.2, upper_percent=0.8, 
                   target_size=(320, 240), noise_std=5, blur_sigma=0.5):
    """
    Load a NIfTI file and extract slices from a full‐resolution scan.
    Creates both high-resolution and low-resolution versions.
    Supports both 3D and 4D data.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    subject = generate_bids_identifier(nifti_file)

    if data.ndim == 3:
        extract_slices_3d(data, subject, hr_output_dir, lr_output_dir,
                          n_slices=n_slices, lower_percent=lower_percent,
                          upper_percent=upper_percent, target_size=target_size,
                          preprocess_function=preprocess_high_res_slice,
                          apply_simulation=True,
                          noise_std=noise_std, blur_sigma=blur_sigma)
    elif data.ndim == 4:
        num_timepoints = data.shape[3]
        print(f"Processing 4D file with {num_timepoints} time points: {nifti_file}")
        for t in range(num_timepoints):
            data_3d = data[:, :, :, t]
            extract_slices_3d(data_3d, subject, hr_output_dir, lr_output_dir,
                              timepoint=t, n_slices=n_slices,
                              lower_percent=lower_percent,
                              upper_percent=upper_percent, target_size=target_size,
                              preprocess_function=preprocess_high_res_slice,
                              apply_simulation=True,
                              noise_std=noise_std, blur_sigma=blur_sigma)
    else:
        print(f"Unexpected data dimensionality for {nifti_file}: {data.ndim}D")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract both full-resolution and simulated low-resolution slices from NIfTI scans."
    )
    parser.add_argument('--datasets_dir', type=str, default='./datasets', 
                        help='Directory containing dataset subfolders')
    parser.add_argument('--hr_output_dir', type=str, default='./training_data', 
                        help='Output directory for high-resolution slices')
    parser.add_argument('--lr_output_dir', type=str, default='./training_data_1.5T', 
                        help='Output directory for simulated low-resolution slices')
    parser.add_argument('--n_slices', type=int, default=10, 
                        help='Number of slices to extract per volume')
    parser.add_argument('--lower_percent', type=float, default=0.2, 
                        help='Lower percentile for slice selection')
    parser.add_argument('--upper_percent', type=float, default=0.8, 
                        help='Upper percentile for slice selection')
    # Set target_size default to 320x240 for consistent image dimensions
    parser.add_argument('--target_size', type=int, nargs=2, default=[320, 240], 
                        help='Target size for resizing slices (width height), default is 320x240')
    # Simulation parameters
    parser.add_argument('--noise_std', type=float, default=5, 
                       help='Standard deviation for noise (for 0-255 range, internally scaled)')
    parser.add_argument('--blur_sigma', type=float, default=0.5, 
                       help='Sigma for Gaussian blur (default: 0.5)')


    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    hr_output_dir = args.hr_output_dir
    lr_output_dir = args.lr_output_dir
    
    # Create output directories
    os.makedirs(hr_output_dir, exist_ok=True)
    if lr_output_dir:
        os.makedirs(lr_output_dir, exist_ok=True)
    
    print("=== MRI Paired Slice Extraction ===")
    print(f"Datasets Directory: {datasets_dir}")
    print(f"High-Resolution Output: {hr_output_dir}")
    if lr_output_dir:
        print(f"Low-Resolution Output: {lr_output_dir}")
        print(f"Simulation Settings:")
        print(f"  - Noise Standard Deviation: {args.noise_std}")
        print(f"  - Gaussian Blur Sigma: {args.blur_sigma}")
    else:
        print("Low-Resolution Simulation: Disabled")
    print("===================================")
    
    # Loop over dataset folders (e.g., set1, set2, ...)
    for set_name in os.listdir(datasets_dir):
        set_path = os.path.join(datasets_dir, set_name)
        if os.path.isdir(set_path):
            print(f"Processing dataset: {set_name}")
            # Walk through the entire directory tree
            for root, dirs, files in os.walk(set_path):
                # Process only directories named "anat"
                if os.path.basename(root).lower() != "anat":
                    continue
                for file in files:
                    if file.endswith('.nii') or file.endswith('.nii.gz'):
                        nifti_path = os.path.join(root, file)
                        print(f"Processing {nifti_path}")
                        try:
                            extract_slices(nifti_path, hr_output_dir, lr_output_dir,
                                           n_slices=args.n_slices,
                                           lower_percent=args.lower_percent,
                                           upper_percent=args.upper_percent,
                                           target_size=tuple(args.target_size),
                                           noise_std=args.noise_std,
                                           blur_sigma=args.blur_sigma)
                        except Exception as e:
                            print(f"Error processing {nifti_path}: {e}") 