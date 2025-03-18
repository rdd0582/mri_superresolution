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
from utils.preprocessing import preprocess_slice, InterpolationMethod  # Import common preprocessing functions

import argparse

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
        # Skip slices with too much background
        if white_ratio > max_white_ratio:
            continue
        
        # Process slice
        processed_slice = preprocess_slice(
            slice_data, 
            target_size=target_size,
            interpolation=InterpolationMethod.CUBIC
        )
        
        # Generate filename
        filename = generate_filename(subject, idx, timepoint)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_slice)
        print(f"Saved: {output_path}")

def extract_slices(nifti_file, output_dir, n_slices=10,
                   lower_percent=0.2, upper_percent=0.8, target_size=None):
    """
    Load a NIfTI file and extract slices from a full‐resolution scan.
    Supports both 3D and 4D data.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    subject = os.path.splitext(os.path.basename(nifti_file))[0]

    if data.ndim == 3:
        extract_slices_3d(data, subject, output_dir,
                          n_slices=n_slices, lower_percent=lower_percent,
                          upper_percent=upper_percent, target_size=target_size)
    elif data.ndim == 4:
        num_timepoints = data.shape[3]
        print(f"Processing 4D file with {num_timepoints} time points: {nifti_file}")
        for t in range(num_timepoints):
            data_3d = data[:, :, :, t]
            extract_slices_3d(data_3d, subject, output_dir, timepoint=t,
                              n_slices=n_slices, lower_percent=lower_percent,
                              upper_percent=upper_percent, target_size=target_size)
    else:
        print(f"Unexpected data dimensionality for {nifti_file}: {data.ndim}D")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract full-resolution anatomical slices from NIfTI scans."
    )
    parser.add_argument('--datasets_dir', type=str, default='./datasets', 
                        help='Directory containing dataset subfolders')
    parser.add_argument('--output_dir', type=str, default='./training_data', 
                        help='Output directory for full-resolution slices')
    parser.add_argument('--n_slices', type=int, default=10, 
                        help='Number of slices to extract per volume')
    parser.add_argument('--lower_percent', type=float, default=0.2, 
                        help='Lower percentile for slice selection')
    parser.add_argument('--upper_percent', type=float, default=0.8, 
                        help='Upper percentile for slice selection')

    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
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
                            extract_slices(nifti_path, output_dir,
                                           n_slices=args.n_slices,
                                           lower_percent=args.lower_percent,
                                           upper_percent=args.upper_percent)
                        except Exception as e:
                            print(f"Error processing {nifti_path}: {e}")
