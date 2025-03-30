#!/usr/bin/env python
import os
import re
import numpy as np
from typing import Tuple, Optional, List

def generate_bids_identifier(nifti_file: str) -> str:
    """
    Extract a robust identifier from a NIfTI filename, 
    parsing BIDS-like entities (sub, ses, task, acq, run, etc.).
    
    Args:
        nifti_file: Path to the NIfTI file
        
    Returns:
        A unique identifier string derived from the filename
    """
    # Get the filename without the full path
    basename = os.path.basename(nifti_file)
    
    # Remove the extension(s)
    # Handle both .nii and .nii.gz extensions
    if basename.endswith('.nii.gz'):
        basename = basename[:-7]
    elif basename.endswith('.nii'):
        basename = basename[:-4]
    
    # For BIDS formatted filenames (e.g., sub-01_ses-1_acq-MPRAGE_T1w.nii.gz)
    # Extract all entities using a regex pattern
    # BIDS entities are in the format key-value
    bids_entities = re.findall(r'([a-zA-Z0-9]+)-([a-zA-Z0-9]+)', basename)
    
    # If BIDS entities were found, create an identifier 
    if bids_entities:
        # Join all entities to form the identifier
        # Filter out modality entities (like T1w, T2w) which typically come after underscores
        # These are not typically in key-value format and are handled separately
        base_id = '_'.join([f"{key}-{value}" for key, value in bids_entities])
        
        # Look for modality suffix (typically after the last underscore)
        modality_match = re.search(r'_([A-Za-z0-9]+)$', basename)
        if modality_match:
            modality = modality_match.group(1)
            # Only add modality if it's a standard MRI modality identifier
            if modality in ['T1w', 'T2w', 'FLAIR', 'BOLD', 'PD', 'PDw', 'DWI']:
                base_id += f"_{modality}"
                
        return base_id
    
    # For non-BIDS filenames, just return the basename without extension
    return basename

def generate_filename(subject: str, slice_idx: int, timepoint: Optional[int] = None) -> str:
    """
    Generate a filename with the following format:
      SubjectName[_T{timepoint}]_s{slice_idx:03d}.png
    (No modality suffix is added so that paired images have identical names.)
    
    Args:
        subject: Subject identifier
        slice_idx: Slice index
        timepoint: Optional timepoint for 4D data
        
    Returns:
        Formatted filename string
    """
    if timepoint is not None:
        return f"{subject}_T{timepoint}_s{slice_idx:03d}.png"
    else:
        return f"{subject}_s{slice_idx:03d}.png"

def extract_slices_3d(data: np.ndarray, 
                      subject: str, 
                      hr_output_dir: str,
                      lr_output_dir: Optional[str] = None,
                      timepoint: Optional[int] = None,
                      n_slices: int = 10, 
                      lower_percent: float = 0.2, 
                      upper_percent: float = 0.8, 
                      target_size: Tuple[int, int] = (320, 240),
                      preprocess_function=None,
                      apply_simulation: bool = False,
                      noise_std: float = 5.0,
                      blur_sigma: float = 0.5):
    """
    Extract n_slices equally spaced from the central portion of a 3D volume,
    preprocess (using the provided preprocessing function),
    and save each slice in both high-resolution and optionally low-resolution formats.
    
    Args:
        data: 3D numpy array with volume data
        subject: Subject identifier for filename generation
        hr_output_dir: Directory to save high-resolution processed slices
        lr_output_dir: Directory to save low-resolution processed slices (if None, only save HR)
        timepoint: Optional timepoint for 4D data
        n_slices: Number of slices to extract
        lower_percent: Lower percentile of slices to consider
        upper_percent: Upper percentile of slices to consider
        target_size: Target size for resizing as (width, height)
        preprocess_function: Function to preprocess each slice (should accept apply_simulation parameter)
        apply_simulation: Whether to apply low-resolution simulation for LR images
        noise_std: Noise standard deviation for simulation
        blur_sigma: Sigma for Gaussian blur in simulation
    """
    # If no preprocessing function provided, raise an error
    if preprocess_function is None:
        raise ValueError("A preprocessing function must be provided")
    
    num_slices = data.shape[2]
    lower_index = int(lower_percent * num_slices)
    upper_index = int(upper_percent * num_slices)
    slice_indices = np.linspace(lower_index, upper_index, n_slices, dtype=int)

    import cv2  # Import here to avoid circular imports
    
    for idx in slice_indices:
        slice_data = data[:, :, idx]
        
        # Process slice for high-resolution (no simulation)
        hr_processed_slice = preprocess_function(slice_data, target_size, 
                                               apply_simulation=False)
        
        # Generate filename (same for both HR and LR to maintain pairing)
        filename = generate_filename(subject, idx, timepoint)
        
        # Save the high-resolution slice
        hr_output_path = os.path.join(hr_output_dir, filename)
        # Convert from float [0,1] to uint8 [0,255] for saving
        hr_processed_uint8 = (hr_processed_slice * 255).astype(np.uint8)
        cv2.imwrite(hr_output_path, hr_processed_uint8)
        print(f"Saved HR: {hr_output_path}")
        
        # If low-resolution output directory is provided, create LR version too
        if lr_output_dir is not None:
            # Process slice for low-resolution (with simulation)
            lr_processed_slice = preprocess_function(slice_data, target_size, 
                                                  apply_simulation=apply_simulation,
                                                  noise_std=noise_std,
                                                  blur_sigma=blur_sigma)
            
            # Save the low-resolution slice
            lr_output_path = os.path.join(lr_output_dir, filename)
            # Convert from float [0,1] to uint8 [0,255] for saving
            lr_processed_uint8 = (lr_processed_slice * 255).astype(np.uint8)
            cv2.imwrite(lr_output_path, lr_processed_uint8)
            print(f"Saved LR: {lr_output_path}") 