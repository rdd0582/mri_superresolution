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
                      output_dir: str, 
                      timepoint: Optional[int] = None,
                      n_slices: int = 10, 
                      lower_percent: float = 0.2, 
                      upper_percent: float = 0.8, 
                      target_size: Tuple[int, int] = (320, 240),
                      preprocess_function=None):
    """
    Extract n_slices equally spaced from the central portion of a 3D volume,
    preprocess (using the provided preprocessing function),
    and save each slice.
    
    Args:
        data: 3D numpy array with volume data
        subject: Subject identifier for filename generation
        output_dir: Directory to save processed slices
        timepoint: Optional timepoint for 4D data
        n_slices: Number of slices to extract
        lower_percent: Lower percentile of slices to consider
        upper_percent: Upper percentile of slices to consider
        target_size: Target size for resizing as (width, height)
        preprocess_function: Function to preprocess each slice (should return uint8 array)
    """
    # If no preprocessing function provided, raise an error
    if preprocess_function is None:
        raise ValueError("A preprocessing function must be provided")
    
    num_slices = data.shape[2]
    lower_index = int(lower_percent * num_slices)
    upper_index = int(upper_percent * num_slices)
    slice_indices = np.linspace(lower_index, upper_index, n_slices, dtype=int)

    for idx in slice_indices:
        slice_data = data[:, :, idx]
        
        # Process slice using the provided function
        processed_slice = preprocess_function(slice_data, target_size)
        
        # Generate filename
        filename = generate_filename(subject, idx, timepoint)
        output_path = os.path.join(output_dir, filename)
        
        # Save the processed slice
        import cv2  # Import here to avoid circular imports
        cv2.imwrite(output_path, processed_slice)
        print(f"Saved: {output_path}") 