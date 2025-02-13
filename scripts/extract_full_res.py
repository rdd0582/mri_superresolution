#!/usr/bin/env python
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def extract_slices(nifti_file, output_dir, n_slices=10, lower_percent=0.2, upper_percent=0.8):
    """
    Load a NIfTI file, extract n_slices equally spaced slices from the central portion,
    and save each slice as a PNG in output_dir.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()

    if data.ndim == 4:
        print(f"Skipping 4D file: {nifti_file}")
        return
    if data.ndim != 3:
        print(f"Unexpected data dimensionality for {nifti_file}: {data.ndim}D")
        return

    num_slices = data.shape[2]
    lower_index = int(lower_percent * num_slices)
    upper_index = int(upper_percent * num_slices)
    slice_indices = np.linspace(lower_index, upper_index, n_slices, dtype=int)

    base_name = os.path.splitext(os.path.basename(nifti_file))[0]
    for idx in slice_indices:
        slice_data = data[:, :, idx]
        output_filename = f"{base_name}_slice{idx}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.imshow(slice_data.T, cmap='gray', origin='lower')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    dataset_dir = './dataset'
    output_dir = './training_data'
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifti_path = os.path.join(root, file)
                print(f"Processing {nifti_path}")
                try:
                    extract_slices(nifti_path, output_dir, n_slices=10)
                except Exception as e:
                    print(f"Error processing {nifti_path}: {e}")
