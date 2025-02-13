#!/usr/bin/env python
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_to_output

def downsample_image(nifti_file, output_dir, target_voxel=(2.0, 2.0, 2.0)):
    """
    Downsample a 3D NIfTI image to target_voxel size and save it with a _1.5T suffix.
    """
    img = nib.load(nifti_file)
    if img.ndim != 3:
        print(f"Skipping non-3D file: {nifti_file}")
        return None
    resampled_img = resample_to_output(img, voxel_sizes=target_voxel)
    base_name = os.path.splitext(os.path.basename(nifti_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_1.5T.nii.gz")
    nib.save(resampled_img, output_file)
    print(f"Saved downsampled image: {output_file}")
    return output_file

def extract_slices_from_downsampled(ds_nifti_file, orig_nifti_file, output_dir, n_slices=10, lower_percent=0.2, upper_percent=0.8):
    """
    Extract slices from the downsampled image (ds_nifti_file) but name them using slice indices
    computed from the original full-resolution image (orig_nifti_file).
    """
    import os, numpy as np, matplotlib.pyplot as plt, nibabel as nib

    # Load the original (full-resolution) image to get the full number of slices.
    orig_img = nib.load(orig_nifti_file)
    orig_data = orig_img.get_fdata()
    full_num_slices = orig_data.shape[2]

    # Load the downsampled image.
    ds_img = nib.load(ds_nifti_file)
    ds_data = ds_img.get_fdata()
    ds_num_slices = ds_data.shape[2]

    # Compute desired slice indices based on the full-resolution image.
    lower_index_full = int(lower_percent * full_num_slices)
    upper_index_full = int(upper_percent * full_num_slices)
    full_slice_indices = np.linspace(lower_index_full, upper_index_full, n_slices, dtype=int)

    # Use the original file's base name.
    base_name = os.path.splitext(os.path.basename(orig_nifti_file))[0]

    for full_idx in full_slice_indices:
        # Map the full-res index to an index in the downsampled image.
        # This computes the proportional index and rounds it.
        ds_idx = int(round(full_idx * ds_num_slices / full_num_slices))
        ds_idx = min(ds_idx, ds_num_slices - 1)  # ensure we don't go out of bounds

        slice_data = ds_data[:, :, ds_idx]
        output_filename = f"{base_name}_slice{full_idx}.png"
        output_path = os.path.join(output_dir, output_filename)

        # (Optional) flip the slice horizontally if desired.
        flipped_slice = np.fliplr(slice_data.T)
        plt.imshow(flipped_slice, cmap='gray', origin='lower')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved downsampled slice as: {output_path}")

        


if __name__ == '__main__':
    import os
    import nibabel as nib
    os.makedirs('./downsampled', exist_ok=True)
    os.makedirs('./training_data_1.5T', exist_ok=True)
    
    for root, dirs, files in os.walk('./dataset'):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                orig_path = os.path.join(root, file)
                print(f"Processing {orig_path}")
                ds_file = downsample_image(orig_path, './downsampled', target_voxel=(2.0, 2.0, 2.0))
                if ds_file is not None:
                    extract_slices_from_downsampled(ds_file, orig_path, './training_data_1.5T', n_slices=10)

