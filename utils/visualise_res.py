

import os
import nibabel as nib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt # Keep for histogram plotting
import pandas as pd
from pathlib import Path
from PIL import Image # Import Pillow

def find_nifti_files(root_dir="./datasets"):
    """
    Recursively scan the directory structure and find NIfTI files in 'anat' folders.

    Args:
        root_dir (str): The root directory to start scanning from.

    Returns:
        list: A list of paths to NIfTI files in 'anat' folders.
    """
    nifti_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current directory is named 'anat'
        if os.path.basename(dirpath) == 'anat':
            # Filter for NIfTI files
            for filename in filenames:
                if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                    nifti_files.append(os.path.join(dirpath, filename))

    return nifti_files

def extract_middle_slice(nifti_file, output_dir="./png_slices"):
    """
    Extract the middle slice from a NIfTI file and save it as a PNG
    with its original dimensions, without resizing.

    Args:
        nifti_file (str): Path to the NIfTI file.
        output_dir (str): Directory to save the PNG slices to.

    Returns:
        tuple: The resolution (width, height) of the extracted slice, or None if error.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the NIfTI file
        img = nib.load(nifti_file)
        # Ensure data is loaded into memory if it's a proxy using get_fdata
        # REMOVED: img.get_data() # This line caused the ExpiredDeprecationError
        data = img.get_fdata(dtype=np.float32) # Use float32 for calculations

        # Handle potential 4D data (e.g., fMRI) by taking the first volume
        if data.ndim > 3:
            print(f"Warning: NIfTI file {nifti_file} has {data.ndim} dimensions. Using first volume.")
            data = data[..., 0]
        elif data.ndim < 3:
             print(f"Warning: NIfTI file {nifti_file} has less than 3 dimensions ({data.ndim}). Skipping.")
             return None

        # Get the middle slice index (along z-axis/third dimension)
        if data.shape[2] == 0:
            print(f"Warning: NIfTI file {nifti_file} has zero size along the 3rd dimension. Skipping.")
            return None
        middle_slice_idx = data.shape[2] // 2

        # Extract the middle slice - preserving exact dimensions
        middle_slice = data[:, :, middle_slice_idx]

        # Get the original dimensions (width, height) - note numpy shape is (height, width)
        height, width = middle_slice.shape[0], middle_slice.shape[1]

        # Check for empty slices
        if middle_slice.size == 0 or width == 0 or height == 0:
            print(f"Warning: Extracted middle slice from {nifti_file} is empty or has zero dimensions ({width}x{height}). Skipping.")
            return None

        # Normalize to 0-255 for PNG (without changing dimensions)
        slice_min = middle_slice.min()
        slice_max = middle_slice.max()
        if slice_max == slice_min: # Handle constant value slices
             slice_normalized = np.zeros_like(middle_slice, dtype=np.uint8)
        else:
            # Add epsilon to prevent division by zero if max == min somehow missed
            slice_normalized = ((middle_slice - slice_min) /
                               (slice_max - slice_min + 1e-10) * 255).astype(np.uint8)


        # Create a filename for the PNG
        base_name = os.path.basename(nifti_file).replace('.nii.gz', '').replace('.nii', '')
        png_path = os.path.join(output_dir, f"{base_name}_slice.png")

        # --- Save directly using PIL (Pillow) ---
        # Create PIL image from numpy array (mode 'L' for grayscale)
        pil_image = Image.fromarray(slice_normalized, mode='L')

        # Save as PNG without any resizing or interpolation
        pil_image.save(png_path)
        # --- End of PIL saving ---

        # Optional: Verify the saved PNG dimensions match the original slice
        try:
            with Image.open(png_path) as saved_img:
                saved_width, saved_height = saved_img.size
                # Only print verification if dimensions differ or for debugging
                # print(f"Processed {base_name}: Original slice {width}x{height}, Saved PNG {saved_width}x{saved_height}")
                if (width, height) != (saved_width, saved_height):
                     print(f"ERROR: Dimension mismatch for {png_path}! Original: {width}x{height}, Saved: {saved_width}x{saved_height}")
                     # Decide if you want to return None or the original dimensions despite mismatch
                     # return None
        except Exception as verify_e:
            print(f"Warning: Could not verify saved PNG dimensions for {png_path}: {str(verify_e)}")


        return (width, height)  # Return the original dimensions

    except FileNotFoundError:
        print(f"Error: NIfTI file not found at {nifti_file}")
        return None
    except nib.filebasedimages.ImageFileError as nib_e:
         print(f"Error: Nibabel could not read {nifti_file}: {str(nib_e)}")
         return None
    except Exception as e:
        print(f"Error: Could not process {nifti_file}: {type(e).__name__} - {str(e)}")
        # Optionally print traceback for debugging
        # import traceback
        # traceback.print_exc()
        return None

def analyze_resolutions(resolutions):
    """
    Analyze the distribution of resolutions.

    Args:
        resolutions (list): A list of (width, height) tuples.

    Returns:
        pandas.DataFrame: A DataFrame containing the resolution frequencies.
    """
    # Filter out any None values that might have slipped through
    valid_resolutions = [res for res in resolutions if res is not None and isinstance(res, tuple) and len(res) == 2]

    if not valid_resolutions:
        print("Warning: No valid resolutions found to analyze.")
        return pd.DataFrame(columns=['Width', 'Height', 'Count'])

    # Count frequencies
    resolution_counts = Counter(valid_resolutions)

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(
        [(width, height, count) for (width, height), count in resolution_counts.items()],
        columns=['Width', 'Height', 'Count']
    )

    # Sort by frequency (descending)
    df = df.sort_values('Count', ascending=False).reset_index(drop=True)

    return df

def visualize_resolution_histogram(resolution_df, output_file="spatial_resolution_histogram.png"):
    """
    Create a detailed histogram visualization of spatial resolutions.

    Args:
        resolution_df (pandas.DataFrame): DataFrame with resolution data
        output_file (str): Path to save the visualization to
    """
    if resolution_df.empty:
        print("Cannot generate visualization: Resolution DataFrame is empty.")
        return

    # Create a more descriptive resolution label for each entry
    resolution_df['Resolution'] = resolution_df.apply(
        lambda row: f"{int(row['Width'])}×{int(row['Height'])}", axis=1
    )

    # Sort by frequency for the bar chart
    bar_df = resolution_df.sort_values('Count', ascending=False)

    # Limit the number of bars if there are too many unique resolutions for readability
    max_bars = 40
    if len(bar_df) > max_bars:
        print(f"Warning: Too many unique resolutions ({len(bar_df)}). Displaying top {max_bars} in histogram.")
        bar_df = bar_df.head(max_bars)


    # --- Bar Chart ---
    plt.figure(figsize=(max(14, len(bar_df) * 0.5), 8)) # Adjust width based on number of bars

    bars = plt.bar(
        bar_df['Resolution'],
        bar_df['Count'],
        color='steelblue',
        width=0.7,
        edgecolor='black'
    )

    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            str(int(height)),
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9 # Smaller font if many bars
        )

    plt.title('Histogram of Original Spatial Resolutions (Width × Height)', fontsize=16)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=60, ha='right', fontsize=10) # Increased rotation for potentially long labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nHistogram visualization saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving histogram visualization: {str(e)}")
    plt.close()


    # --- Scatter Plot ---
    plt.figure(figsize=(10, 8))
    # Use original DataFrame (resolution_df) for scatter plot
    scatter = plt.scatter(
        resolution_df['Width'],
        resolution_df['Height'],
        s=resolution_df['Count'] * 20, # Size represents count
        alpha=0.7,
        c=resolution_df['Count'], # Color represents count
        cmap='viridis', # Use a colormap
        edgecolors='black'
    )

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frequency (Count)', fontsize=12)

    # Optionally add labels for the most frequent points (e.g., top 5)
    top_n = 5
    # Ensure we don't try to annotate more points than exist
    num_to_annotate = min(top_n, len(resolution_df))
    for i, row in resolution_df.head(num_to_annotate).iterrows():
         # Calculate offset based on point size to avoid overlap
         offset_val = 10 + (row['Count']*20)**0.5 / 2
         plt.annotate(
             f"{int(row['Width'])}x{int(row['Height'])}\n(Count: {int(row['Count'])})",
             (row['Width'], row['Height']),
             xytext=(0, offset_val),
             textcoords='offset points',
             ha='center',
             va='bottom', # Place text above the point center
             fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6)
         )

    plt.title('Distribution of Spatial Resolutions (Width vs Height)', fontsize=16)
    plt.xlabel('Width (pixels)', fontsize=14)
    plt.ylabel('Height (pixels)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the scatter plot
    scatter_output = f"{Path(output_file).stem}_scatter.png"
    try:
        plt.savefig(scatter_output, dpi=300, bbox_inches='tight')
        print(f"Scatter visualization saved to '{scatter_output}'")
    except Exception as e:
        print(f"Error saving scatter visualization: {str(e)}")
    plt.close()


def main(root_dir="./datasets", output_png_dir="./png_slices", output_viz_file="spatial_resolution_histogram.png"):
    """
    Main function to analyze original spatial resolutions of NIfTI slices.

    Args:
        root_dir (str): The root directory to start scanning NIfTI files from.
        output_png_dir (str): Directory to save the extracted PNG slices.
        output_viz_file (str): Base filename for the output visualizations.
    """
    print(f"Scanning for NIfTI files in '{root_dir}'...")

    # Find NIfTI files in 'anat' folders
    nifti_files = find_nifti_files(root_dir)
    if not nifti_files:
        print("No NIfTI files found in 'anat' subdirectories. Exiting.")
        return
    print(f"Found {len(nifti_files)} NIfTI files in 'anat' folders.")

    # Extract middle slices and get their original resolutions
    print(f"\nExtracting middle slices to '{output_png_dir}'...")
    resolutions = []
    processed_count = 0
    error_count = 0
    for i, nifti_file in enumerate(nifti_files):
        # print(f"Processing file {i+1}/{len(nifti_files)}: {nifti_file}") # Uncomment for verbose progress
        resolution = extract_middle_slice(nifti_file, output_dir=output_png_dir)
        if resolution:
            resolutions.append(resolution)
            processed_count += 1
        else:
            error_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully extracted resolutions from {processed_count} files.")
    if error_count > 0:
        print(f"Encountered errors or skipped {error_count} files (check logs above for details).")

    if not resolutions:
        print("\nNo valid resolutions were extracted. Cannot perform analysis or visualization.")
        return

    # Analyze resolutions
    print("\nAnalyzing resolution distribution...")
    resolution_df = analyze_resolutions(resolutions)

    # Print summary table
    print("\nSpatial Resolution Distribution Summary:")
    if not resolution_df.empty:
         # Display top N resolutions or all if less than N
        top_n_display = 20
        print(resolution_df.head(top_n_display).to_string(index=False))
        if len(resolution_df) > top_n_display:
            print(f"... (showing top {top_n_display} of {len(resolution_df)} unique resolutions)")
    else:
        print("No resolution data to display.")


    # Create histogram and scatter visualizations
    print("\nGenerating visualizations...")
    visualize_resolution_histogram(resolution_df, output_file=output_viz_file)

    print("\nScript finished.")


if __name__ == "__main__":
    # Example usage: You might want to pass arguments via command line later
    # import argparse
    # parser = argparse.ArgumentParser(description="Analyze NIfTI spatial resolutions.")
    # parser.add_argument('--root_dir', type=str, default='./datasets', help='Root directory containing datasets.')
    # parser.add_argument('--png_dir', type=str, default='./png_slices', help='Directory to save extracted PNG slices.')
    # parser.add_argument('--viz_file', type=str, default='spatial_resolution_histogram.png', help='Output filename for visualizations.')
    # args = parser.parse_args()
    # main(root_dir=args.root_dir, output_png_dir=args.png_dir, output_viz_file=args.viz_file)

    main() # Use default paths for now
