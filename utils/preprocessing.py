# File: utils/preprocessing.py
import cv2
import numpy as np
import torch
from enum import Enum
from typing import Tuple, Optional, Union, Dict, List
import scipy.ndimage as ndimage

class ResizeMethod(Enum):
    """Enumeration of different resize methods."""
    LETTERBOX = "letterbox"  # Preserve aspect ratio with padding
    CROP = "crop"            # Center crop to target size
    STRETCH = "stretch"      # Stretch to target size (distorts aspect ratio)
    PAD = "pad"              # Pad to target size without resizing

class InterpolationMethod(Enum):
    """Enumeration of different interpolation methods."""
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS = cv2.INTER_LANCZOS4

def letterbox_resize(image: np.ndarray, 
                     target_size: Tuple[int, int], 
                     interpolation: InterpolationMethod = InterpolationMethod.LANCZOS,
                     pad_value: Optional[float] = None) -> np.ndarray:
    """
    Resize an image to fit within target_size while preserving its aspect ratio.
    
    Args:
        image: Input image (2D numpy array)
        target_size: Target size as (width, height)
        interpolation: Interpolation method to use
        pad_value: Value to use for padding (if None, uses image mean)
        
    Returns:
        Resized and padded image
    """
    h, w = image.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation.value)
    
    # Compute padding value
    if pad_value is None:
        pad_value = image.mean()  # Removed int() casting to preserve float values for [0,1] images
    
    # Create canvas and place the resized image
    canvas = np.full((target_h, target_w), pad_value, dtype=image.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to the target size.
    
    Args:
        image: Input image (2D numpy array)
        target_size: Target size as (width, height)
        
    Returns:
        Cropped image
    """
    h, w = image.shape
    target_w, target_h = target_size
    
    # Calculate crop coordinates
    start_x = max(0, (w - target_w) // 2)
    start_y = max(0, (h - target_h) // 2)
    end_x = min(w, start_x + target_w)
    end_y = min(h, start_y + target_h)
    
    # Crop the image
    cropped = image[start_y:end_y, start_x:end_x]
    
    # If the image is smaller than the target size, pad it
    if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
        pad_value = image.mean()  # Use float mean value for [0,1] images
        result = np.full((target_h, target_w), pad_value, dtype=image.dtype)
        paste_y = (target_h - cropped.shape[0]) // 2
        paste_x = (target_w - cropped.shape[1]) // 2
        result[paste_y:paste_y+cropped.shape[0], paste_x:paste_x+cropped.shape[1]] = cropped
        return result
    
    return cropped

def pad_to_size(image: np.ndarray, 
                target_size: Tuple[int, int], 
                pad_value: Optional[float] = None) -> np.ndarray:
    """
    Pad an image to the target size without resizing.
    
    Args:
        image: Input image (2D numpy array)
        target_size: Target size as (width, height)
        pad_value: Value to use for padding (if None, uses image mean)
        
    Returns:
        Padded image
    """
    h, w = image.shape
    target_w, target_h = target_size
    
    # Compute padding value
    if pad_value is None:
        pad_value = image.mean()  # Removed int() casting for [0,1] images
    
    # Create canvas and place the image
    canvas = np.full((target_h, target_w), pad_value, dtype=image.dtype)
    paste_y = (target_h - h) // 2
    paste_x = (target_w - w) // 2
    
    # Ensure we don't go out of bounds
    paste_h = min(h, target_h)
    paste_w = min(w, target_w)
    
    canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = image[:paste_h, :paste_w]
    return canvas

def robust_normalize(slice_data: np.ndarray, 
                     lower_percentile: float = 1, 
                     upper_percentile: float = 99,
                     target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Apply robust normalization by clipping intensities at the given percentiles and
    scaling to the target range.
    
    Args:
        slice_data: Input image data
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        target_range: Target range for normalization
        
    Returns:
        Normalized image
    """
    # Handle empty or constant images
    if slice_data.size == 0 or np.all(slice_data == slice_data.flat[0]):
        return np.zeros_like(slice_data, dtype=np.float32)
    
    # Calculate percentiles
    lower = np.percentile(slice_data, lower_percentile)
    upper = np.percentile(slice_data, upper_percentile)
    
    # Avoid division by zero
    if upper == lower:
        return np.zeros_like(slice_data, dtype=np.float32)
    
    # Clip and normalize
    clipped = np.clip(slice_data, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    
    # Scale to target range
    min_val, max_val = target_range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized.astype(np.float32)

def histogram_equalization(image: np.ndarray, 
                           adaptive: bool = False, 
                           clip_limit: float = 2.0,
                           tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image: Input image (uint8)
        adaptive: Whether to use adaptive histogram equalization
        clip_limit: Clipping limit for adaptive equalization
        tile_grid_size: Tile grid size for adaptive equalization
        
    Returns:
        Equalized image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    if adaptive:
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    else:
        # Global histogram equalization
        return cv2.equalizeHist(image)

def apply_windowing(image: np.ndarray, 
                    window_center: float, 
                    window_width: float,
                    output_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Apply intensity windowing (common in medical imaging).
    
    Args:
        image: Input image
        window_center: Center of the window
        window_width: Width of the window
        output_range: Output intensity range
        
    Returns:
        Windowed image
    """
    min_val, max_val = output_range
    
    # Calculate window boundaries
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Apply windowing
    windowed = np.clip(image, window_min, window_max)
    
    # Normalize to output range
    if window_max > window_min:
        windowed = (windowed - window_min) / (window_max - window_min)
        windowed = windowed * (max_val - min_val) + min_val
    
    return windowed

def simulate_15T_data(data, noise_std=5, blur_sigma=0.5):
    """
    Simulate a 1.5T image from high-quality data by applying Gaussian blur and adding
    Rician-like noise.
    
    Args:
        data: Input image data in [0,1] range
        noise_std: Noise standard deviation (will be scaled by 1/255 for [0,1] range)
        blur_sigma: Sigma for Gaussian blur
    
    Returns:
        Simulated 1.5T image in [0,1] range
    """
    # Scale noise_std to be appropriate for [0,1] range data
    # (original values were for [0,255] range)
    scaled_noise_std = noise_std / 255.0
    
    # Apply Gaussian blur to mimic a smoother appearance
    blurred = ndimage.gaussian_filter(data, sigma=blur_sigma)
    
    # Create Rician-like noise by combining two independent Gaussian noise fields
    # with appropriate scale for [0,1] range
    noise1 = np.random.normal(0, scaled_noise_std, data.shape)
    noise2 = np.random.normal(0, scaled_noise_std, data.shape)
    
    # Apply Rician noise model
    simulated = np.sqrt((blurred + noise1)**2 + noise2**2)
    
    return simulated

def simulate_low_field_mri(data, kspace_crop_factor=0.5, noise_std=5):
    """
    Simulate a low-field MRI image using k-space manipulation and proper Rician noise.
    
    This function:
    1. Transforms the image to k-space using FFT
    2. Crops the k-space to simulate lower resolution
    3. Transforms back to image space using IFFT
    4. Adds Rician noise to simulate lower SNR
    
    Args:
        data: Input image data in [0,1] range
        kspace_crop_factor: Factor to determine how much of k-space to keep (0.5 = 50%)
        noise_std: Noise standard deviation (will be scaled by 1/255 for [0,1] range)
    
    Returns:
        Simulated low-field MRI image in [0,1] range
    """
    # Scale input to appropriate range for FFT
    # Preserve original min/max for rescaling back later
    orig_min, orig_max = data.min(), data.max()
    
    # Convert to k-space using FFT
    kspace = np.fft.fft2(data)
    kspace = np.fft.fftshift(kspace)  # Shift to center the low frequencies
    
    # Get dimensions
    rows, cols = kspace.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Calculate crop size
    crop_size_row = int(rows * kspace_crop_factor)
    crop_size_col = int(cols * kspace_crop_factor)
    
    # Create a mask (1 for kept regions, 0 for discarded regions)
    mask = np.zeros((rows, cols), dtype=np.complex128)
    row_start = center_row - crop_size_row // 2
    row_end = center_row + crop_size_row // 2
    col_start = center_col - crop_size_col // 2
    col_end = center_col + crop_size_col // 2
    
    # Apply mask to keep only the center of k-space
    mask[row_start:row_end, col_start:col_end] = 1
    low_res_kspace = kspace * mask
    
    # Transform back to image space
    low_res_image = np.fft.ifftshift(low_res_kspace)
    low_res_image = np.fft.ifft2(low_res_image)
    
    # Get real and imaginary components
    real_component = np.real(low_res_image)
    imag_component = np.imag(low_res_image)
    
    # Scale noise_std to be appropriate for [0,1] range data
    scaled_noise_std = noise_std / 255.0
    
    # Add Gaussian noise to both real and imaginary components
    real_noisy = real_component + np.random.normal(0, scaled_noise_std, real_component.shape)
    imag_noisy = imag_component + np.random.normal(0, scaled_noise_std, imag_component.shape)
    
    # Compute magnitude (this creates Rician noise distribution)
    magnitude = np.sqrt(real_noisy**2 + imag_noisy**2)
    
    # Normalize back to [0,1] range
    simulated = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    simulated = simulated * (orig_max - orig_min) + orig_min
    
    return simulated

def preprocess_slice(slice_data, target_size=None, interpolation=InterpolationMethod.CUBIC,
                     equalize=False, window_center=None, window_width=None, 
                     min_percentile=0.5, max_percentile=99.5, resize_method=ResizeMethod.LETTERBOX,
                     apply_simulation=False, noise_std=5, blur_sigma=0.5, pad_value=None,
                     kspace_crop_factor=0.5, use_kspace_simulation=True):
    """
    Process a single MRI slice with options for normalization, windowing, and resizing.
    Can optionally simulate low-resolution effects (blur and noise).
    
    Note: This function uses the specified interpolation method (default CUBIC) for all resize operations.
    For MRI super-resolution, it's recommended to use LANCZOS for HR images and CUBIC for LR images,
    which should be specified by the caller.
    
    Args:
        slice_data: 2D numpy array with the MRI slice data
        target_size: Target size as (width, height) or None to skip resizing
        interpolation: Interpolation method for resizing
        equalize: Whether to apply histogram equalization
        window_center: Center of intensity window (None for auto)
        window_width: Width of intensity window (None for auto)
        min_percentile: Minimum percentile for auto-windowing
        max_percentile: Maximum percentile for auto-windowing
        resize_method: Method for resizing (letterbox, crop, stretch, pad)
        apply_simulation: Whether to apply low-resolution simulation
        noise_std: Noise standard deviation for simulation (for 0-255 range, internally scaled)
        blur_sigma: Sigma for Gaussian blur in simulation
        pad_value: Value to use for padding (if None, uses image mean after normalization)
        kspace_crop_factor: Factor to determine how much of k-space to keep (0.5 = 50%)
        use_kspace_simulation: Whether to use k-space based simulation (True) or the old method (False)
        
    Returns:
        Processed slice as float32 array with values in [0, 1]
    """
    # Make a copy to avoid modifying the original
    processed = slice_data.astype(np.float32)
    
    # Apply windowing if requested (auto or manual)
    if window_center is not None and window_width is not None:
        center, width = window_center, window_width
        processed = apply_windowing(processed, center, width)
    else:
        # Auto-windowing using percentiles
        min_val = np.percentile(processed, min_percentile)
        max_val = np.percentile(processed, max_percentile)
        processed = np.clip(processed, min_val, max_val)

    # Normalize to [0, 1]
    min_val, max_val = processed.min(), processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    # Calculate padding value after normalization if None was provided
    # This ensures the padding value is in the same [0,1] range as the normalized image
    calculated_pad_value = pad_value
    if pad_value is None and target_size is not None and resize_method == ResizeMethod.LETTERBOX:
        calculated_pad_value = processed.mean()
    
    # Apply simulation if requested (after normalization but before resizing)
    if apply_simulation:
        if use_kspace_simulation:
            processed = simulate_low_field_mri(processed, kspace_crop_factor=kspace_crop_factor, noise_std=noise_std)
        else:
            processed = simulate_15T_data(processed, noise_std=noise_std, blur_sigma=blur_sigma)
        # Clip after simulation to ensure values stay in [0, 1] range
        processed = np.clip(processed, 0, 1)
    
    # Apply histogram equalization if requested
    if equalize:
        processed = histogram_equalization(processed, adaptive=True)
    
    # Resize if target size is provided
    if target_size:
        if resize_method == ResizeMethod.LETTERBOX:
            processed = letterbox_resize(processed, target_size, interpolation, calculated_pad_value)
        elif resize_method == ResizeMethod.CROP:
            processed = center_crop(processed, target_size)
        elif resize_method == ResizeMethod.PAD:
            processed = pad_to_size(processed, target_size, calculated_pad_value)
        elif resize_method == ResizeMethod.STRETCH:
            # Simple resize that may change aspect ratio - use the specified interpolation
            processed = cv2.resize(processed, target_size, interpolation=interpolation.value)
        else:
            # Use letterbox as default fallback
            max_dim = max(target_size)
            processed = letterbox_resize(processed, (max_dim, max_dim), interpolation, calculated_pad_value)
    
    return processed

def tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to a numpy array, handling device and shape."""
    # Move to CPU if on another device
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # Convert to numpy and handle channel dimension
    if tensor.ndim == 4:  # batch, channel, height, width
        img = tensor.squeeze(0).squeeze(0).numpy()
    elif tensor.ndim == 3:  # channel, height, width
        img = tensor.squeeze(0).numpy() 
    else:
        img = tensor.numpy()
        
    return img

def denormalize_from_range(tensor, low=-1.0, high=1.0):
    """Denormalize tensor from [low, high] to [0, 1]."""
    # Convert to float32 range [0, 1]
    return (tensor - low) / (high - low)

def numpy_to_tensor(array, device='cpu'):
    """Convert a numpy array to a PyTorch tensor with proper channel dimensions."""
    # Ensure array is float32
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    
    # Add channel dimension if needed
    if array.ndim == 2:
        array = array[np.newaxis, ...]
        
    # Convert to tensor and move to device
    tensor = torch.from_numpy(array).to(device)
    
    return tensor
