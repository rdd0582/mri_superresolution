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
                     interpolation: InterpolationMethod = InterpolationMethod.AREA,
                     pad_value: Optional[int] = None) -> np.ndarray:
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
        pad_value = int(image.mean())
    
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
        pad_value = int(image.mean())
        result = np.full((target_h, target_w), pad_value, dtype=image.dtype)
        paste_y = (target_h - cropped.shape[0]) // 2
        paste_x = (target_w - cropped.shape[1]) // 2
        result[paste_y:paste_y+cropped.shape[0], paste_x:paste_x+cropped.shape[1]] = cropped
        return result
    
    return cropped

def pad_to_size(image: np.ndarray, 
                target_size: Tuple[int, int], 
                pad_value: Optional[int] = None) -> np.ndarray:
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
        pad_value = int(image.mean())
    
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

def preprocess_slice(slice_data: np.ndarray,
                     target_size: Optional[Tuple[int, int]] = None,
                     resize_method: ResizeMethod = ResizeMethod.LETTERBOX,
                     normalize: bool = True,
                     equalize: bool = False,
                     apply_window: bool = False,
                     window_params: Optional[Dict] = None,
                     to_uint8: bool = True,
                     interpolation: InterpolationMethod = InterpolationMethod.AREA) -> np.ndarray:
    """
    Comprehensive preprocessing pipeline for MRI slices.
    
    Args:
        slice_data: Input slice data
        target_size: Target size as (width, height)
        resize_method: Method to use for resizing
        normalize: Whether to apply robust normalization
        equalize: Whether to apply histogram equalization
        apply_window: Whether to apply intensity windowing
        window_params: Parameters for windowing (center, width)
        to_uint8: Whether to convert to uint8
        interpolation: Interpolation method for resizing
        
    Returns:
        Preprocessed slice
    """
    # Make a copy to avoid modifying the original
    processed = slice_data.copy()
    
    # Apply normalization if requested
    if normalize:
        processed = robust_normalize(processed)
    
    # Apply windowing if requested
    if apply_window and window_params is not None:
        center = window_params.get('center', 0.5)
        width = window_params.get('width', 1.0)
        processed = apply_windowing(processed, center, width)
    
    # Convert to uint8 if requested
    if to_uint8:
        processed = np.clip(processed * 255, 0, 255).astype(np.uint8)
    
    # Apply histogram equalization if requested
    if equalize:
        if processed.dtype != np.uint8:
            processed = np.clip(processed * 255, 0, 255).astype(np.uint8)
        processed = histogram_equalization(processed, adaptive=True)
    
    # Handle resizing
    if target_size is not None:
        if resize_method == ResizeMethod.LETTERBOX:
            processed = letterbox_resize(processed, target_size, interpolation)
        elif resize_method == ResizeMethod.CROP:
            processed = center_crop(processed, target_size)
        elif resize_method == ResizeMethod.STRETCH:
            processed = cv2.resize(processed, target_size, interpolation=interpolation.value)
        elif resize_method == ResizeMethod.PAD:
            processed = pad_to_size(processed, target_size)
    elif processed.shape[0] != processed.shape[1]:
        # If no target size provided but image is not square, make it square
        max_dim = max(processed.shape)
        if resize_method == ResizeMethod.LETTERBOX:
            processed = letterbox_resize(processed, (max_dim, max_dim), interpolation)
        elif resize_method == ResizeMethod.PAD:
            processed = pad_to_size(processed, (max_dim, max_dim))
    
    return processed

def simulate_low_resolution(image: np.ndarray, 
                            scale_factor: float = 0.5,
                            noise_level: float = 0.01,
                            blur_sigma: float = 0.5,
                            add_artifacts: bool = False) -> np.ndarray:
    """
    Simulate a low-resolution image from a high-resolution one.
    
    Args:
        image: High-resolution input image
        scale_factor: Scale factor for downsampling
        noise_level: Standard deviation of noise to add
        blur_sigma: Sigma for Gaussian blur
        add_artifacts: Whether to add simulated artifacts
        
    Returns:
        Simulated low-resolution image
    """
    # Apply Gaussian blur
    blurred = ndimage.gaussian_filter(image.astype(np.float32), sigma=blur_sigma)
    
    # Downsample
    h, w = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    downsampled = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Add Rician noise (common in MRI)
    if noise_level > 0:
        noise1 = np.random.normal(0, noise_level, downsampled.shape)
        noise2 = np.random.normal(0, noise_level, downsampled.shape)
        downsampled = np.sqrt((downsampled + noise1)**2 + noise2**2)
    
    # Add simulated artifacts if requested
    if add_artifacts:
        # Simulate motion artifacts (horizontal lines)
        if np.random.random() < 0.3:
            num_lines = np.random.randint(1, 5)
            for _ in range(num_lines):
                line_pos = np.random.randint(0, new_h)
                line_width = np.random.randint(1, 3)
                line_intensity = np.random.uniform(0.8, 1.2)
                downsampled[line_pos:line_pos+line_width, :] *= line_intensity
        
        # Simulate intensity non-uniformity
        if np.random.random() < 0.3:
            x, y = np.meshgrid(np.linspace(-1, 1, new_w), np.linspace(-1, 1, new_h))
            bias_field = 1 + 0.1 * np.random.random() * (x**2 + y**2)
            downsampled *= bias_field
    
    # Upsample back to original size
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Ensure values are in valid range
    upsampled = np.clip(upsampled, 0, 1) if image.dtype == np.float32 else np.clip(upsampled, 0, 255)
    
    return upsampled.astype(image.dtype)

def batch_preprocess(images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """
    Apply preprocessing to a batch of images with the same parameters.
    
    Args:
        images: List of input images
        **kwargs: Preprocessing parameters passed to preprocess_slice
        
    Returns:
        List of preprocessed images
    """
    return [preprocess_slice(img, **kwargs) for img in images]

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy array.
    
    Args:
        tensor: Input tensor (C,H,W)
        
    Returns:
        Numpy array (H,W,C)
    """
    if tensor.ndim == 3:  # C,H,W
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    elif tensor.ndim == 4:  # B,C,H,W
        return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:
        return tensor.detach().cpu().numpy()

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a PyTorch tensor.
    
    Args:
        array: Input array (H,W,C) or (H,W)
        
    Returns:
        PyTorch tensor (C,H,W)
    """
    if array.ndim == 3:  # H,W,C
        return torch.from_numpy(array.transpose(2, 0, 1))
    elif array.ndim == 4:  # B,H,W,C
        return torch.from_numpy(array.transpose(0, 3, 1, 2))
    else:
        return torch.from_numpy(array)
