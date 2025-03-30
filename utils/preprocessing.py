# File: utils/preprocessing.py
import cv2
import numpy as np
import torch
from enum import Enum
from typing import Tuple, Optional, Union, Dict, List
# import scipy.ndimage as ndimage # Removed unused import

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
                     pad_value: Optional[Union[int, float]] = None) -> np.ndarray:
    """
    Resize an image to fit within target_size while preserving its aspect ratio.

    Args:
        image: Input image (H, W) or (H, W, C) numpy array.
        target_size: Target size as (width, height).
        interpolation: Interpolation method to use.
        pad_value: Value to use for padding (if None, uses image mean).

    Returns:
        Resized and padded image.
    """
    original_shape = image.shape[:2]  # (h, w)
    target_w, target_h = target_size
    scale = min(target_w / original_shape[1], target_h / original_shape[0])
    new_w, new_h = int(original_shape[1] * scale), int(original_shape[0] * scale)

    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation.value)

    # Compute padding value
    if pad_value is None:
        pad_value = image.mean()
        if image.dtype == np.uint8:
            pad_value = int(round(pad_value))
        elif np.issubdtype(image.dtype, np.integer):
             pad_value = int(round(pad_value))
        # else keep float pad_value

    # Create canvas and place the resized image
    if image.ndim == 3:
        canvas_shape = (target_h, target_w, image.shape[2])
    else:
        canvas_shape = (target_h, target_w)

    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Ensure offsets are non-negative
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)

    # Ensure slicing does not go out of bounds for canvas
    end_y = min(target_h, y_offset + new_h)
    end_x = min(target_w, x_offset + new_w)

    # Ensure slicing does not go out of bounds for resized image
    slice_h = end_y - y_offset
    slice_w = end_x - x_offset

    if image.ndim == 3:
        canvas[y_offset:end_y, x_offset:end_x, :] = resized[:slice_h, :slice_w, :]
    else:
        canvas[y_offset:end_y, x_offset:end_x] = resized[:slice_h, :slice_w]

    return canvas

def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to the target size. If the image is smaller than
    the target size, it will be padded.

    Args:
        image: Input image (H, W) or (H, W, C) numpy array.
        target_size: Target size as (width, height).

    Returns:
        Cropped (and potentially padded) image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate crop coordinates
    start_x = max(0, (w - target_w) // 2)
    start_y = max(0, (h - target_h) // 2)
    end_x = min(w, start_x + target_w)
    end_y = min(h, start_y + target_h)

    # Crop the image
    if image.ndim == 3:
        cropped = image[start_y:end_y, start_x:end_x, :]
    else:
        cropped = image[start_y:end_y, start_x:end_x]

    # If the image was smaller than the target size in any dimension, pad it
    cropped_h, cropped_w = cropped.shape[:2]
    if cropped_h < target_h or cropped_w < target_w:
        pad_value = image.mean()
        if image.dtype == np.uint8:
            pad_value = int(round(pad_value))
        elif np.issubdtype(image.dtype, np.integer):
             pad_value = int(round(pad_value))

        if image.ndim == 3:
            canvas_shape = (target_h, target_w, image.shape[2])
        else:
            canvas_shape = (target_h, target_w)

        result = np.full(canvas_shape, pad_value, dtype=image.dtype)
        paste_y = (target_h - cropped_h) // 2
        paste_x = (target_w - cropped_w) // 2
        if image.ndim == 3:
            result[paste_y:paste_y+cropped_h, paste_x:paste_x+cropped_w, :] = cropped
        else:
            result[paste_y:paste_y+cropped_h, paste_x:paste_x+cropped_w] = cropped
        return result

    return cropped

def pad_to_size(image: np.ndarray,
                target_size: Tuple[int, int],
                pad_value: Optional[Union[int, float]] = None) -> np.ndarray:
    """
    Pad an image to the target size without resizing. If the image is larger,
    it will be cropped from the center.

    Args:
        image: Input image (H, W) or (H, W, C) numpy array.
        target_size: Target size as (width, height).
        pad_value: Value to use for padding (if None, uses image mean).

    Returns:
        Padded (or cropped) image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute padding value
    if pad_value is None:
        pad_value = image.mean()
        if image.dtype == np.uint8:
            pad_value = int(round(pad_value))
        elif np.issubdtype(image.dtype, np.integer):
             pad_value = int(round(pad_value))

    # Create canvas
    if image.ndim == 3:
        canvas_shape = (target_h, target_w, image.shape[2])
    else:
        canvas_shape = (target_h, target_w)
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)

    # Calculate paste coordinates (center alignment)
    paste_y = max(0, (target_h - h) // 2)
    paste_x = max(0, (target_w - w) // 2)

    # Calculate crop coordinates from source image (if image is larger than target)
    crop_y = max(0, (h - target_h) // 2)
    crop_x = max(0, (w - target_w) // 2)

    # Calculate the dimensions of the region to copy
    copy_h = min(h - crop_y, target_h - paste_y)
    copy_w = min(w - crop_x, target_w - paste_x)

    # Perform the copy
    if image.ndim == 3:
        canvas[paste_y : paste_y + copy_h, paste_x : paste_x + copy_w, :] = \
            image[crop_y : crop_y + copy_h, crop_x : crop_x + copy_w, :]
    else:
        canvas[paste_y : paste_y + copy_h, paste_x : paste_x + copy_w] = \
            image[crop_y : crop_y + copy_h, crop_x : crop_x + copy_w]

    return canvas


def robust_normalize(slice_data: np.ndarray,
                     lower_percentile: float = 1,
                     upper_percentile: float = 99,
                     target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Apply robust normalization by clipping intensities at the given percentiles and
    scaling to the target range.

    Args:
        slice_data: Input image data (numpy array).
        lower_percentile: Lower percentile for clipping.
        upper_percentile: Upper percentile for clipping.
        target_range: Target range for normalization (min_val, max_val).

    Returns:
        Normalized image as float32.
    """
    # Handle empty images
    if slice_data.size == 0:
        return np.zeros_like(slice_data, dtype=np.float32)

    # Calculate percentiles on non-zero values if appropriate, or all values
    # Consider if ignoring zeros is desired for medical images (e.g., background)
    # For general purpose, use all values:
    pixels = slice_data.flatten()

    # Handle constant images
    if np.all(pixels == pixels[0]):
        # If constant, map to the middle of the target range or min_val?
        # Let's map to min_val for consistency with division by zero case below.
        min_target, max_target = target_range
        return np.full_like(slice_data, min_target, dtype=np.float32)

    lower = np.percentile(pixels, lower_percentile)
    upper = np.percentile(pixels, upper_percentile)

    # Avoid division by zero if percentiles are the same
    if upper <= lower:
        # Normalize everything to the minimum of the target range
        min_target, _ = target_range
        return np.full_like(slice_data, min_target, dtype=np.float32)

    # Clip and normalize to [0, 1]
    clipped = np.clip(slice_data, lower, upper)
    normalized = (clipped - lower) / (upper - lower)

    # Scale to target range
    min_target, max_target = target_range
    normalized = normalized * (max_target - min_target) + min_target

    return normalized.astype(np.float32)

def histogram_equalization(image: np.ndarray,
                           adaptive: bool = False,
                           clip_limit: float = 2.0,
                           tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply histogram equalization to enhance image contrast. Input image should
    typically be in the range [0, 255] for standard methods.

    Args:
        image: Input image (numpy array, ideally uint8).
        adaptive: Whether to use adaptive histogram equalization (CLAHE).
        clip_limit: Clipping limit for CLAHE.
        tile_grid_size: Tile grid size for CLAHE.

    Returns:
        Equalized image (same dtype as input if uint8, otherwise uint8).
    """
    # Ensure image is uint8, scaling if necessary
    if image.dtype != np.uint8:
        if image.min() >= 0 and image.max() <= 1:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            # Attempt robust scaling to [0, 255]
            img_norm = robust_normalize(image, target_range=(0, 255))
            img_uint8 = img_norm.astype(np.uint8)
    else:
        img_uint8 = image

    if adaptive:
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized = clahe.apply(img_uint8)
    else:
        # Global histogram equalization
        equalized = cv2.equalizeHist(img_uint8)

    # Return in the same format if original was uint8, otherwise maybe float?
    # For now, returning uint8 as that's the direct output of cv2 functions.
    # If float output is desired, scale back to [0, 1]: return equalized.astype(np.float32) / 255.0
    return equalized


def apply_windowing(image: np.ndarray,
                    window_center: float,
                    window_width: float,
                    output_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Apply intensity windowing (common in medical imaging).

    Args:
        image: Input image (numpy array).
        window_center: Center of the intensity window.
        window_width: Width of the intensity window.
        output_range: Output intensity range (min_val, max_val).

    Returns:
        Windowed image (float32).
    """
    min_target, max_target = output_range

    # Calculate window boundaries
    window_min = window_center - window_width / 2.0
    window_max = window_center + window_width / 2.0

    # Clip image intensities to the window
    windowed = np.clip(image, window_min, window_max)

    # Normalize to output range
    if window_max > window_min:
        # Normalize to [0, 1] first
        windowed = (windowed - window_min) / (window_max - window_min)
        # Scale to target range
        windowed = windowed * (max_target - min_target) + min_target
    else:
        # Handle zero width window: map everything to the min target value
        windowed = np.full_like(image, min_target, dtype=np.float32)


    return windowed.astype(np.float32)


def preprocess_slice(slice_data: np.ndarray,
                     target_size: Optional[Tuple[int, int]] = None,
                     interpolation: InterpolationMethod = InterpolationMethod.CUBIC,
                     equalize: bool = False,
                     window_center: Optional[float] = None,
                     window_width: Optional[float] = None,
                     min_percentile: float = 0.5,
                     max_percentile: float = 99.5,
                     resize_method: ResizeMethod = ResizeMethod.LETTERBOX,
                     normalize: bool = True) -> np.ndarray:
    """
    Process a single 2D image slice with options for windowing, normalization,
    equalization, and resizing.

    Args:
        slice_data: 2D numpy array with the image data.
        target_size: Target size as (width, height) or None to skip resizing.
        interpolation: Interpolation method for resizing.
        equalize: Whether to apply adaptive histogram equalization (CLAHE).
        window_center: Center of intensity window. If None with window_width,
                       uses robust normalization based on percentiles instead.
        window_width: Width of intensity window. If None with window_center,
                      uses robust normalization based on percentiles instead.
        min_percentile: Min percentile for robust normalization (if windowing not used).
        max_percentile: Max percentile for robust normalization (if windowing not used).
        resize_method: Method for resizing (letterbox, crop, stretch, pad).
        normalize: Whether to normalize the output to [0, 1]. If False, the range
                   depends on previous steps (e.g., windowing output range or
                   equalization output [0, 255]).

    Returns:
        Processed slice as float32 array. Output range is [0, 1] if normalize=True.
    """
    # Make a copy to avoid modifying the original
    processed = slice_data.astype(np.float32)

    # 1. Apply windowing or robust normalization (mutually exclusive logic here)
    if window_center is not None and window_width is not None:
        # Apply manual windowing, output range is [0, 1] by default
        processed = apply_windowing(processed, window_center, window_width, output_range=(0, 1))
        # Set normalize flag to False as windowing already handled scaling
        normalize = False
    elif normalize:
        # Apply robust normalization to [0, 1] using percentiles
        processed = robust_normalize(processed, min_percentile, max_percentile, target_range=(0, 1))
        # Set normalize flag to False as robust_normalize handled it
        normalize = False

    # 2. Normalize to [0, 1] if not already done by windowing/robust_normalize
    if normalize:
        min_val, max_val = processed.min(), processed.max()
        if max_val > min_val:
            processed = (processed - min_val) / (max_val - min_val)
        else:
            processed = np.zeros_like(processed, dtype=np.float32) # Handle constant image

    # 3. Apply histogram equalization if requested (operates best on [0, 1] or [0, 255])
    if equalize:
        # CLAHE works on uint8, so scale to [0, 255], apply, then scale back to [0, 1]
        processed_uint8 = (np.clip(processed, 0, 1) * 255).astype(np.uint8)
        equalized_uint8 = histogram_equalization(processed_uint8, adaptive=True)
        processed = equalized_uint8.astype(np.float32) / 255.0

    # 4. Resize if target size is provided
    if target_size:
        if resize_method == ResizeMethod.LETTERBOX:
            processed = letterbox_resize(processed, target_size, interpolation)
        elif resize_method == ResizeMethod.CROP:
            processed = center_crop(processed, target_size)
        elif resize_method == ResizeMethod.PAD:
            processed = pad_to_size(processed, target_size)
        elif resize_method == ResizeMethod.STRETCH:
            # Simple resize that may change aspect ratio
            processed = cv2.resize(processed, target_size,
                                 interpolation=interpolation.value)
        else:
            # Default fallback: Letterbox
            print(f"Warning: Unknown resize method '{resize_method}'. Defaulting to LETTERBOX.")
            processed = letterbox_resize(processed, target_size, interpolation)

    # Ensure final output is float32
    return processed.astype(np.float32)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array, handling device and shape.
       Assumes tensor is image-like (e.g., CHW, BCHW, HW).
    """
    # Move to CPU if on another device
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    # Detach from computation graph
    tensor = tensor.detach()

    # Convert to numpy
    img_np = tensor.numpy()

    # Handle channel dimension if present (move channel to last dim for typical numpy usage)
    if img_np.ndim == 3:  # CHW -> HWC
        img_np = np.transpose(img_np, (1, 2, 0))
    elif img_np.ndim == 4: # BCHW -> BHWC
        img_np = np.transpose(img_np, (0, 2, 3, 1))
        if img_np.shape[0] == 1: # Remove batch dim if size 1
            img_np = img_np.squeeze(0)

    # Squeeze potential singleton dimensions (e.g., if channel was 1)
    img_np = img_np.squeeze()

    return img_np

def denormalize_from_range(tensor: Union[torch.Tensor, np.ndarray],
                           low: float = -1.0,
                           high: float = 1.0) -> Union[torch.Tensor, np.ndarray]:
    """Denormalize tensor/array from [low, high] to [0, 1]."""
    # Convert to float range [0, 1]
    return (tensor - low) / (high - low)

def numpy_to_tensor(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert a numpy array to a PyTorch tensor with proper channel dimensions (CHW).
       Handles HW, HWC inputs.
    """
    # Ensure array is float32
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    # Add channel dimension if needed (HW -> CHW)
    if array.ndim == 2:
        array = array[np.newaxis, ...] # HW -> 1HW
    # Transpose if HWC (HWC -> CHW)
    elif array.ndim == 3 and array.shape[-1] in [1, 3, 4]: # Basic check for channels last
         array = np.transpose(array, (2, 0, 1)) # HWC -> CHW

    # Convert to tensor and move to device
    tensor = torch.from_numpy(array).to(device)

    return tensor