# File: utils/preprocessing.py
import cv2
import numpy as np

def letterbox_resize(image, target_size):
    """
    Resize an image to fit within target_size while preserving its aspect ratio.
    Pads with the image's mean intensity to reach the target dimensions.
    """
    h, w = image.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Compute the mean intensity of the image for padding.
    mean_val = int(image.mean())
    canvas = np.full((target_h, target_w), mean_val, dtype=image.dtype)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def robust_normalize(slice_data, lower_percentile=1, upper_percentile=99):
    """
    Apply robust normalization by clipping intensities at the given percentiles and
    scaling to [0, 1].
    """
    lower = np.percentile(slice_data, lower_percentile)
    upper = np.percentile(slice_data, upper_percentile)
    clipped = np.clip(slice_data, lower, upper)
    normalized = (clipped - lower) / (upper - lower) if upper != lower else clipped
    return normalized

def preprocess_slice(slice_data, target_size=None):
    """
    Normalize intensities robustly, convert to 0-255 uint8, and letterbox resize.
    If target_size is None, letterbox the image into a square with side equal to
    the maximum dimension of the original slice (preserving full resolution).
    """
    slice_norm = robust_normalize(slice_data)
    slice_uint8 = np.uint8(slice_norm * 255)
    if target_size is None:
        h, w = slice_uint8.shape
        target_size = (max(w, h), max(w, h))
    slice_letterboxed = letterbox_resize(slice_uint8, target_size)
    return slice_letterboxed

