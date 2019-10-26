"""
This file contains the various helper functions used in
the project.
"""

import numpy as np
import albumentations

def get_segmentation_mask(mask_rle: str = '', img_shape: tuple = (1400, 2100)):
    """
    Takes the start value and the following pixel count, returns a mask of the same size as image. 
    Input:
        mask_rle: The string of encoded pixels
        img_shape: The image dimensions. Tuple (HxW)
    Returns:
        mask: Numpy array: 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    # First, work on a single dimensional array, reshape later
    mask = np.zeros(img_shape[0] * img_shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start: start + length] = 1
    return mask.reshape(img_shape, order='F')

# The following are the functions for Segmentation Models
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albumentations.Lambda(image=preprocessing_fn),
        # albumentations.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albumentations.Compose(_transform)