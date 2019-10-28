"""
This file contains the various helper functions used in
the project.
"""

import numpy as np
import torch

def get_segmentation_mask(mask_rle: str = '', label: str = 'None'):
    """
    Takes the start value and the following pixel count, returns a mask of the same size as image. 
    Input:
        mask_rle: The string of encoded pixels
        img_shape: The image dimensions. Tuple (HxW)
    Returns:
        mask: Numpy array: 1 - mask, 0 - background
    """
    img_shape = (1400, 2100) # The RLE's are according to 1400x2100.

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    # First, work on a single dimensional array, reshape later
    mask = np.zeros(img_shape[0] * img_shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start: start + length] = 1
    return mask.reshape(img_shape, order='F')

class DiceCoefficient:

    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        """
        Calculate the dice coefficeint of the given PyTorch Tensors.
        """
        assert(type(y_pred) == type(y_true) == torch.Tensor), f"Both the arrays must be PyTorch tensors"
        assert(y_pred.shape == y_true.shape), f"Both the tensors must have equal shape. y_pred has shape: {y_pred.shape} and y_true has shape: {y_true.shape}"
        y_true_flat = y_true.view(y_true.shape[0], -1)
        y_pred_flat = y_pred.view(y_pred.shape[0], -1)
        intersection = torch.sum(y_true_flat * y_pred_flat)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + self.smooth)

class DiceLoss:

    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        assert(type(y_pred) == type(y_true) == torch.Tensor), f"Both the arrays must be PyTorch tensors"
        assert(y_pred.shape == y_true.shape), f"Both the tensors must have equal shape. y_pred has shape: {y_pred.shape} and y_true has shape: {y_true.shape}"
        y_true_flat = y_true.view(y_true.shape[0], -1)
        y_pred_flat = y_pred.view(y_pred.shape[0], -1)
        intersection = y_true_flat * y_pred_flat
        score = (2. * torch.sum(intersection) + self.smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + self.smooth)
        return 1. - score

class BCEDiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth

        self.dice_loss = DiceLoss(self.smooth)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, y_pred, y_true):
        return self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)

def dice_coefficient(y_pred, y_true, smooth=1):
    """
    Calculate the dice coefficeint of the given numpy arrays.
    """
    assert(type(y_pred) == type(y_true) == torch.Tensor), f"Both the arrays must be PyTorch tensors"
    assert(y_pred.shape == y_true.shape), f"Both the tensors must have equal shape. y_pred has shape: {y_pred.shape} and y_true has shape: {y_true.shape}"
    y_true_flat = y_true.view(y_true.shape[0], -1)
    y_pred_flat = y_pred.view(y_pred.shape[0], -1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth)

def dice_loss(y_pred, y_true, smooth=1):
    assert(type(y_pred) == type(y_true) == torch.Tensor), f"Both the arrays must be PyTorch tensors"
    assert(y_pred.shape == y_true.shape), f"Both the tensors must have equal shape. y_pred has shape: {y_pred.shape} and y_true has shape: {y_true.shape}"
    y_true_flat = y_true.view(y_true.shape[0], -1)
    y_pred_flat = y_pred.view(y_pred.shape[0], -1)
    intersection = y_true_flat * y_pred_flat
    score = (2. * torch.sum(intersection) + smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth)
    return 1. - score

def bce_dice_loss(y_pred, y_true, smooth=1):
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true) + dice_loss(y_pred, y_true)

if __name__ == '__main__':
    y_true = torch.from_numpy(np.random.rand(1, 3, 350, 525))
    y_pred = torch.from_numpy(np.random.rand(1, 3, 350, 525))

    # print(f'y_true: {y_true}\n\ny_pred: {y_pred}')

    dice_coefficient = DiceCoefficient()
    dice_loss = DiceLoss()
    bce_loss = BCEDiceLoss()

    print(f'dice coeeficient: {dice_coefficient(y_pred, y_true)}\n' \
        f'dice loss: {dice_loss(y_pred, y_true)}\n' \
        f'bce_loss: {bce_dice_loss(y_pred, y_true)}')