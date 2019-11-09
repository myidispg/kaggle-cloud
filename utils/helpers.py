"""
This file contains the various helper functions used in
the project.
"""

import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

import albumentations

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

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albumentations.Lambda(image=preprocessing_fn),
    ]
    return albumentations.Compose(_transform)

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
        return self.bce_loss(y_pred.float(), y_true.float()) + self.dice_loss(y_pred, y_true)

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
    y_true = torch.from_numpy(np.random.rand(1, 3, 350, 525)).cuda()
    y_pred = torch.from_numpy(np.random.rand(1, 3, 350, 525)).cuda()

    # print(f'y_true: {y_true}\n\ny_pred: {y_pred}')

    dice_coefficient = DiceCoefficient()
    dice_loss = DiceLoss()
    bce_loss = BCEDiceLoss()

    print(f'dice coeeficient: {dice_coefficient(y_pred, y_true)}\n' \
        f'dice loss: {dice_loss(y_pred, y_true)}\n' \
        f'bce_loss: {bce_dice_loss(y_pred, y_true)}')

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss