import numpy as np
import cv2
import os

from torch.utils.data import Dataset

from constants import DATA_DIR, LABEL2IDX, IDX2LABELS, VALIDATION_SPLIT, IMAGE_SIZE_SMALL, IMAGE_SIZE_ORIG
from helpers import get_segmentation_mask, dice_coefficient, dice_loss, bce_dice_loss

from cloud_dataset import CloudDataset