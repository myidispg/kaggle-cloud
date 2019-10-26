import numpy as np
import cv2
import os

from torch.utils.data import Dataset

from constants import DATA_DIR, LABEL2IDX
from helpers import get_segmentation_mask

class CloudDataset(Dataset):
    """Cloud Dataset for Segmentation"""
    def __init__(self, df, transforms, preprocessing = None, train: bool=True):
        """
        Custom PyTorch Cloud Dataset.
        Inputs: 
            df: The dataframe with labels and rle
            transforms: The albumentations transforms pipeline
            train: A boolean value to denote whether train set or validations set
        """
        self.df = df
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_label = row[0]
        rle_string = row[1] if type(row[1]) == str else ""

        image_name = image_label.split('_')[0]
        label = image_label.split('_')[1]

        # 0 for grayscale mode to be threshlolded later.
        if self.train:
            img = cv2.imread(os.path.join(DATA_DIR, f'train/{image_name}'))
        else:
            img = cv2.imread(os.path.join(DATA_DIR, f'test/{image_name}'))

        mask = get_segmentation_mask(rle_string, 'None' if len(rle_string) == 0 else label)
        
        # Apply the transforms
        if self.transforms:
          augmented = self.transforms(image=img, mask=mask)
          img = augmented['image']
          mask = augmented['mask']

        img = img.transpose(2, 0, 1).astype('float32')
        
        # Return the image, mask and the label for determing class
        return img, mask, LABEL2IDX[label]
