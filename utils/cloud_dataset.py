#%%
from torch.utils.data import Dataset
import cv2
import os

from .constants import DATA_DIR, LABEL2IDX, IDX2LABELS
from .helpers import get_segmentation_mask

class CloudDataset(Dataset):
    """Cloud Dataset for Segmentation"""
    def __init__(self, df, transforms, train: bool=True, output_img_shape: tuple=True):
        """
        Custom PyTorch Cloud Dataset.
        Inputs: 
            df: The dataframe with labels and rle
            transforms: The albumentations transforms pipeline
            train: A boolean value to denote whether train set or validations set
            output_img_shape: The output shape of the image and the mask
        """
        # assert type(output_img_shape) == tuple, "The output_img_shape must be a tuple"
        assert len(output_img_shape) == 2, "The output_img_shape must be of 2 dimensions"

        self.df = df
        self.transforms = transforms
        self.train = train

        self.output_img_shape = output_img_shape
        
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
            img = cv2.imread(os.path.join(os.getcwd(), DATA_DIR, f'train/{image_name}'))
        else:
            img = cv2.imread(os.path.join(DATA_DIR, f'test/{image_name}'))

        img = cv2.resize(img, self.output_img_shape[::-1])
        mask = cv2.resize(get_segmentation_mask(mask_rle=rle_string,
                label='None' if len(rle_string) == 0 else label),
                self.output_img_shape[::-1])
        
        # Apply the transforms
        if self.transforms:
          augmented = self.transforms(image=img, mask=mask)
          img = augmented['image']
          mask = augmented['mask']

        img = img.transpose(2, 0, 1).astype('float32')
        
        # Return the image, mask and the label for determing class
        return img, mask, LABEL2IDX[label]
#%%
if __name__ == '__main__':
    import pandas as pd
    import os
    
    import torch
    from torch.utils.data import DataLoader

    from constants import SHUFFLE_DATASET, VALIDATION_SPLIT, IDX2LABELS, IMAGE_SIZE_SMALL

    import albumentations

    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    print(df_train.head())

    transforms = albumentations.Compose([
        albumentations.Normalize(mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)),
        albumentations.GridDistortion(p=0.2),
        albumentations.HorizontalFlip(p=0.2),
        albumentations.ShiftScaleRotate(p=0.2),
        ])
    cloud_dataset = CloudDataset(df_train, transforms, output_img_shape=IMAGE_SIZE_SMALL)
    # Creating indices for train and test set
    dataset_size = len(cloud_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    if SHUFFLE_DATASET:
        # np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(cloud_dataset, batch_size=4, sampler=train_sampler)
    val_loader = DataLoader(cloud_dataset, batch_size=4, sampler=val_sampler)

    myiter = iter(train_loader)
    image, mask, label = next(myiter)
    print(f'Image: {image.shape}, Mask: {mask.shape}, Label: {label.shape}')

    import matplotlib.pyplot as plt
    
    f, axarr = plt.subplots(4, 2, figsize=(10, 20))
    for i in range(4):
        axarr[i, 0].imshow(image[i].permute(1, 2, 0).numpy())
        axarr[i, 0].set_title(f'Generated label: {IDX2LABELS[label[i].item()]}')
        axarr[i, 1].imshow(mask[i].numpy())
        axarr[i, 1].set_title(f'Generated label: {IDX2LABELS[label[i].item()]}')

# %%
