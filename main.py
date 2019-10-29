import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2

import torch

import utils

# First read the data into a pandas dataframe
df_train = pd.read_csv(os.path.join(utils.DATA_DIR, 'train.csv'))
# print(df_train.head())

# Start training the model
model = utils.fcdensenet_tiny(n_classes=1, drop_rate=0.2)
# model = utils.UNet(n_channels=3, n_classes=1, bilinear=False)
# print(model)

train = utils.Train(model, resume=False, log_every=1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
train.train(n_epochs=2)