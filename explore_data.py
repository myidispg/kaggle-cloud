#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2
import albumentations

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import DATA_DIR
from utils.helpers import get_segmentation_mask

#%%
df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_train.head()

#%%
# Count the number of labelled and unlabelled datapoints.

print(f'Counting the number of labelled and unlabelled datapoints.')
count_labelled = 0
count_unlabelled = 0
for index, value in df_train.iterrows():
    # If labelled, we get a string
    if type(value['EncodedPixels']) == str:
        count_labelled += 1
    else:
        count_unlabelled += 1

print(f'The number of unlabelled samples is: {count_unlabelled}')
print(f'The number of labelled samples is: {count_labelled}')
counts = [count_labelled, count_unlabelled]
labels = ['Labelled', 'Unlabelled']

# Draw the pie chart
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=labels, explode=(0.05, 0.05), colors=('b', 'r'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis=('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

del fig1, ax1

"""As is evident by the chart, the classes seem to be balanced."""

#%%

print('Counting the number of instances of each class.')
label_count_dict = dict()

# Count the number of datapoints for each label.
for index, value in df_train.iterrows():
    # print(value['Image_Label'].split('_')[1])
    try:
        label_count_dict[value['Image_Label'].split('_')[1]] += 1
    except KeyError:
        label_count_dict[value['Image_Label'].split('_')[1]] = 1
print(f'The classes are: {list(label_count_dict.keys())}')

fig1, ax1 = plt.subplots()
ax1.pie(list(label_count_dict.values()),
 labels=list(label_count_dict.keys()),
 explode=[0.05 for key in label_count_dict.keys()],
#  colors=('b', 'r'),
 autopct='%1.1f%%',
 shadow=True,
 startangle=90)
ax1.axis=('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

"""We can see that the labels are perfectly balanced too."""
# %%
print('Now, to see some images and their mask.')
print('Plotting 20 images and their masks.')
fig = plt.figure(figsize= (20, 20))
rows = 5
cols = 4
for i in range(1, rows*cols + 1):
    ax = fig.add_subplot(rows, cols, i)
    data = df_train.sample() # Get a single row as a dataframe
    label = data.iloc[0, 0]
    rle_string = data.iloc[0, 1] if type(data.iloc[0, 1]) == str else ""
    image_name = str(label.split("_")[0])
    img = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR,f'train/{image_name}')), cv2.COLOR_BGR2RGB)
    mask = get_segmentation_mask(rle_string)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5, cmap='gray')
    ax.set_title(f"label: {label}")

