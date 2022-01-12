"""PACKAGES"""

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import cv2
from PIL import Image
import albumentations as A

"""CONFIG"""

IMAGE_PATH = 'Insert path here'
MASK_PATH = 'Insert path here'
BATCH_SIZE = 10

"""DATA"""

def create_df(path):
    name = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


df_train = create_df(IMAGE_PATH)
print('Total Training Images: ', len(df_train))

X_train = df_train['id'].values

img = Image.open(IMAGE_PATH + df_train['id'][len(X_train) - 1] + '.tif')
mask = Image.open(MASK_PATH + df_train['id'][len(X_train) - 1] + '.tif')

"""PYTORCH DATALOADER"""

class SegmentationDataset(Dataset):

    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        return img, mask


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

t_train = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST)])
train_set = SegmentationDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, transform=t_train)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)