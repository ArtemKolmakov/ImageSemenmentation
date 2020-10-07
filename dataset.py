import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from dataset_utils import get_training_augmentation, get_validation_augmentation, get_preprocessing

class SegDataset(Dataset):
    def __init__(
        self,
        df_paths,
        num_cls,
        augmentation=None, 
        preprocessing=None,
    ):
        self.data = df_paths
        self.num_classes = num_cls
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        image = cv2.imread(self.data.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.data.masks[idx], 0)
        masks = [(mask == v) for v in range(self.num_classes)]
        mask = np.stack(masks, axis=-1).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask
  

 
def get_dataloader(
    path2csv_dataset,
    encoder,
    encoder_weights,
    num_cls=12,
    is_train=True,
    batch_size=8,
    shuffle=True,
    num_workers=12
):
    df = pd.read_csv(path2csv_dataset, sep=' ', names=['images', 'masks'])
    df.images = df.images.map(lambda x: x.replace('/SegNet', 'data'))
    df.masks = df.masks.map(lambda x: x.replace('/SegNet', 'data'))
    preprocessing_func = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    augmentation = get_training_augmentation if is_train else get_validation_augmentation
    dataset = SegDataset(
        df_paths=df,
        num_cls=num_cls,
        augmentation=augmentation(), 
        preprocessing=get_preprocessing(preprocessing_func),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
