import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(
        self,
        root_dir,
        df_paths,
        num_cls2name,
        augmentation=None, 
        preprocessing=None,
    ):
        self.root_dir = root_dir
        self.data = df_paths
        self.num_classes = len(num_cls2name)
        self.cls_encoder = num_cls2name
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path_to_image = os.join(self.root_dir, self.data['image'][idx])
        path_to_mask = os.join(self.root_dir, self.data['mask'][idx])
        
        image = cv2.imread(path_to_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(path_to_mask, 0)
        masks = [(mask == v) for v in num_cls2name]
        mask = np.stack(masks, axis=-1).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask
