# LICENSE-DATASET  candle  capsules  cashew  chewinggum  fryum  macaroni1  macaroni2  pcb1  pcb2  pcb3  pcb4  pipe_fryum  split_csv

import os
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

class VisA(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        transform: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False, 
        **kwargs
    ):
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.custom_transforms = transform
        self.is_mask = is_mask
        self.cls_label = cls_label

        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'test'
        
        # Load csv meta file
        csv_path = os.path.join(self.data_root, "split_csv", "1cls.csv")
        self.meta = pd.read_csv(csv_path)
        
        # Get files
        self.img_files = self.get_files()
        self.labels = [0] * len(self.img_files)
        if self.split == 'test':
            def mask_to_tensor(img):
                return torch.from_numpy(np.array(img, dtype=np.uint8)).long()
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize((input_res, input_res), interpolation=InterpolationMode.NEAREST),
                    transforms.Lambda(mask_to_tensor)
                ]
            )
            
            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'Normal':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
            
        self.num_classes = len(VISA_CLASSES)
        self.labels_to_names = {i: name for i, name in enumerate(VISA_CLASSES)}
        
    def __len__(self):
        return len(self.img_files)

    def get_files(self):
        if self.split == 'train':
            files = self.meta[(self.meta['object'] == self.category) & (self.meta['split'] == 'train')]
        else:
            normal_img_files = self.meta[(self.meta['object'] == self.category) & (self.meta['split'] == 'test') & (self.meta['label'] == 'normal')]
            anom_img_files = self.meta[(self.meta['object'] == self.category) & (self.meta['split'] == 'test') & (self.meta['label'] == 'anomaly')]
            files = pd.concat([normal_img_files, anom_img_files])
        files = sorted([os.path.join(self.data_root, f) for f in files['image']])
        return files

    def __getitem__(self, index):
        inputs = {}
        
        img_file = self.img_files[index]
        label = self.labels[index]
        
        cls_name = img_file.split(os.path.sep)[-5]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsname"] = cls_name
        inputs["clslabel"] = VISA_CLASSES.index(cls_name)
        inputs["path"] = img_file

        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["img"] = sample
            return inputs
        else:
            inputs["img"] = sample
            inputs["label"] = label
            if "Normal" in str(img_file):
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = "anomaly"
            if self.is_mask:
                mask_dir =  Path(img_file).parent.parent.parent / 'Masks' / 'Anomaly'
                mask_file = mask_dir / (Path(img_file).stem + '.png')
                if inputs["anom_type"] == "good":
                    mask = Image.new('L', (self.input_res, self.input_res), 0)
                else:
                    with open(mask_file, 'rb') as f:
                        mask = Image.open(f)
                        mask = mask.convert('L')
                mask = self.mask_transform(mask).bool()
                inputs["mask"] = mask
            return inputs
