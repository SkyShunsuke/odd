
import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset

from .mvtec_ad import MVTecAD, AD_CLASSES
from .visa import VisA, VISA_CLASSES
from .mpdd import MPDD, MPDD_CLASSES

import logging
logger = logging.getLogger(__name__)

def build_transforms(img_size, transform_type):
    # standarization
    default_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    if transform_type == 'default':
        return default_transform
    elif transform_type == 'imagenet':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif transform_type == 'crop':
        return transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif transform_type == 'rotate':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        raise ValueError(f"Invalid transform: {transform_type}")
    
class GlobalIndexConcatDataset(torch.utils.data.ConcatDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["index"] = idx  
        return item

_DATASETS = {
    'mvtec_ad': (MVTecAD, AD_CLASSES),
    'visa': (VisA, VISA_CLASSES),
    'mpdd': (MPDD, MPDD_CLASSES),
}

def build_dataset(*, dataset_name: str, data_root: str, train: bool, img_size: int, transform_type: str, **kwargs):
    logger.info(f"Building dataset: {dataset_name}, train: {train}, img_size: {img_size}, transform: {transform_type}")
    base_name = dataset_name[:-4] if dataset_name.endswith('_all') else dataset_name
    if base_name not in _DATASETS:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    cls, classes = _DATASETS[base_name]
    if dataset_name.endswith('_all'):
        dss = []
        for cat in classes:
            kwargs['category'] = cat
            dss.append(cls(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
                transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs))
        return GlobalIndexConcatDataset(dss)
    else:
        return GlobalIndexConcatDataset([cls(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)])
