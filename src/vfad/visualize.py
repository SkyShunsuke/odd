import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
import matplotlib.pyplot as plt

from src.utils.misc import save_yaml_config
from src.flow_matching import VelocityField

import logging
logger = logging.getLogger(__name__)

def create_save_dir(save_dir: str):
    """Create the directory to save anomaly maps if it does not exist.
    Args:
        save_dir: Directory path to create.
    """
    if os.path.exists(save_dir):
        logger.warning(f"Save directory {save_dir} already exists. Overwriting ...")   
    else:
        logger.info(f"Creating save directory at {save_dir} ...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

def denormalize_image(img_tensor: torch.Tensor, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
    """Denormalize image tensor.
    Args:
        img_tensor: Normalized image tensor of shape (C, H, W) or (N, C, H, W).
        mean: Mean used for normalization.
        std: Standard deviation used for normalization.
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(img_tensor.device)
    denorm_img = img_tensor * std + mean
    denorm_img = torch.clamp(denorm_img, 0.0, 1.0)
    if denorm_img.size(0) == 1:
        denorm_img = denorm_img.squeeze(0)  # (C, H, W)
    return denorm_img

def save_histogram(
    scores: np.ndarray,
    gts: np.ndarray,
    save_path: str,
    title: str = 'Anomaly Score Histogram',
):
    """Save histogram of anomaly scores.
    Args:
        scores: Anomaly scores of shape (N,).
        gts: Ground truth labels of shape (N,).
        save_path: Path to save the histogram image.
        title: Title of the histogram plot.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(scores[gts == 0], bins=50, alpha=0.5, label='Normal', color='gray')
    plt.hist(scores[gts > 0], bins=50, alpha=0.5, label='Anomalous', color='salmon')
    plt.title(title)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_anomaly_maps(
    save_dir: str,
    img_scores_by_class: dict,
    img_gts_by_class: dict,
    px_scores_by_class: dict,
    px_gts_by_class: dict,
    org_img_by_class: dict,
    class_map: dict,
):
    """Save anomaly maps to the specified directory.
    Args:
        save_dir: Directory to save the anomaly maps.
        img_scores_by_class: Dictionary of image-level scores by class.
        img_gts_by_class: Dictionary of image-level ground truths by class.
        px_scores_by_class: Dictionary of pixel-level scores by class.
        px_gts_by_class: Dictionary of pixel-level ground truths by class.
        org_img_by_class: Dictionary of original images by class.
        class_map: Mapping from class labels to class names.
    Saved File Format:
        - save_dir/
            class_name_1/
                vis_config.yaml
                img_score_histogram.png
                px_score_histogram.png
                gt_masks/
                   xxx.png
                   ...
                anomaly_maps/
                   xxx.png
                   ...
                org_images/
                   xxx.png
                   ...
            class_name_2/
                ...
    """
    create_save_dir(save_dir)
    
    all_keys = list(px_gts_by_class.keys())
    for cls_label, cls_name in class_map.items():
        if cls_label not in all_keys:
            continue
        
        cls_save_dir = os.path.join(save_dir, cls_name)
        create_save_dir(cls_save_dir)
        
        # -- create sub-directories
        gt_mask_dir = os.path.join(cls_save_dir, 'gt_masks')
        anomap_dir = os.path.join(cls_save_dir, 'anomaly_maps')
        orgimg_dir = os.path.join(cls_save_dir, 'org_images')
        create_save_dir(gt_mask_dir)
        create_save_dir(anomap_dir)
        create_save_dir(orgimg_dir)
        
        px_scores = px_scores_by_class[cls_label]  # (N, H, W)
        img_scores = img_scores_by_class[cls_label]  # (N,)
        img_gts = img_gts_by_class[cls_label]        # (N,)
        px_gts = px_gts_by_class[cls_label]        # (N, H, W)
        org_imgs = org_img_by_class[cls_label]     # (N, C, H, W)
        
        num_samples = px_scores.shape[0]
        # class_min = px_scores.min()
        # class_max = px_scores.max()
        for idx in tqdm(range(num_samples), desc=f"Saving anomaly maps for class {cls_name}"):
            img_gt = img_gts[idx]
            
            # -- save ground truth mask
            gt_mask = px_gts[idx]  # (H, W)
            gt_mask_path = os.path.join(gt_mask_dir, f"{idx:04d}_{img_gt}.png")
            Image.fromarray(gt_mask).save(gt_mask_path)
            
            # -- save anomaly map
            anomap_path = os.path.join(anomap_dir, f"{idx:04d}_{img_gt}.png")
            sample_min, sample_max = px_scores[idx].min(), px_scores[idx].max()
            plt.imsave(anomap_path, px_scores[idx], cmap='viridis', vmin=sample_min, vmax=sample_max)
            
            # -- save original image
            org_img = org_imgs[idx]  # (H, W, C)
            org_img_path = os.path.join(orgimg_dir, f"{idx:04d}_{img_gt}.png")
            Image.fromarray(org_img).save(org_img_path) 
        
        # -- save score histogram
        hist_path = os.path.join(cls_save_dir, 'img_score_histogram.png')
        save_histogram(img_scores, img_gts, hist_path, title=f'Image-level Anomaly Score Histogram for Class: {cls_name}')
        hist_path = os.path.join(cls_save_dir, 'px_score_histogram.png')
        save_histogram(px_scores.flatten(), px_gts.flatten(), hist_path, title=f'Pixel-level Anomaly Score Histogram for Class: {cls_name}')
        
        # -- save visualization config
        vis_config_path = os.path.join(cls_save_dir, 'vis_config.yaml')
        vis_config = {
            'class_name': cls_name,
            'num_samples': num_samples,
            # 'anom_map_normalization': {
            #     'min_value': float(class_min),
            #     'max_value': float(class_max),
            # },
            'img_scores': list(zip(img_gts.tolist(), img_scores.tolist())),
        }
        save_yaml_config(vis_config, vis_config_path)
            
    logger.info(f"Anomaly maps saved to {save_dir}")

@torch.no_grad()
def generate_samples(
    vf: VelocityField,
    dataloader: torch.utils.data.DataLoader,
    num_samples_per_class: int = 1,
    device: torch.device = torch.device('cuda'),
    input_sz: tuple = (3, 256, 256),
    steps: int = 10,
    solver_name: str = 'midpoint',
    solver_params: dict = {},
    distributed: bool = False,
    denorm_type: str = 'imagenet',
):
    """Generate samples using the trained model.
    Args:
        model: Trained VelocityField model.
        dataloader: DataLoader for the dataset.
        num_samples_per_class: Number of samples to generate per class.
        device: Device to perform computation on.
    Returns:
        dict: Dictionary of generated samples by class.
    """
    was_train = vf.training
    vf.eval()
    
    samples_by_class = {}
    org_imgs_by_class = {}
    class_map = dataloader.dataset.datasets[0].labels_to_names
    num_classes = len(dataloader.dataset.datasets)
    
    if distributed:
        vf = vf.module
    
    for cls_label, _ in class_map.items():
        samples_by_class[cls_label] = []
        org_imgs_by_class[cls_label] = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Generating samples"):
        imgs = batch['img'].to(device)
        cls_labels = batch['clslabel']
        bs = len(imgs)
        
        # -- sample generation
        x0 = torch.randn((bs, ) + input_sz).to(device)
        x1 = vf.sample(x0, y=cls_labels.to(device), steps=steps, solver_name=solver_name, \
            solver_params=solver_params, start_t=0)  # (B, C, H, W)
        
        if denorm_type == 'imagenet':
            x1 = [denormalize_image(x1)[i].cpu() for i in range(bs)]  
            imgs = [denormalize_image(imgs)[i].cpu() for i in range(bs)]
        elif denorm_type == 'default':
            default_mu = [0.5, 0.5, 0.5]
            default_std = [0.5, 0.5, 0.5]
            x1 = [denormalize_image(x1, mean=default_mu, std=default_std)[i].cpu() for i in range(bs)]
            imgs = [denormalize_image(imgs, mean=default_mu, std=default_std)[i].cpu() for i in range(bs)]
        else:
            x1 = [x1[i].cpu() for i in range(bs)]  
            imgs = [imgs[i].cpu() for i in range(bs)]

        for i in range(bs):
            cls_label = cls_labels[i].item()
            if len(samples_by_class[cls_label]) < num_samples_per_class:
                samples_by_class[cls_label].append(x1[i])
                org_imgs_by_class[cls_label].append(imgs[i])
                
        # if all classes have enough samples, break
        all_done = (sum([len(samples_by_class[cls_label]) for cls_label in class_map.keys()]) \
            >= num_classes * num_samples_per_class)
        if all_done:
            tqdm.write("All classes have enough samples. Stopping generation.")
            break
    
    # del empty entries
    for cls_label in list(samples_by_class.keys()):
        if len(samples_by_class[cls_label]) == 0:
            del samples_by_class[cls_label]
            del org_imgs_by_class[cls_label]

    all_keys = list(samples_by_class.keys())
    for cls_label, _ in class_map.items():
        if cls_label not in all_keys:
            continue
        samples_by_class[cls_label] = torch.stack(samples_by_class[cls_label], dim=0)  # (N, C, H, W)
        org_imgs_by_class[cls_label] = torch.stack(org_imgs_by_class[cls_label], dim=0)  # (N, C, H, W)
            
    if was_train:
        vf.train()
    return samples_by_class, org_imgs_by_class
    
def save_samples(
    save_dir: str,
    samples_by_class: dict,
    org_imgs_by_class: dict,
    class_map: dict,
):
    """Save generated samples to the specified directory.
    Args:
        save_dir: Directory to save the generated samples.
        samples_by_class: Dictionary of generated samples by class.
        org_imgs_by_class: Dictionary of original images by class.
        class_map: Mapping from class labels to class names.
    Saved File Format:
        - save_dir/
            class_name_1/
                generated_samples/
                   xxx.png
                   ...
                org_images/
                   xxx.png
                   ...
            class_name_2/
                ...
    """
    create_save_dir(save_dir)
    
    all_keys = list(samples_by_class.keys())
    for cls_label, cls_name in class_map.items():
        if cls_label not in all_keys:
            continue
        
        cls_save_dir = os.path.join(save_dir, cls_name)
        create_save_dir(cls_save_dir)
        
        # -- create sub-directories
        gen_sample_dir = os.path.join(cls_save_dir, 'generated_samples')
        orgimg_dir = os.path.join(cls_save_dir, 'org_images')
        create_save_dir(gen_sample_dir)
        create_save_dir(orgimg_dir)
        
        samples = samples_by_class[cls_label]      # (N, C, H, W)
        org_imgs = org_imgs_by_class[cls_label]    # (N, C, H, W)
        
        num_samples = samples.shape[0]
        for idx in tqdm(range(num_samples), desc=f"Saving generated samples for class {cls_name}"):
            # -- save generated sample
            gen_sample = samples[idx]  # (C, H, W)
            gen_sample = gen_sample * 255.0
            gen_sample = gen_sample.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, C)
            gen_sample_path = os.path.join(gen_sample_dir, f"{idx:04d}.png")
            Image.fromarray(gen_sample).save(gen_sample_path)
            
            # -- save original image
            org_img = org_imgs[idx]  # (C, H, W)
            org_img = org_img * 255.0
            org_img = org_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, C)
            org_img_path = os.path.join(orgimg_dir, f"{idx:04d}.png")
            Image.fromarray(org_img).save(org_img_path)
            
        logger.info(f"Generated samples saved to {cls_save_dir}")
    
    
    