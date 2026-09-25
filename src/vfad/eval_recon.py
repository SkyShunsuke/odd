import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.utils.distributed import concat_all_gather, get_rank, get_world_size
from src.flow_matching import VelocityField
from src.utils.adeval.eval_utils import (
    calculate_img_metrics,
    calculate_px_metrics,
    divide_by_class,
    extract_features,
    aggregate_px_values,
    SUPPORTED_METRICS
)
from src.vfad.visualize import save_anomaly_maps, denormalize_image

import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_recon(
    vf: VelocityField, 
    fe: torch.nn.Module,
    norm_fn: callable,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device, 
    img_sz: tuple,
    verbose: bool = True,
    use_bfloat16: bool = False,
    distributed: bool = False,
    eval_params: dict = None,
    save_anomaps: bool = False,
    save_dir: str = None,
    img_metric_only: bool = False,
    train_loader=None,
    noise_shape=None,
    **kwargs
):
    """Evaluate on the given dataset with inversion-informed reconstruction, returning AD metrics.
    Args:
        vf: VelocityField model for evaluation.
        fe: Feature extractor model.
        dataloader: DataLoader providing the evaluation dataset.
        device: Device to perform evaluation on.
        img_sz: Size of the input images (H, W).
        verbose: Whether to display progress bar.
        use_bfloat16: Whether to use bfloat16 precision during evaluation.
        distributed: Whether to use distributed evaluation.
        eval_params (dict, optional): Additional evaluation parameters. Defaults to None.
    Returns:
        dict: Dictionary containing evaluation metrics (e.g., AUROC, AUPR, F1-score) for each class.
    """
    was_train = vf.training
    vf.eval()
    
    # -- evaluation configuration
    solver_name = eval_params.get('solver_name', 'euler') if eval_params is not None else 'euler'
    solver_params = eval_params.get('solver_params', {}) if eval_params is not None else {}
    img_score_agg = eval_params.get('img_score_agg', 'diff') if eval_params is not None else 'diff'
    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params is not None else ['img_auroc', 'px_auroc']
    
    if img_metric_only:
        eval_metrics = [met for met in eval_metrics if met.startswith('img_')]
    
    inv_steps = eval_params.get('inv_steps', 1) if eval_params is not None else 1
    recon_steps = eval_params.get('recon_steps', 1) if eval_params is not None else 1
    inv_stop_time = eval_params.get('inv_stop_time', 0.0) if eval_params is not None else 0.0
    
    assert all([met in SUPPORTED_METRICS for met in eval_metrics]), f"Some evaluation metrics are not supported. Supported metrics: {SUPPORTED_METRICS}"
    
    logger.info(f"Starting evaluation with inversion steps: {inv_steps}, reconstruction steps: {recon_steps}, using {solver_name} solver.")
    logger.info(f"Inversion stop time: {inv_stop_time}") 
    
    # -- evaluation loop
    N = len(dataloader.dataset)
    masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
    clslabels_all = np.zeros((N,), dtype=np.uint8)
    anom_types_all = np.zeros((N,), dtype=np.uint8)
    anom_labels_all = np.zeros((N,), dtype=np.uint8)
    mse_all = np.zeros((N, *img_sz), dtype=np.float32)
    org_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
    recon_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)

    if distributed:
        # -- extract the underlying model from DDP
        vf = vf.module
        
    print("▶️Closed-form velocity fields constructed for each class.")
    
    logger.info("Extracting features and computing anomaly scores...")
    noise_base = torch.randn((1, *noise_shape), device=device)
    perturb_t = torch.tensor([inv_stop_time], device=device)
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        
        # -- prepare data
        imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        anom_labels_local, anom_masks_local = batch['label'], batch['mask'] # (B,), (B, H, W)
        anom_labels_local, anom_masks_local = anom_labels_local.to(device, non_blocking=True), anom_masks_local.to(device, non_blocking=True)
        idx_local = batch["index"].to(device, non_blocking=True)  # (B,)
        
        # -- extract features
        z1, _ = extract_features(fe, imgs_local, device)  # (B, c, h, w)
        z1 = norm_fn(z1)
        
        # -- pertubation
        zt = vf.perturb(z1, noise_base.repeat(len(z1), 1, 1, 1), perturb_t.repeat(len(z1))) # (B, c, h, w)
        
        # -- reconstruction through the velocity field
        if use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z1_local = vf.sample(zt, clslabels_local, steps=recon_steps, solver_name=solver_name, solver_params=solver_params, start_t=inv_stop_time)  # (B, c, h, w)
        else:
            z1_local = vf.sample(zt, clslabels_local, steps=recon_steps, start_t=inv_stop_time)
            # z1_local = vf.sample_ddim(zt, clslabels_local, steps=recon_steps, solver_name=solver_name, solver_params=solver_params, start_t=inv_stop_time)  # (B, c, h, w)
            # z1_local = vf.reconstruct_edm(
            #     z1, sigma_start=eval_params.get('sigma_start', 0.5), steps=recon_steps, noise_base=noise_base.repeat(len(z1), 1, 1, 1)
            # )
        # -- compute reconstruction error as anomaly score
        mse_local = F.mse_loss(z1_local, z1, reduction='none').sum(dim=1)  # (B, h, w)
        imgs_local = denormalize_image(imgs_local) * 255.0
        
        # -- share results across GPUs
        if distributed:
            mse = concat_all_gather(mse_local)
            anom_labels = concat_all_gather(anom_labels_local)
            cls_labels = concat_all_gather(clslabels_local)
            anom_masks = concat_all_gather(anom_masks_local)
            imgs = concat_all_gather(imgs_local)
            idx = concat_all_gather(idx_local)
        else:
            mse = mse_local
            anom_labels = anom_labels_local
            cls_labels = clslabels_local
            anom_masks = anom_masks_local
            imgs = imgs_local
            idx = idx_local
        
        # -- compute anomaly scores
        mse = F.interpolate(
            mse.unsqueeze(1), size=img_sz, mode='bilinear', align_corners=False
        ).squeeze(1)  # (N, H, W)
        mse_np = mse.cpu().numpy()
        
        # -- reshape masks
        anom_masks = anom_masks.squeeze(1)  # (N, H, W)
        anom_masks = F.interpolate(
            anom_masks.unsqueeze(1).float(), size=img_sz, mode='nearest'
        ).squeeze(1).long()  # (N, H, W)
        
        # -- store results
        idx_np = idx.cpu().numpy()
        mse_all[idx_np] = mse_np
        clslabels_all[idx_np] = cls_labels.cpu().numpy().astype(np.uint8)
        masks_all[idx_np] = anom_masks.cpu().numpy().astype(np.uint8)
        anom_labels_all[idx_np] = (anom_labels.cpu().numpy() > 0).astype(np.uint8)
    
    if distributed:
        torch.distributed.barrier()
    
    # -- compute anomaly scores
    px_scores = mse_all  # (N, H, W)
    img_scores = aggregate_px_values(
        agg_method=img_score_agg,
        px_values=px_scores
    )  # (N,)
    px_gts = masks_all  # (N, H, W)
    img_gts = anom_labels_all  # (N,)
    
    # -- divide by class
    img_scores_by_class = divide_by_class(img_scores, clslabels_all)
    img_gts_by_class = divide_by_class(img_gts, clslabels_all)
    px_scores_by_class = divide_by_class(px_scores, clslabels_all)
    px_gts_by_class = divide_by_class(px_gts, clslabels_all)

    clsname_map = dataloader.dataset.datasets[0].labels_to_names

    # -- save anomaly maps if required
    if save_anomaps and save_dir is not None and get_rank() == 0:
        save_anomaly_maps(
            save_dir=save_dir,
            img_scores_by_class=img_scores_by_class,
            img_gts_by_class=img_gts_by_class,
            px_scores_by_class=px_scores_by_class,
            px_gts_by_class=px_gts_by_class,
            org_img_by_class=org_imgs_by_class,
            class_map=clsname_map,
            recon_imgs_by_class=recon_imgs_by_class,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


    logger.info("Calculating image-level metrics...")
    eval_results = {v : {} for v in clsname_map.values()}
    img_metrics = [met for met in eval_metrics if met.startswith('img_')]
    for cls_label in img_scores_by_class.keys():
        
        cls_img_scores = img_scores_by_class[cls_label]
        cls_img_gts = img_gts_by_class[cls_label]
        cls_img_metrics = calculate_img_metrics(
            gt_labels=cls_img_gts,
            pred_scores=cls_img_scores,
            metrics=img_metrics,
        )
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_img_metrics.items():
            eval_results[cls_name][met_name] = met_value
    
    if img_metric_only:
        eval_results['average'] = {}
        for cls_name in eval_results.keys():
            for met_name in eval_results[cls_name].keys():
                if cls_name == 'average':
                    continue
                if met_name not in eval_results['average']:
                    eval_results['average'][met_name] = 0.0
                eval_results['average'][met_name] += eval_results[cls_name][met_name]
        num_classes = len(dataloader.dataset.datasets)
        for met_name in eval_results['average'].keys():
            eval_results['average'][met_name] /= num_classes
        
        # -- calculate averaged metrics (mad) for all classes
        del_classes = []
        for cls_name in eval_results.keys():
            mad_sum = 0.0
            for met_name, met_value in eval_results[cls_name].items():
                mad_sum += met_value
            if len(eval_results[cls_name]) > 0:
                eval_results[cls_name]['mad'] = mad_sum / len(eval_results[cls_name])
            else:
                del_classes.append(cls_name)
        
        for cls_name in del_classes:
            del eval_results[cls_name]
        
        logger.info(f"Evaluation completed. \\ Results: {eval_results}")
        
        if was_train:
            vf.train()
        return eval_results
    
    # -- compute pixel-level metrics
    logger.info("Calculating pixel-level metrics...")
    px_metrics = [met for met in eval_metrics if met.startswith('px_')]
    for cls_label in px_scores_by_class.keys():
        
        cls_px_scores = px_scores_by_class[cls_label]
        cls_px_gts = px_gts_by_class[cls_label]  # (N_cls, H, W)
        # cls_latents = latents_by_class[cls_label]  # (N_cls, c, h, w)
        # cls_anom_px_scores = cls_px_scores[cls_px_gts > 0.5]  # (N_anom_px,)
        # cls_normal_px_scores = cls_px_scores[cls_px_gts < 0.5]  # (N_normal_px,)
        # cls_anom_scores_mean = cls_anom_px_scores.mean() if len(cls_anom_px_scores) > 0 else 0.0
        # cls_normal_gts_mean = cls_normal_px_scores.mean() if len(cls_normal_px_scores) > 0 else 0.0
        # logger.info(f"Class {clsname_map[cls_label]}: Anomalous pixel scores mean: {cls_anom_scores_mean:.4f}, Normal pixel scores mean: {cls_normal_gts_mean:.4f}")
        
        cls_px_metrics = {
            k: float(v)
            for k, v in calculate_px_metrics(
                gt_masks=cls_px_gts,
                pred_scores=cls_px_scores,
                metrics=px_metrics,
            ).items()
        }
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_px_metrics.items():
            eval_results[cls_name][met_name] = met_value
        
    # -- average over classes
    eval_results['average'] = {}
    for cls_name in eval_results.keys():
        for met_name in eval_results[cls_name].keys():
            if cls_name == 'average':
                continue
            if met_name not in eval_results['average']:
                eval_results['average'][met_name] = 0.0
            eval_results['average'][met_name] += eval_results[cls_name][met_name]
    num_classes = len(dataloader.dataset.datasets)
    for met_name in eval_results['average'].keys():
        eval_results['average'][met_name] /= num_classes

    # -- calculate averaged metrics (mad) for all classes
    del_classes = []
    for cls_name in eval_results.keys():
        mad_sum = 0.0
        for met_name, met_value in eval_results[cls_name].items():
            mad_sum += met_value
        if len(eval_results[cls_name]) > 0:
            eval_results[cls_name]['mad'] = mad_sum / len(eval_results[cls_name])
        else:
            del_classes.append(cls_name)
    
    for cls_name in del_classes:
        del eval_results[cls_name]
    
    logger.info(f"Evaluation completed. \\ Results: {eval_results}")
    
    if was_train:
        vf.train()
    return eval_results

def evaluate_recon_grad(
    vf: VelocityField, 
    fe: torch.nn.Module,
    norm_fn: callable,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device, 
    img_sz: tuple,
    verbose: bool = True,
    use_bfloat16: bool = False,
    distributed: bool = False,
    eval_params: dict = None,
    save_anomaps: bool = False,
    save_dir: str = None,
    img_metric_only: bool = False,
    train_loader=None,
    noise_shape=None,
    **kwargs
):
    """Evaluate on the given dataset with inversion-informed reconstruction, returning AD metrics.
    Args:
        vf: VelocityField model for evaluation.
        fe: Feature extractor model.
        dataloader: DataLoader providing the evaluation dataset.
        device: Device to perform evaluation on.
        img_sz: Size of the input images (H, W).
        verbose: Whether to display progress bar.
        use_bfloat16: Whether to use bfloat16 precision during evaluation.
        distributed: Whether to use distributed evaluation.
        eval_params (dict, optional): Additional evaluation parameters. Defaults to None.
    Returns:
        dict: Dictionary containing evaluation metrics (e.g., AUROC, AUPR, F1-score) for each class.
    """
    was_train = vf.training
    vf.eval()
    
    # -- evaluation configuration
    solver_name = eval_params.get('solver_name', 'euler') if eval_params is not None else 'euler'
    solver_params = eval_params.get('solver_params', {}) if eval_params is not None else {}
    img_score_agg = eval_params.get('img_score_agg', 'diff') if eval_params is not None else 'diff'
    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params is not None else ['img_auroc', 'px_auroc']
    
    if img_metric_only:
        eval_metrics = [met for met in eval_metrics if met.startswith('img_')]
    
    inv_steps = eval_params.get('inv_steps', 1) if eval_params is not None else 1
    recon_steps = eval_params.get('recon_steps', 1) if eval_params is not None else 1
    inv_stop_time = eval_params.get('inv_stop_time', 0.0) if eval_params is not None else 0.0
    
    # -- ttt parameters
    grad_steps = eval_params.get('grad_steps', 1) if eval_params is not None else 1
    grad_lr = eval_params.get('grad_lr', 1e-3) if eval_params is not None else 1e-3
        
    assert all([met in SUPPORTED_METRICS for met in eval_metrics]), f"Some evaluation metrics are not supported. Supported metrics: {SUPPORTED_METRICS}"
    
    logger.info(f"Starting evaluation with inversion steps: {inv_steps}, reconstruction steps: {recon_steps}, using {solver_name} solver.")
    logger.info(f"Inversion stop time: {inv_stop_time}") 
    
    # -- evaluation loop
    N = len(dataloader.dataset)
    masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
    clslabels_all = np.zeros((N,), dtype=np.uint8)
    anom_types_all = np.zeros((N,), dtype=np.uint8)
    anom_labels_all = np.zeros((N,), dtype=np.uint8)
    mse_all = np.zeros((N, *img_sz), dtype=np.float32)
    org_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
    recon_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)

    if distributed:
        # -- extract the underlying model from DDP
        vf = vf.module
        
    print("▶️Closed-form velocity fields constructed for each class.")
    
    logger.info("Extracting features and computing anomaly scores...")
    noise_base = torch.randn((1, *noise_shape), device=device)
    perturb_t = torch.tensor([inv_stop_time], device=device)
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        
        # -- prepare data
        imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        anom_labels_local, anom_masks_local = batch['label'], batch['mask'] # (B,), (B, H, W)
        anom_labels_local, anom_masks_local = anom_labels_local.to(device, non_blocking=True), anom_masks_local.to(device, non_blocking=True)
        idx_local = batch["index"].to(device, non_blocking=True)  # (B,)
        
        # -- extract features
        with torch.no_grad():
            z1, _ = extract_features(fe, imgs_local, device)  # (B, c, h, w)
            z1 = norm_fn(z1)
        
        # -- pertubation
        noise = noise_base.repeat(len(z1), 1, 1, 1)
        ttt_optim = torch.optim.Adam([noise], lr=grad_lr)
        
        for gs in range(grad_steps):
            zt = vf.perturb(z1, noise, perturb_t.repeat(len(z1))) # (B, c, h, w)
            z1_local = vf.sample(zt, clslabels_local, steps=recon_steps, start_t=inv_stop_time)
            # -- compute reconstruction error as anomaly score
            loss = F.mse_loss(z1_local, z1.detach(), reduction='none').sum()
            
            ttt_optim.zero_grad()
            loss.backward()
            ttt_optim.step()
            
            noise_norm = torch.norm(noise).item()
            logger.info(f"Step {gs+1}/{grad_steps}: Loss={loss.item():.6f}, Noise Norm={noise_norm:.6f}")
        
        mse_local = F.mse_loss(z1_local, z1, reduction='none').sum(dim=1)  # (B, h, w)
        
        imgs_local = denormalize_image(imgs_local) * 255.0
        
        # -- share results across GPUs
        if distributed:
            mse = concat_all_gather(mse_local)
            anom_labels = concat_all_gather(anom_labels_local)
            cls_labels = concat_all_gather(clslabels_local)
            anom_masks = concat_all_gather(anom_masks_local)
            imgs = concat_all_gather(imgs_local)
            idx = concat_all_gather(idx_local)
        else:
            mse = mse_local
            anom_labels = anom_labels_local
            cls_labels = clslabels_local
            anom_masks = anom_masks_local
            imgs = imgs_local
            idx = idx_local
        
        # -- compute anomaly scores
        mse = F.interpolate(
            mse.unsqueeze(1), size=img_sz, mode='bilinear', align_corners=False
        ).squeeze(1)  # (N, H, W)
        mse_np = mse.cpu().numpy()
        
        # -- reshape masks
        anom_masks = anom_masks.squeeze(1)  # (N, H, W)
        anom_masks = F.interpolate(
            anom_masks.unsqueeze(1).float(), size=img_sz, mode='nearest'
        ).squeeze(1).long()  # (N, H, W)
        
        # -- store results
        idx_np = idx.cpu().numpy()
        mse_all[idx_np] = mse_np
        clslabels_all[idx_np] = cls_labels.cpu().numpy().astype(np.uint8)
        masks_all[idx_np] = anom_masks.cpu().numpy().astype(np.uint8)
        anom_labels_all[idx_np] = (anom_labels.cpu().numpy() > 0).astype(np.uint8)
    
    if distributed:
        torch.distributed.barrier()
    
    # -- compute anomaly scores
    px_scores = mse_all  # (N, H, W)
    img_scores = aggregate_px_values(
        agg_method=img_score_agg,
        px_values=px_scores
    )  # (N,)
    px_gts = masks_all  # (N, H, W)
    img_gts = anom_labels_all  # (N,)
    
    # -- divide by class
    img_scores_by_class = divide_by_class(img_scores, clslabels_all)
    img_gts_by_class = divide_by_class(img_gts, clslabels_all)
    px_scores_by_class = divide_by_class(px_scores, clslabels_all)
    px_gts_by_class = divide_by_class(px_gts, clslabels_all)

    clsname_map = dataloader.dataset.datasets[0].labels_to_names

    # -- save anomaly maps if required
    if save_anomaps and save_dir is not None and get_rank() == 0:
        save_anomaly_maps(
            save_dir=save_dir,
            img_scores_by_class=img_scores_by_class,
            img_gts_by_class=img_gts_by_class,
            px_scores_by_class=px_scores_by_class,
            px_gts_by_class=px_gts_by_class,
            org_img_by_class=org_imgs_by_class,
            class_map=clsname_map,
            recon_imgs_by_class=recon_imgs_by_class,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


    logger.info("Calculating image-level metrics...")
    eval_results = {v : {} for v in clsname_map.values()}
    img_metrics = [met for met in eval_metrics if met.startswith('img_')]
    for cls_label in img_scores_by_class.keys():
        
        cls_img_scores = img_scores_by_class[cls_label]
        cls_img_gts = img_gts_by_class[cls_label]
        cls_img_metrics = calculate_img_metrics(
            gt_labels=cls_img_gts,
            pred_scores=cls_img_scores,
            metrics=img_metrics,
        )
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_img_metrics.items():
            eval_results[cls_name][met_name] = met_value
    
    if img_metric_only:
        eval_results['average'] = {}
        for cls_name in eval_results.keys():
            for met_name in eval_results[cls_name].keys():
                if cls_name == 'average':
                    continue
                if met_name not in eval_results['average']:
                    eval_results['average'][met_name] = 0.0
                eval_results['average'][met_name] += eval_results[cls_name][met_name]
        num_classes = len(dataloader.dataset.datasets)
        for met_name in eval_results['average'].keys():
            eval_results['average'][met_name] /= num_classes
        
        # -- calculate averaged metrics (mad) for all classes
        del_classes = []
        for cls_name in eval_results.keys():
            mad_sum = 0.0
            for met_name, met_value in eval_results[cls_name].items():
                mad_sum += met_value
            if len(eval_results[cls_name]) > 0:
                eval_results[cls_name]['mad'] = mad_sum / len(eval_results[cls_name])
            else:
                del_classes.append(cls_name)
        
        for cls_name in del_classes:
            del eval_results[cls_name]
        
        logger.info(f"Evaluation completed. \\ Results: {eval_results}")
        
        if was_train:
            vf.train()
        return eval_results
    
    # -- compute pixel-level metrics
    logger.info("Calculating pixel-level metrics...")
    px_metrics = [met for met in eval_metrics if met.startswith('px_')]
    for cls_label in px_scores_by_class.keys():
        
        cls_px_scores = px_scores_by_class[cls_label]
        cls_px_gts = px_gts_by_class[cls_label]  # (N_cls, H, W)
        cls_px_metrics = {
            k: float(v)
            for k, v in calculate_px_metrics(
                gt_masks=cls_px_gts,
                pred_scores=cls_px_scores,
                metrics=px_metrics,
            ).items()
        }
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_px_metrics.items():
            eval_results[cls_name][met_name] = met_value
        
    # -- average over classes
    eval_results['average'] = {}
    for cls_name in eval_results.keys():
        for met_name in eval_results[cls_name].keys():
            if cls_name == 'average':
                continue
            if met_name not in eval_results['average']:
                eval_results['average'][met_name] = 0.0
            eval_results['average'][met_name] += eval_results[cls_name][met_name]
    num_classes = len(dataloader.dataset.datasets)
    for met_name in eval_results['average'].keys():
        eval_results['average'][met_name] /= num_classes

    # -- calculate averaged metrics (mad) for all classes
    del_classes = []
    for cls_name in eval_results.keys():
        mad_sum = 0.0
        for met_name, met_value in eval_results[cls_name].items():
            mad_sum += met_value
        if len(eval_results[cls_name]) > 0:
            eval_results[cls_name]['mad'] = mad_sum / len(eval_results[cls_name])
        else:
            del_classes.append(cls_name)
    
    for cls_name in del_classes:
        del eval_results[cls_name]
    
    logger.info(f"Evaluation completed. \\ Results: {eval_results}")
    
    if was_train:
        vf.train()
    return eval_results