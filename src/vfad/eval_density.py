import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.utils.distributed import concat_all_gather, get_world_size, get_rank
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
def evaluate_density(
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
):
    """Evaluate on the given dataset with density estimation, returning AD metrics.
    Args:
        vf: VelocityField model for evaluation.
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
    steps = eval_params.get('steps', 1) if eval_params is not None else 1
    solver_name = eval_params.get('solver_name', 'euler') if eval_params is not None else 'euler'
    solver_params = eval_params.get('solver_params', {}) if eval_params is not None else {}
    img_score_agg = eval_params.get('img_score_agg', 'diff') if eval_params is not None else 'diff'
    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params is not None else ['img_auroc', 'px_auroc']
    
    exact_divergence = eval_params.get('exact_divergence', False) if eval_params is not None else False
    hte_acc = eval_params.get('hutchinson_samples', 10) if eval_params is not None else 10
    hte_bs = eval_params.get('hutchinson_batch_size', 20) if eval_params is not None else 20
    assert all([met in SUPPORTED_METRICS for met in eval_metrics]), f"Some evaluation metrics are not supported. Supported metrics: {SUPPORTED_METRICS}"
    if exact_divergence:
        raise NotImplementedError("Exact divergence computation is not implemented yet.")
    
    logger.info(f"Starting evaluation with {steps} steps using {solver_name} solver.")
    logger.info(f"Exact divergence: {exact_divergence}, Hutchinson samples: {hte_acc}")
    
    # -- evaluation loop
    N = len(dataloader.dataset)
    masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
    clslabels_all = np.zeros((N,), dtype=np.uint8)
    anom_labels_all = np.zeros((N,), dtype=np.uint8)
    nll_all = np.zeros((N, *img_sz), dtype=np.float32)
    org_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
    
    if distributed:
        # -- extract the underlying model from DDP
        vf = vf.module
    
    logger.info("Extracting features and computing anomaly scores...")
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        current_idx = step * dataloader.batch_size * get_world_size()
        
        # -- prepare data
        imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        anom_labels_local, anom_masks_local = batch['label'], batch['mask'] # (B,), (B, H, W)
        anom_labels_local, anom_masks_local = anom_labels_local.to(device, non_blocking=True), anom_masks_local.to(device, non_blocking=True)
        
        # -- extract features
        z1, _ = extract_features(fe, imgs_local, device)  # (B, c, h, w)
        z1 = norm_fn(z1)
        
        # -- inversion through the velocity field
        if use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                nll_local = -1 * vf.density(z1, clslabels_local, steps=steps, solver_name=solver_name, solver_params=solver_params, exact_divergence=exact_divergence, hutchinson_samples=hte_acc, hutchinson_bs=hte_bs)  # (B, c, h, w)
        else:
            nll_local = -1 * vf.density(z1, clslabels_local, steps=steps, solver_name=solver_name, solver_params=solver_params, exact_divergence=exact_divergence, hutchinson_samples=hte_acc, hutchinson_bs=hte_bs)  # (B, c, h, w)
        
        # -- share results across GPUs
        if distributed:
            nll = concat_all_gather(nll_local)
            anom_labels = concat_all_gather(anom_labels_local)
            cls_labels = concat_all_gather(clslabels_local)
            anom_masks = concat_all_gather(anom_masks_local)
            imgs = concat_all_gather(imgs_local)
        else:
            nll = nll_local
            anom_labels = anom_labels_local
            cls_labels = clslabels_local
            anom_masks = anom_masks_local
            imgs = imgs_local
        
        # -- compute anomaly scores
        nll = F.interpolate(
            nll.unsqueeze(1), size=img_sz, mode='bilinear', align_corners=False
        ).squeeze(1)  # (N, H, W)
        nll_np = nll.cpu().numpy()
        
        # -- store results
        bs = nll.shape[0]
        nll_all[current_idx: current_idx + bs] = nll_np
        clslabels_all[current_idx: current_idx + bs] = cls_labels.cpu().numpy().astype(np.uint8)
        masks_all[current_idx: current_idx + bs] = anom_masks.cpu().numpy().astype(np.uint8)
        anom_labels_all[current_idx: current_idx + bs] = (anom_labels.cpu().numpy() > 0).astype(np.uint8)

        imgs = denormalize_image(imgs) * 255.0
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # (B, H, W, C)
        org_imgs_all[current_idx: current_idx + bs] = imgs
        
    # -- compute anomaly scores
    px_scores = nll_all  # (N, H, W)
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
    org_imgs_by_class = divide_by_class(org_imgs_all, clslabels_all)
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
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # -- compute image-level metrics
    logger.info("Calculating image-level metrics...")
    clsname_map = dataloader.dataset.datasets[0].labels_to_names
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
    
    # -- compute pixel-level metrics
    logger.info("Calculating pixel-level metrics...")
    px_metrics = [met for met in eval_metrics if met.startswith('px_')]
    for cls_label in px_scores_by_class.keys():
        cls_px_scores = px_scores_by_class[cls_label]
        cls_px_gts = px_gts_by_class[cls_label]
        cls_px_metrics = calculate_px_metrics(
            gt_masks=cls_px_gts,
            pred_scores=cls_px_scores,
            metrics=px_metrics,
        )
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