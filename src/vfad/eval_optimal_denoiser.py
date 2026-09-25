import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import os

import torch
import torch.distributed as dist
import numpy as np

import yaml
import logging
import pandas as pd

from src.models import init_model
from src.datasets import build_dataset
from src.backbones import get_backbone, get_backbone_feature_shape, get_normalization_func
from src.flow_matching import LocalizedClosedFM, VelocityField
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size, concat_all_gather
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.utils.adeval.eval_utils import (
    calculate_img_metrics,
    calculate_px_metrics,
    divide_by_class,
    extract_features,
    aggregate_px_values,
    SUPPORTED_METRICS
)
from src.vfad.visualize import save_anomaly_maps, denormalize_image
from src.utils.opt.optimizer import load_model_only

import logging
logger = logging.getLogger(__name__)


class TiedSphericalGaussianMixture(GaussianMixture):
    """
    GMM with a common spherical covariance:

        Sigma_k = sigma^2 I   for all k
    """

    def __init__(self, n_components=1, **kwargs):
        super().__init__(
            n_components=n_components,
            covariance_type="spherical",
            **kwargs
        )

    def _m_step(self, X, log_resp):
        # Standard spherical GMM M-step
        super()._m_step(X, log_resp)

        # Pool component-wise variances:
        # sigma^2 = sum_k pi_k sigma_k^2
        sigma2 = np.dot(self.weights_, self.covariances_)

        # Force all components to share the same spherical variance
        self.covariances_[:] = sigma2
        self.precisions_cholesky_[:] = 1.0 / np.sqrt(sigma2)

    def _n_parameters(self):
        """
        Correct number of free parameters for AIC/BIC.

        means: K * D
        weights: K - 1
        common variance: 1
        """
        _, n_features = self.means_.shape

        return (
            self.n_components * n_features
            + (self.n_components - 1)
            + 1
        )

def main(params, args):
    # init distributed mode
    init_distributed_mode(args)

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device('cuda:%s'%args.gpu)
    
    # -- setup logging
    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

    os.makedirs(params['logging']['log_dir'], exist_ok=True)
    
    # -- make logging stuff
    if is_main():
        log_dir = params['logging']['log_dir']
        
        # save config file
        config_save_path = os.path.join(log_dir, 'eval_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(params, f)    
    else:
        log_dir = params['logging']['log_dir']
    
    # set seed
    seed = params['meta']['global_seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Building datasets... with config: \\ {params['data']}")
    test_bs = params['data'].get('test_batch_size', 8)
    train_dataset = build_dataset(train=True, **params['data'])
    test_dataset = build_dataset(train=False, **params['data'])
    
    num_classes = len(test_dataset.datasets)
    logger.info(f"Number of classes: {num_classes}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=8,
        pin_memory=params['data'].get('pin_memory', True),
        num_workers=params['data'].get('num_workers', 4),
        persistent_workers=params['data'].get('persistent_workers', True),
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=test_bs,
        pin_memory=params['data'].get('pin_memory', True),
        num_workers=params['data'].get('num_workers', 4),
        persistent_workers=params['data'].get('persistent_workers', True),
        drop_last=False,
    )
    logger.info(f"Data loaders built. Number of evaluation samples: {len(test_dataset)}, "
                f"Number of test samples: {len(test_dataset)}, ")
    
    # build model    
    feat_sz = get_backbone_feature_shape(model_name=params['model']['backbone']['model_name'],)
    fe = get_backbone(**params['model']['backbone'])
    feat_norm_method = params['model']['backbone'].get('normalization', None)

    norm_fn = get_normalization_func(feat_norm_method)
    fe.to(device).eval()
    logger.info(f"Backbone {params['model']['backbone']['model_name']} initialized.")

    logger.info(f"Using input shape {feat_sz} for the flow matching.")
    logger.info(f"Extract latents from the backbone")
    
    all_latents = []
    for batch in tqdm(train_loader, desc="Extracting latents"):
        imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        with torch.no_grad():
            feats_local, _ = fe(imgs_local)    # (B, C, H', W')
            feats_local = norm_fn(feats_local)
            
        feats = concat_all_gather(feats_local)    # (B_total, C, H', W')
        all_latents.append(feats.cpu())
    all_latents = torch.cat(all_latents, dim=0)    # (N_total, C, H', W')
    
    # fit gmm based on training set
    n_components = 4  # number of GMM components, you can adjust this based on your needs
    gmm_train_ratio = 0.8  # use 80% of the data for training the GMM
    gmm_data_np = all_latents.permute(0, 2, 3, 1).contiguous().view(-1, all_latents.size(1)).numpy()
    gmm_train_size = int(gmm_train_ratio * gmm_data_np.shape[0])
    gmm_train_data = gmm_data_np[:gmm_train_size]
    gmm_val_data = gmm_data_np[gmm_train_size:]

    gmm = TiedSphericalGaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(gmm_train_data)
    logger.info(f"GMM fitted with {n_components} components.")
    
    # Evaluate approximation error
    gmm_train_log_likelihood = gmm.score(gmm_train_data)
    gmm_val_log_likelihood = gmm.score(gmm_val_data)
    logger.info(f"GMM train log-likelihood: {gmm_train_log_likelihood}")
    logger.info(f"GMM validation log-likelihood: {gmm_val_log_likelihood}")
    

    # # Each point represents one spatial feature vector, not one image.
    # # Subsample to keep visualization fast.
    # rng = np.random.default_rng(0)
    # n_plot = min(30_000, len(gmm_data_np))
    # indices = rng.choice(len(gmm_data_np), size=n_plot, replace=False)
    # X_plot = gmm_data_np[indices]

    # # Assign clusters using the already-fitted GMM.
    # cluster_ids = gmm.predict(X_plot)

    # # Project features and GMM means using the same PCA.
    # pca = PCA(n_components=2, random_state=0)
    # X_2d = pca.fit_transform(X_plot)
    # means_2d = pca.transform(gmm.means_)

    # fig, ax = plt.subplots(figsize=(9, 7))
    # colors = plt.get_cmap("tab10")

    # for k in range(n_components):
    #     mask = cluster_ids == k
    #     color = colors(k % 10)

    #     ax.scatter(
    #         X_2d[mask, 0],
    #         X_2d[mask, 1],
    #         s=6,
    #         alpha=0.3,
    #         color=color,
    #         edgecolors="none",
    #         rasterized=True,
    #         label=f"Cluster {k}",
    #     )

    #     ax.scatter(
    #         means_2d[k, 0],
    #         means_2d[k, 1],
    #         s=220,
    #         marker="X",
    #         color=color,
    #         edgecolors="black",
    #         linewidths=1.5,
    #         zorder=5,
    #     )

    # variance = pca.explained_variance_ratio_ * 100
    # ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    # ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    # ax.set_title("Latent features colored by GMM cluster\nX = projected GMM mean")
    # ax.legend(markerscale=3)
    # ax.grid(alpha=0.15)
    
    # # save as PDF
    # pdf_path = 'latents_pca.pdf'
    # fig.savefig(pdf_path)
    # logger.info(f"Saved PCA visualization to {pdf_path}")
    # plt.close(fig)
    
    vf = LocalizedClosedFM(all_latents).to(device)

    # logger.info(f"Using input shape {feat_sz} for the flow matching.")
    # pred_type, loss_type = params['flow_matching'].get('pred_type', 'velocity'), params['flow_matching'].get('loss_type', 'velocity')
    # train_steps = params['flow_matching'].get('train_steps', -1)
    # t_scheduler_train = params['flow_matching']['scheduler'].get('t_scheduler_train', 'linear')
    # t_scheduler_infer = params['flow_matching']['scheduler'].get('t_scheduler_infer', 'linear')
    # t_mu = params['flow_matching']['scheduler'].get('t_mu', 0.0)
    # t_sigma = params['flow_matching']['scheduler'].get('t_sigma', 1.0)
    # div_eps = params['flow_matching'].get('div_eps', 0.05)
    # logger.info(f"Flow Matching Settings: t_scheduler_train: {t_scheduler_train}, t_scheduler_infer: {t_scheduler_infer}, t_mu: {t_mu}, t_sigma: {t_sigma}, div_eps: {div_eps}")
    # logger.info(f"Flow Matching Prediction Type: {pred_type}, Loss Type: {loss_type}")
    # logger.info(f"Using partial time sampling with {train_steps} training steps." if train_steps > 0 else "Using full time sampling.")
    # model = init_model(input_sz=feat_sz, num_classes=15, **params['model']).to(device)
    # vf_learned = VelocityField(
    #     model=model,
    #     input_sz=feat_sz,
    #     scheduler_name=params['flow_matching']['scheduler']['name'],
    #     solver_name=params['flow_matching']['solver']['name'],
    #     loss_type=loss_type,
    #     pred_type=pred_type,
    #     train_steps=train_steps,
    #     t_scheduler_train=t_scheduler_train,
    #     t_scheduler_infer=t_scheduler_infer,
    #     t_mu=t_mu,
    #     t_sigma=t_sigma,
    #     div_eps=div_eps,
    #     scheduler_params=params['flow_matching']['scheduler'].get('params', None),
    #     solver_params=params['flow_matching']['solver'].get('params', None),
    # )
    # logger.info(f"Velocity Field Model {params['model']['model_name']} has been initialized.")
    # resume_path = params['resume']['resume_path']
    # assert resume_path is not None, "Please specify the checkpoint path for evaluation."
    # assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!, Please check the path."
    
    # vf_learned = load_model_only(
    #     resume_path, vf_learned
    # )
    # logger.info(f"Resumed from checkpoint: {resume_path}")

    eval_params = params['logging']['eval']

    # -- evaluate
    logger.info(f"Starting evaluation...")
    ablation_steps = eval_params.get('ablation_steps', [-1])
    do_ablation = ablation_steps != [-1]
    
    if do_ablation:
        logger.info(f"Performing ablation study with steps: {ablation_steps}")
        for step in ablation_steps:
            logger.info(f"Evaluating with {step} steps...")
            eval_params["recon_steps"] = step
            eval_results = eval_recon(
                vf=vf,
                gmm=gmm,
                fe=fe,
                norm_fn=norm_fn,
                dataloader=test_loader,
                device=device,
                img_sz=(params['data']['img_size'], params['data']['img_size']),
                verbose=True,
                use_bfloat16=params['meta']['use_bfloat16'],
                distributed=args.distributed,
                save_anomaps=eval_params.get('save_anomaps', False),
                save_dir=eval_params.get('save_dir', None),
                eval_params=eval_params,
                noise_shape=feat_sz,
            )
            csv_path = os.path.join(eval_params.get('save_dir') or os.path.join(log_dir, 'results'), f'eval_results_{step}.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            df = pd.DataFrame.from_dict(eval_results, orient='index')
            df.index.name = 'category'
            df.reset_index(inplace=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved evaluation results to {csv_path}")
    else:
        eval_results = eval_recon(
            vf=vf,
            gmm=gmm,
            fe=fe,
            norm_fn=norm_fn,
            dataloader=test_loader,
            device=device,
            img_sz=(params['data']['img_size'], params['data']['img_size']),
            verbose=True,
            use_bfloat16=params['meta']['use_bfloat16'],
            distributed=args.distributed,
            save_anomaps=eval_params.get('save_anomaps', False),
            save_dir=eval_params.get('save_dir', None),
            eval_params=eval_params,
            noise_shape=feat_sz,
        )
    
    # -- save results
    if is_main():
        results_save_path = os.path.join(log_dir, 'eval_results.yaml')
        with open(results_save_path, 'w') as f:
            yaml.dump(eval_results, f)
        logger.info(f"Saved evaluation results to {results_save_path}")
        
    # -- close distributed process
    dist.barrier()
    dist.destroy_process_group()

    # -- end of main
    logger.info(f"Evaluation completed. Evaluation results are saved at {log_dir}")

@torch.no_grad()
def eval_recon(
    vf, gmm, fe, norm_fn, dataloader, device, img_sz, verbose=True, use_bfloat16=False,
    distributed=False, save_anomaps=False, save_dir=None, eval_params=None, noise_shape=None,
):

    img_score_agg = eval_params.get('img_score_agg', 'diff') if eval_params is not None else 'diff'
    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params is not None else ['img_auroc', 'px_auroc']
    
    recon_steps = eval_params.get('recon_steps', 1) if eval_params is not None else 1
    stop_time = eval_params.get('stop_time', 0.0) if eval_params is not None else 0.0
    
    assert all([met in SUPPORTED_METRICS for met in eval_metrics]), f"Some evaluation metrics are not supported. Supported metrics: {SUPPORTED_METRICS}"
    
    logger.info(f"Starting evaluation with reconstruction steps: {recon_steps}.")
    logger.info(f"Reconstruction stop time: {stop_time}")

    # -- evaluation loop
    N = len(dataloader.dataset)
    masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
    clslabels_all = np.zeros((N,), dtype=np.uint8)
    anom_types_all = np.zeros((N,), dtype=np.uint8)
    anom_labels_all = np.zeros((N,), dtype=np.uint8)
    mse_all = np.zeros((N, *img_sz), dtype=np.float32)
    traj_all = np.zeros((N, recon_steps+1, 272, 16, 16), dtype=np.float32)
    org_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
    recon_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
    all_avg_w = []
        
    print("▶️Closed-form velocity fields constructed for each class.")
    
    logger.info("Extracting features and computing anomaly scores...")
    noise = torch.randn((1, *noise_shape), device=device)
    perturb_t = torch.tensor([stop_time], device=device)
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
        zt = vf.perturb(z1, noise.repeat(len(z1), 1, 1, 1), perturb_t.repeat(len(z1))) # (B, c, h, w)
        
        # -- reconstruction through the velocity field
        if use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pass
                z1_local, traj = vf.sample(zt, steps=recon_steps, start_t=stop_time, return_intermediate=True)
        else:
            z1_local, traj = vf.sample(zt, steps=recon_steps, start_t=stop_time, return_intermediate=True, )
        
        traj_local = torch.stack(traj, dim=1)  # (B, recon_steps, c, h, w)
        # traj_init = traj[0].permute(0, 2, 3, 1).contiguous().view(-1, z1.shape[1])  # (B*h*w, c)
        # gmm_centers = torch.from_numpy(gmm.means_).float().to(device)  # (M, c)
        # gmm_init_dist = torch.cdist(traj_init, gmm_centers)  # (B*h*w, M)
        # gmm_init_idxs = torch.argmin(gmm_init_dist, dim=1)  # (B*h*w)
        
        # is_center_fixed = torch.ones_like(gmm_init_idxs)  # (B*h*w)
        # is_weight_suffice = torch.ones_like(gmm_init_idxs)  # (B*h*w)
        # w_avg_list = []
        # max_w_list = []
        # w_threshold = 0.99
        # timesteps = torch.linspace(stop_time, 1, steps=recon_steps+1)
        
        # # logger.info(f"Timesteps: {timesteps}")
        # for t_i in range(0, len(traj)):
        #     current_t = timesteps[t_i]
        #     if current_t == 1:
        #         break
        #     traj_t = traj[t_i].permute(0, 2, 3, 1).contiguous().view(-1, z1.shape[1])  # (B*h*w, c)
            
        #     softmax_w = compute_softmax_weights(gmm, traj_t, current_t, device)  # (B*h*w, M)
        #     max_w = softmax_w.max(dim=1)[0]  # (B*h*w)
        #     gathered_w = softmax_w.gather(1, gmm_init_idxs.unsqueeze(1)).squeeze(1)  # (B*h*w)
        #     is_weight_suffice &= (gathered_w >= w_threshold)  # (B*h*w)
        #     w_avg_list.append(gathered_w.mean().item())
        #     max_w_list.append(max_w.mean().item())
        #     # logger.info(f"Softmax weights at timestep {current_t}: {softmax_w}")
        #     gmm_dist_t = torch.cdist(traj_t, gmm_centers)  # (B*h*w, M)
        #     gmm_center_idx_t = torch.argmin(gmm_dist_t, dim=1)  # (B*h*w)
        #     is_center_fixed &= (gmm_center_idx_t == gmm_init_idxs)  # (B*h*w)
        # fixed_ratio = is_center_fixed.float().mean()  # scalar
        # weight_suffice_ratio = is_weight_suffice.float().mean()  # scalar
        
        # logger.info(f"Fixed ratio💾: {fixed_ratio}")
        # logger.info(f"Weight suffice ratio💾: {weight_suffice_ratio}")
        # logger.info(f"Average softmax weight💾: {w_avg_list}")
        # used_idxs = torch.unique(gmm_init_idxs)
        # logger.info(f"Used GMM center indices💾: {used_idxs}")
        
        # import pdb; pdb.set_trace()
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
        traj_all[idx_np] = traj_local.cpu().numpy()
        
    if distributed:
        torch.distributed.barrier()
        
    traj_init = torch.from_numpy(traj_all[:, 0])  # (N, 16, 16, 16)
    traj_init = traj_init.permute(0, 2, 3, 1).contiguous().view(-1, 272).to(device)
    traj_rest = torch.from_numpy(traj_all[:, 1:]).to(device)
    
    gmm_centers = torch.from_numpy(gmm.means_).float().to(device)  # (M, c)
    gmm_init_dist = torch.cdist(traj_init, gmm_centers)  # (B*h*w, M)
    gmm_init_idxs = torch.argmin(gmm_init_dist, dim=1)  # (B*h*w)
    
    gt_masks = torch.from_numpy(masks_all)  # (N, H, W)
    gt_masks = F.interpolate(
        gt_masks.unsqueeze(1).float(), size=(16, 16), mode='nearest'
    ).squeeze(1).long()  # (N, h, w)
    gt_masks = gt_masks.flatten()  # (N*h*w,)
    
    timesteps = torch.linspace(stop_time, 1, steps=recon_steps+1)
    w_means_normal = []
    w_means_anomaly = []
    
    is_suffice_condition_normal = torch.ones_like(gt_masks).to(device)
    is_suffice_condition_normal[gt_masks > 0.5] = 0
    is_suffice_condition_anomaly = torch.ones_like(gt_masks).to(device)
    is_suffice_condition_anomaly[gt_masks <= 0.5] = 0
    lower_ths = 0.99
    for t_i in range(0, recon_steps):
        t = timesteps[t_i]
        traj_t = traj_rest[:, t_i].permute(0, 2, 3, 1).contiguous().view(-1, 272).to(device)  # (N*16*16, 272)
        softmax_w = compute_softmax_weights(gmm, traj_t, t, device)
        gathered_w = softmax_w.gather(1, gmm_init_idxs.unsqueeze(1)).squeeze(1)  # (B*h*w)
        
        gathered_w_normal = gathered_w[gt_masks <= 0.5]
        gathered_w_anomaly = gathered_w[gt_masks > 0.5]
        
        w_means_normal.append(gathered_w_normal.mean().item())
        w_means_anomaly.append(gathered_w_anomaly.mean().item())
        is_suffice_condition_normal[gt_masks <= 0.5] &= (gathered_w_normal >= lower_ths)
        is_suffice_condition_anomaly[gt_masks > 0.5] &= (gathered_w_anomaly >= lower_ths)
    num_normal_pixels = (gt_masks <= 0.5).sum().item()
    num_anomaly_pixels = (gt_masks > 0.5).sum().item()
    logger.info("Mean softmax weights over reconstruction steps (normal): %s", w_means_normal)
    logger.info("Mean softmax weights over reconstruction steps (anomaly): %s", w_means_anomaly)
    suffice_ratio_normal = is_suffice_condition_normal.float().sum().item() / num_normal_pixels
    suffice_ratio_anomaly = is_suffice_condition_anomaly.float().sum().item() / num_anomaly_pixels
    logger.info("Ratio of sufficient softmax weights over reconstruction steps (normal): %s", suffice_ratio_normal)
    logger.info("Ratio of sufficient softmax weights over reconstruction steps (anomaly): %s", suffice_ratio_anomaly)
    
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
    return eval_results

def compute_softmax_weights(gmm, x, t, device):
    assert isinstance(gmm, TiedSphericalGaussianMixture), "GMM must have same and sperical components"
    gmm_w = torch.from_numpy(gmm.weights_).float().to(device)  # (M,)
    labmda = torch.from_numpy(gmm.covariances_)[0].float().to(device)  # (1,)
    ct = t ** 2 * labmda + (1-t) ** 2
    centers = torch.from_numpy(gmm.means_).float().to(device)  # (M, D)
    # import pdb; pdb.set_trace()
    dist = torch.sum((x.unsqueeze(1) - t * centers) ** 2, dim=-1) / (-2 * ct)  # (N, M)
    weights = torch.log(gmm_w.unsqueeze(0)) + dist  # (N, M)
    softmax_weights = torch.softmax(weights, dim=-1)  # (N, M)
    return softmax_weights