import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.flow_matching import VelocityField, ShortcutVelocityField


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
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger
from src.utils.opt.optimizer import load_model_only
from src.vfad.eval_inversion import evaluate_inv
from src.vfad.eval_density import evaluate_density
from src.vfad.eval_recon import evaluate_recon

import logging
logger = logging.getLogger(__name__)

def main(params, args):
    """XXX for VFAD evaluation.
    We use following terms in the code:
    - 
    - 
    - 
    """
    # init distributed mode
    init_distributed_mode(args)

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device('cuda:%s'%args.gpu)
    
    # -- setup logging
    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

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
    pred_type, loss_type = params['flow_matching'].get('pred_type', 'velocity'), params['flow_matching'].get('loss_type', 'velocity')
    train_steps = params['flow_matching'].get('train_steps', -1)
    t_scheduler_train = params['flow_matching']['scheduler'].get('t_scheduler_train', 'linear')
    t_scheduler_infer = params['flow_matching']['scheduler'].get('t_scheduler_infer', 'linear')
    t_mu = params['flow_matching']['scheduler'].get('t_mu', 0.0)
    t_sigma = params['flow_matching']['scheduler'].get('t_sigma', 1.0)
    div_eps = params['flow_matching'].get('div_eps', 0.05)
    logger.info(f"Flow Matching Settings: t_scheduler_train: {t_scheduler_train}, t_scheduler_infer: {t_scheduler_infer}, t_mu: {t_mu}, t_sigma: {t_sigma}, div_eps: {div_eps}")
    logger.info(f"Flow Matching Prediction Type: {pred_type}, Loss Type: {loss_type}")
    logger.info(f"Using partial time sampling with {train_steps} training steps." if train_steps > 0 else "Using full time sampling.")
    model = init_model(input_sz=feat_sz, num_classes=num_classes, **params['model']).to(device)
    vf = VelocityField(
        model=model,
        input_sz=feat_sz,
        scheduler_name=params['flow_matching']['scheduler']['name'],
        solver_name=params['flow_matching']['solver']['name'],
        loss_type=loss_type,
        pred_type=pred_type,
        train_steps=train_steps,
        t_scheduler_train=t_scheduler_train,
        t_scheduler_infer=t_scheduler_infer,
        t_mu=t_mu,
        t_sigma=t_sigma,
        div_eps=div_eps,
        scheduler_params=params['flow_matching']['scheduler'].get('params', None),
        solver_params=params['flow_matching']['solver'].get('params', None),
    )
    logger.info(f"Velocity Field Model {params['model']['model_name']} has been initialized.")

    if args.distributed:
        vf = torch.nn.parallel.DistributedDataParallel(vf, static_graph=True)
    
    # -- resume training if needed
    resume_path = params['resume']['resume_path']
    assert resume_path is not None, "Please specify the checkpoint path for evaluation."
    assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!, Please check the path."
    
    vf = load_model_only(
        resume_path, vf
    )
    logger.info(f"Resumed from checkpoint: {resume_path}")
    
    # -- evaluation setup
    eval_params = params['logging']['eval']
    eval_strategy = eval_params.get('eval_strategy', 'inversion') if eval_params is not None else 'inversion'
    if eval_strategy == 'inversion':
        logger.info("Using inversion-based evaluation strategy.")
        eval_func = evaluate_inv
    elif eval_strategy == 'density':
        logger.info("Using density-estimation-based evaluation strategy.")
        eval_func = evaluate_density
    elif eval_strategy == 'recon':
        logger.info("Using reconstruction-based evaluation strategy.")
        eval_func = evaluate_recon
    else:
        raise ValueError(f"Unknown evaluation strategy: {eval_strategy}")
    
    vf.eval()

    # -- evaluate
    logger.info(f"Starting evaluation...")
    ablation_steps = eval_params.get('ablation_steps', [-1])
    do_ablation = ablation_steps != [-1]
    
    if do_ablation:
        logger.info(f"Performing ablation study with steps: {ablation_steps}")
        for step in ablation_steps:
            logger.info(f"Evaluating with {step} steps...")
            eval_params["recon_steps"] = step
            eval_results = eval_func(
                vf=vf,
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
                train_loader=train_loader,
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
        eval_results = eval_func(
            vf=vf,
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
            train_loader=train_loader,
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