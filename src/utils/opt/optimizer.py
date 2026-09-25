import torch
import torch.optim as optim
from torch import nn

from src.utils.opt.scheduler import WarmupCosineLRSchedule,CosineWDSchedule
from src.utils.log import get_logger
logger = get_logger(__name__)

def get_wd_filter(
    bias_decay: bool = False,
    norm_decay: bool = False,
):
    """
    Create a weight decay filter function based on the provided conditions.
    params:
        bias_decay (bool): Whether to apply weight decay to bias parameters.
        norm_decay (bool): Whether to apply weight decay to normalization layers.
    returns:
        function: A function that determines if weight decay should be applied to a parameter.
    """
    def is_bias(n: str) -> bool:
        return n.endswith("bias")

    def is_norm_like(n: str, p) -> bool:
        return len(getattr(p, "shape", ())) == 1

    def should_decay(n, p) -> bool:
        if is_bias(n):
            return bias_decay
        if is_norm_like(n, p):
            return norm_decay
        return True
    return should_decay

def get_optimizer_by_name(
    optimizer_name: str,
    param_groups = None,
    **kwargs
):
    """
    Create an optimizer based on the provided name.
    Args:
        optimizer_name (str): Name of the optimizer ('sgd' or 'adamw').
    Returns:
        torch.optim.Optimizer: The corresponding optimizer class.
    """
    assert param_groups is not None, "param_groups must be provided"
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(param_groups, **kwargs)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(param_groups, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(param_groups, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def build_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    bias_decay: bool = False,
    norm_decay: bool = False,   
):
    """
    Create optimizer for training stage.

    Args:
        model (nn.Module): The model to optimize.
        optimizer_name (str): Name of optimizer ('adamw', 'sgd', etc.)
        bias_decay (bool): Apply weight decay to bias params if True.
        norm_decay (bool): Apply weight decay to norm layers if True.
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    wd_filter = get_wd_filter(bias_decay, norm_decay)

    # Materialize once to avoid exhausting generators multiple times
    all_named = list(model.named_parameters())
    decay_params    = [p for n, p in all_named if wd_filter(n, p)]
    no_decay_params = [p for n, p in all_named if not wd_filter(n, p)]

    param_groups = [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0, "WD_exclude": True},
    ]
    optimizer = get_optimizer_by_name(optimizer_name, param_groups)
    return optimizer

def save_checkpoint(
    save_path: str,
    epoch: int, 
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    lr_scheduler: WarmupCosineLRSchedule,
    wd_scheduler: CosineWDSchedule,
):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'wd_scheduler': wd_scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)
    logger.info(f'Saved checkpoint at epoch {epoch} to {save_path}')

def load_checkpoint(
    resume_path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    lr_scheduler: WarmupCosineLRSchedule,
    wd_scheduler: CosineWDSchedule,
):
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=False)
    epoch = checkpoint['epoch']

    # -- loading model
    pretrained_dict = checkpoint['model']
    msg = model.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained model from epoch {epoch} with msg: {msg}')
    
    # -- loading optimizers
    opt.load_state_dict(checkpoint['opt'])
    scaler.load_state_dict(checkpoint['scaler'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    wd_scheduler.load_state_dict(checkpoint['wd_scheduler'])

    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {resume_path}')
    del checkpoint
    return model, opt, scaler, epoch

def load_model_only(
    resume_path: str,
    model: nn.Module,
):
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=True)

    # -- loading model
    pretrained_dict = checkpoint['model']
    new_dict = {}
    for k, v in pretrained_dict.items():
        if 'ema_model.' in k:
            pass
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    
    msg = model.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained model with msg: {msg}')
    
    del checkpoint
    return model