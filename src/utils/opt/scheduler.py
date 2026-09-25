import math
from bisect import bisect_right
import torch

def get_warmup_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    start_lr: float,
    ref_lr: float,
    T_max: int,
    final_lr: float = 0.,
    fix_lr_thres: int = -1,
    fix_strategy: str = 'const'
) -> 'WarmupCosineLRSchedule':
    """
    Returns a WarmupCosineLRSchedule for learning rate scheduling.
    """
    return WarmupCosineLRSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=ref_lr,
        T_max=T_max,
        final_lr=final_lr,
        fix_lr_thres=fix_lr_thres,
        fix_strategy=fix_strategy
    )

def get_cosine_wd_scheduler(
    optimizer: torch.optim.Optimizer,
    ref_wd: float,
    T_max: int,
    final_wd: float = 0.,
    fix_wd_thres: int = -1,
    fix_strategy: str = 'const'
) -> 'CosineWDSchedule':
    """
    Returns a CosineWDSchedule for weight decay scheduling.
    """
    return CosineWDSchedule(
        optimizer=optimizer,
        ref_wd=ref_wd,
        T_max=T_max,
        final_wd=final_wd,
        fix_wd_thres=fix_wd_thres,
        fix_strategy=fix_strategy
    )

class WarmupCosineLRSchedule(object):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        final_lr=0.,
        fix_lr_thres=-1,
        fix_strategy='const'
    ):
        """
        Warmup and cosine learning rate schedule.
        :param optimizer: optimizer to adjust the learning rate for each step
        :param warmup_steps: number of warmup steps
        :param start_lr: initial learning rate at step 0
        :param ref_lr: reference learning rate after warmup
        :param T_max: total number of steps for cosine schedule
        :param final_lr: final learning rate at the end of cosine schedule
        :param fix_lr_thres: step threshold to fix learning rate
        :param fix_strategy: strategy to fix learning rate ('const' or 'linear')
        Note: this scheduler must be called at each training **step** to update the learning rate.
        """
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.
        self.fix_lr_thres = fix_lr_thres
        self.fix_strategy = fix_strategy

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        
        elif self._step < self.fix_lr_thres or self.fix_lr_thres < 0:                    
        
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        else:
            progress = float(self.fix_lr_thres - self.warmup_steps) / float(max(1, self.T_max))
            last_cosine_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
            min_cosine_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi)))
            
            if self.fix_strategy == 'const':
                new_lr = last_cosine_lr
            elif self.fix_strategy == 'linear':
                progress = float(self._step - self.fix_lr_thres) / float(max(1, self.T_max - (self.fix_lr_thres - self.warmup_steps)))
                new_lr = last_cosine_lr - progress*(last_cosine_lr - min_cosine_lr)            
        
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        return new_lr
    
    def get_current_lr(self):
        for group in self.optimizer.param_groups:
            return group['lr']
        
    def state_dict(self):
        """Return a state dict for saving the scheduler's state."""
        return {
            'start_lr': self.start_lr,
            'ref_lr': self.ref_lr,
            'final_lr': self.final_lr,
            'warmup_steps': self.warmup_steps,
            'T_max': self.T_max,
            '_step': self._step,
            'fix_lr_thres': self.fix_lr_thres,
            'fix_strategy': self.fix_strategy,
        }

    def load_state_dict(self, state_dict):
        """Restore the scheduler's state from a state dict."""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k == 'current_step':
                self._step = v
    
class CosineWDSchedule(object):
    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.,
        fix_wd_thres=-1,
        fix_strategy='const'
    ):
        """
        Cosine weight decay schedule.
        :param optimizer: optimizer to adjust the weight decay for each step
        :param ref_wd: reference weight decay at the beginning
        :param T_max: total number of steps for cosine schedule
        :param final_wd: final weight decay at the end of cosine schedule
        :param fix_wd_thres: step threshold to fix weight decay
        :param fix_strategy: strategy to fix weight decay ('const' or 'linear')
        Note: this scheduler must be called at each training **step** to update the weight decay
        """
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.
        self.fix_wd_thres = fix_wd_thres
        self.fix_strategy = fix_strategy

    def step(self):
        self._step += 1
        
        if self._step < self.fix_wd_thres or self.fix_wd_thres < 0:                    
            progress = self._step / self.T_max
            new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))
        else:
            progress = float(self.fix_wd_thres) / self.T_max
            last_cosine_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))
            if self.fix_strategy == 'const':
                new_wd = last_cosine_wd
            elif self.fix_strategy == 'linear':
                progress = float(self._step - self.fix_wd_thres) / (self.T_max - self.fix_wd_thres)
                max_cosine_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * 1))
                new_wd = last_cosine_wd + progress*(max_cosine_wd - last_cosine_wd)            
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
    
    def get_current_wd(self):
        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                return group['weight_decay']
        return None
    
    def state_dict(self):
        """Return a state dict for saving the scheduler's state."""
        return {
            'ref_wd': self.ref_wd,
            'final_wd': self.final_wd,
            'T_max': self.T_max,
            '_step': self._step,
            'fix_wd_thres': self.fix_wd_thres,
            'fix_strategy': self.fix_strategy,
        }
    def load_state_dict(self, state_dict):
        """Restore the scheduler's state from a state dict."""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
