import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import enum
from torch.nn.utils import spectral_norm # Import spectral_norm

from typing import Tuple

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange
import torch.nn.functional as F

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embed_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embed_size = frequency_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create a sinusodial timestep embedding.
        Args:
            t (Tensor): tensor of shape (B, )
            dim (int): embedding dimension
            max_period (int): maximum period
        Returns:
            Tensor: tensor of shape (B, D)
        """
        half = dim // 2
        freqs = torch.exp(
            -1 * math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (B, 2/D)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, D)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # (B, D)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embed_size)  # (B, D)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        self.channels = channels
        
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )
    
    def forward(self, x, y):
        # conditioning
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)  # (B, C)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)  # (B, N, C)
        h = self.mlp(h)  # (B, N, C)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        # Apply final linear layer
        x = self.linear(x)
        return x

class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        use_cfg_embedding = dropout_prob > 0.0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)  # we add cfg embedding which represent "no class information"
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops label information to enable CFG. 
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob  # (B, )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_drop_out = train and self.dropout_prob > 0.0
        if use_drop_out or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,   
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_blocks: int,
        patch_size: int = 1,
        grad_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_blocks = num_blocks
        self.grad_checkpoint = grad_checkpoint
        
        self.input_proj = nn.Linear(in_channels, model_channels, bias=True)
        self.time_embed = TimestepEmbedder(model_channels)
        self.y_embedder = LabelEmbedder(15, model_channels, 0.0)
        
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)
        
        self.final_layer = FinalLayer(model_channels, out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)   

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        # Zero out adaLN modulation weights
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def forward(self, x, t, y=None, return_tokens=False, **kwargs):
        """Apply model to an input batch
        Args:
            x (Tensor): input tensor (B, C)
            t (Tensor): timestep tensor (B, )
            y (Tensor): condition tensor (B, Z)
        Returns:
            Tensor: output tensor (B, C)
        """
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C)
        x = self.input_proj(x) # (B, D)
        t = self.time_embed(t)
        if y is not None:
            y = self.y_embedder(y, self.training)  # (B, D)
            cond = t + y
        else:
            cond = t
        cond = cond.unsqueeze(1)  # (B, 1, D)
        cond = cond.expand(-1, x.shape[1], -1)  # (B, N, D)
        if self.grad_checkpoint and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, cond)
        else:
            for block in self.res_blocks:
                x = block(x, cond)
        
        x = self.final_layer(x, cond)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (B, C, H, W)
        return x
        
    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[:len(x)//2]  # (B/2, C)
        combined = torch.cat([half, half], dim=0)  # (B, C)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]  
        cond_eps, uncond_eps = torch.split(eps, len(eps)//2, dim=0)  # (B/2, C)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)