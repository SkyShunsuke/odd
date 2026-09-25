

import torch
import logging

from src.utils.tensor.images import *

logger = logging.getLogger(__name__)

def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def pidx_to_ppos(patch_idx, h, w):
    """
    Convert patch index to patch position
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.zeros((patch_idx.shape[0], patch_idx.shape[1], 2), dtype=torch.int)
    for i, idx in enumerate(patch_idx):
        patch_pos[i] = torch.stack([idx % w, idx // w], dim=1)
    return patch_pos

def ppos_to_pidx(patch_pos, h, w):
    """
    Convert patch position to patch index
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_idx = patch_pos[:, :, 1] * w + patch_pos[:, :, 0]
    return patch_idx

def ppos_to_pmask(patch_pos, h, w):
    """
    Convert patch position to binary mask of patches
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
    Return:
        mask: (N, H, W), binary mask
    """
    mask = torch.zeros((patch_pos.shape[0], 1, h, w), dtype=torch.float32)
    for i, pos in enumerate(patch_pos):
        for x, y in pos:
            mask[i, 0, y, x] = 1
    return mask

def pmask_to_ppos(mask):
    """
    Convert binary mask of patches to patch position
    Args:
        mask: (N, H, W), binary mask
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.stack(mask.nonzero(), dim=1)
    return patch_pos

def pidx_to_pmask(patch_idx, h, w):
    """
    Convert patch index to binary mask of patches
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
    Return:
        mask: (N, H, W), binary mask
    """
    patch_pos = pidx_to_ppos(patch_idx, h, w)
    mask = ppos_to_pmask(patch_pos, h, w)
    return mask

def pmask_to_pidx(mask, h, w):
    """
    Convert binary mask of patches to patch index
    Args:
        mask: (N, H, W), binary mask
        h: height of image
        w: width of image
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_pos = pmask_to_ppos(mask)
    patch_idx = ppos_to_pidx(patch_pos, h, w)
    return patch_idx

def ppos_to_imask(patch_pos, h, w, patch_size):
    """Convert patch position to binary mask of image
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        mask: (N, H, W), binary mask (float32)
    """
    H, W = h * patch_size, w * patch_size
    mask = torch.ones((patch_pos.shape[0], 1, H, W), dtype=torch.float32)
    for i, pos in enumerate(patch_pos):
        for x, y in pos:
            mask[i, 0, y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = 0
    return mask

def imask_to_ppos(mask, patch_size):
    """Convert binary mask of image to patch position
    Args:
        mask: (N, H, W), binary mask
        patch_size: size of patch
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.stack(mask.nonzero(), dim=1)
    patch_pos = patch_pos[:, :, [1, 2]]
    patch_pos = patch_pos - patch_pos % patch_size
    return patch_pos

def pidx_to_imask(patch_idx, h, w, patch_size):
    """Convert patch index to binary mask of image
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        mask: (N, H, W), binary mask
    """
    patch_pos = pidx_to_ppos(patch_idx, h, w)
    mask = ppos_to_imask(patch_pos, h, w, patch_size)
    return mask

def imask_to_pidx(mask, h, w, patch_size):
    """Convert binary mask of image to patch index
    Args:
        mask: (N, H, W), binary mask
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_pos = imask_to_ppos(mask, patch_size)
    patch_idx = ppos_to_pidx(patch_pos, h, w)
    return patch_idx
