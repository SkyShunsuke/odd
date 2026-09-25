
def normalize_to_unit_interval(x):
    """
    Normalize tensor values from [-1, 1] to [0, 1].
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized tensor.
    NOTE: Assumes input tensor values are in the range [-1, 1].
    """
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def tensor_to_numpy_image(x):
    """Convert tensor on GPU to numpy image, shape (H, W, C) or (N, H, W, C).
    Args:
        x (torch.Tensor): Input tensor. shape can be (C, H, W) or (N, C, H, W).
    Returns:
        np.ndarray: Converted numpy array. shape will be (H, W, C) or (N, H, W, C).
    """
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()