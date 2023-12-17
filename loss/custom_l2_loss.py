

import torch

def ensure_same_shape(P, T):
    """
    Ensure that tensors P and T have the same shape. If not, reshape P to match the shape of T.

    Args:
    - P (torch.Tensor): Predicted tensor.
    - T (torch.Tensor): Target tensor.

    Returns:
    - torch.Tensor: Reshaped predicted tensor with the same shape as T.
    """
    if P.shape != T.shape:
        P = torch.nn.functional.interpolate(P.unsqueeze(0), size=(T.shape[0], T.shape[1]), mode='nearest').squeeze(0)
    return P

def custom_l2_loss(P, T):
    """
    Custom L2 loss function for tensors P and T with shape HWC.

    Args:
    - P (torch.Tensor): Predicted tensor of shape HWC.
    - T (torch.Tensor): Target tensor of shape HWC.

    Returns:
    - torch.Tensor: L2 loss.
    """
    # Ensure P and T have the same shape
    P = ensure_same_shape(P, T)

    # Calculate element-wise squared difference
    squared_diff = (P - T)**2

    # Sum across all dimensions
    sum_squared_diff = torch.sum(squared_diff)

    # Normalize by the product of H, W, and C
    normalization_factor = P.shape[0] * P.shape[1] * P.shape[2]

    # Calculate the final L2 loss
    l2_loss = sum_squared_diff / normalization_factor

    return l2_loss


