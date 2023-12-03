import torch

"""
    @description: 实现tensor的上采样操作
    @author: ZhouRenjie
    @Date:2023/12/03
"""


def upsample(tensor, scale):
    tensor = tensor.unsqueeze(0)
    assert len(tensor.shape) == 4 and tensor.shape[0] == 1, "tensor结构不规范，无法进行unsample"
    tensor = torch.nn.functional.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0)
    return tensor
