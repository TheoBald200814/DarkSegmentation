import torch

"""
    @description: 用于对两个tensor做concate操作。两个tensor形如[c, m, n]，concate操作组对c维进行组合。
    @author: ZhouRenjie
    @Date: 2023/12/04
"""


def tensor_concate(tensor1, tensor2):
    tensor1_size = tensor1.shape
    tensor2_size = tensor2.shape
    max_dim1 = max(tensor1_size[1], tensor2_size[1])
    max_dim2 = max(tensor1_size[2], tensor2_size[2])

    # 统一分辨率
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)
    tensor1 = torch.nn.functional.interpolate(tensor1, size=[max_dim1, max_dim2], mode='bilinear', align_corners=False)
    tensor2 = torch.nn.functional.interpolate(tensor2, size=[max_dim1, max_dim2], mode='bilinear', align_corners=False)
    tensor1 = tensor1.squeeze(0)
    tensor2 = tensor2.squeeze(0)

    result = torch.cat((tensor1, tensor2), dim=0)
    return result