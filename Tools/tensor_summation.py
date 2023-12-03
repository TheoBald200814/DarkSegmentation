import torch
"""
    @description: 用于对两个tensor做element-wise sum操作。两个tensor形如[c, m, n]，其中m、n任意，而两个tensor的c需要满足c1 = c2 * 2。
    @author: ZhouRenjie
    @Date: 2023/12/03
"""


def tensor_summation(tensor1, tensor2):
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

    # 统一通道数
    if tensor1_size[0] < tensor2_size[0]:
        # 若tensor1的通道数较低
        assert tensor1_size[0] * 2 == tensor2_size[
            0], "输入的Tensors逻辑错误[tensor1.channel < tensor2.channel and tensor1.channel * 2 != tensor2.channel]"
        tensor1 = torch.cat((tensor1, tensor1), dim=0)

    if tensor1_size[0] > tensor2_size[0]:
        # 若tensor2的通道数较低
        assert tensor2_size[0] * 2 == tensor1_size[
            0], "输入的Tensors逻辑错误[tensor1.channel > tensor2.channel and tensor1.channel != tensor2.channel * 2]"
        tensor2 = torch.cat((tensor2, tensor2), dim=0)

    assert tensor1.shape == tensor2.shape, f"Tensors格式仍然不对，tensor1.shape = {tensor1.shape}, tensor2.shape = {tensor2.shape}"
    return tensor1 + tensor2
