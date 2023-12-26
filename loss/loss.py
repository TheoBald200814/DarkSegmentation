from loss import custom_l2_loss
from loss import perceptual_loss
from loss import ssim_loss as sl
from torch import nn


def loss_function(prediction, label, vgg16_layer_number, lambda_1, lambda_2, device):

    # 实例化 perceptual loss 模型
    perceptual_loss_model = perceptual_loss.PerceptualLoss(vgg16_layer_number).to(device)

    # 计算 perceptual loss
    per_loss = perceptual_loss_model(prediction, label)

    ssim_loss = sl.SSIMLoss(prediction, label)

    # 有问题
    l2_loss = custom_l2_loss.custom_l2_loss(prediction, label)

    loss_total = per_loss + lambda_1 * ssim_loss + lambda_2 * l2_loss
    return loss_total


class MyLoss(nn.Module):
    def __init__(self, vgg16_layer_number, lambda_1, lambda_2, device):
        super(MyLoss, self).__init__()
        self.vgg16_layer_number = vgg16_layer_number
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device

    def forward(self, prediction, label):
        return loss_function(prediction, label, vgg16_layer_number=self.vgg16_layer_number, lambda_1=self.lambda_1, lambda_2=self.lambda_2, device=self.device)
