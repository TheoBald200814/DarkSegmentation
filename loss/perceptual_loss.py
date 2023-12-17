import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image



# 加载VGG模型，并截取到目标层的特征
class VGGFeatureExtractor(nn.Module):
    def __init__(self, target_layer):
        super(VGGFeatureExtractor, self).__init__()
        vgg_model = models.vgg16(pretrained=True)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(vgg_model.features.children()))
        self.target_layer = target_layer

    def forward(self, x):
        for layer_name, layer in self.features._modules.items():
            x = layer(x)
            if layer_name == self.target_layer:
                return x
        return x

# 定义 perceptual loss 函数
class PerceptualLoss(nn.Module):
    def __init__(self, target_layer):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(target_layer)
        self.criterion = nn.MSELoss()

    def forward(self, P, T):
        # 提取预测结果 P 和目标 T 的特征
        features_P = self.feature_extractor(P)
        features_T = self.feature_extractor(T)

        # 计算 perceptual loss
        loss = self.criterion(features_P, features_T)

        return loss


