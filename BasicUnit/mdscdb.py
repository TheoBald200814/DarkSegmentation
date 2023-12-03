from torch import nn
import torch
from BasicUnit import dscdb


class MDSCDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDSCDB, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 5),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.dscdb = dscdb.DSCDB(in_channels, out_channels)

    def forward(self, x):
        m = self.dscdb(x)
        n = self.sequential(x)

        # concate
        temp_n = n.unsqueeze(0)
        m_size = [m.shape[1], m.shape[2]]
        new_n = torch.nn.functional.interpolate(temp_n, size=m_size, mode='bilinear', align_corners=False)
        new_n = new_n.squeeze(0)
        result =torch.cat((new_n, m), dim=0)

        return result


# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
#
# model = MDSCDB().to(device)
#
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
#
# model = MDSCDB().to(device)
#
# print(model)
#
# low_image_dir = './LOLdataset/our485/low'
# high_image_dir = './LOLdataset/our485/high'
# my_datasets = Data.MyData(low_image_dir, high_image_dir)
#
# epoch = 1
#
# # 训练轮次
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # 加载优化器
# loss = nn.MSELoss()
#
# for i in range(epoch):
#     for j in range(my_datasets.size()):
#         img = my_datasets.get_by_index(j)
#         y_hat = model(img[0])
#         print(f"y_hat.shape[{y_hat.shape}]")
#         print(f"label.shape[{img[1].shape}]")
#         result_loss = loss(y_hat, img[1])
#         print(result_loss)
#         result_loss.backward()
#         optimizer.step()
#
# torch.save(model, './model_1.pth')
