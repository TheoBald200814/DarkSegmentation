from torch import nn, optim
import Frameworks.level_one_stream as model1
import Tools.Data as dataloader
import torch


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# 设备选择，默认CPU

model = model1.LevelOneStream(3, 3, 6, 6).to(device)

print(model)
loss = nn.MSELoss()
datasets = dataloader.MyData('/Users/theobald/Documents/code_lib/python_lib/DarkSegmentation/LOLdataset/our485/low',
                             '/Users/theobald/Documents/code_lib/python_lib/DarkSegmentation/LOLdataset/our485/high')


epoch = 5
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for i in range(epoch):
    for j in range(datasets.size()):
        data = datasets.get_by_index(j)
        y_hat = model(data[0])
        y = data[1]

        y_hat = y_hat.unsqueeze(0)
        y_hat = torch.nn.functional.interpolate(y_hat, size=[400, 600], mode='bilinear', align_corners=False)
        y_hat = y_hat.squeeze(0)

        result_loss = loss(y_hat, y)
        print(result_loss)
        result_loss.backward()
        optimizer.step()

torch.save(model.state_dict(), './checkpoints/model_4.pth')










# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(10, 5),
#             nn.Linear(5, 1)
#
#         )
#
#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits
# # 构建特定的神经网络模型
#
# model = NeuralNetwork().to(device)
#
# print(model)
#
# # loss = nn.CrossEntropyLoss()
#
# class My_loss(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, y_hat, targets):
#         n = y_hat.shape[0]
#         loss = 0.0
#         for i in range(n):
#             temp = (y_hat[i] - targets[i]) ** 2
#             loss += temp
#         return loss
# # 自定义损失函数
#
# loss = My_loss()
#
# X = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                   [4, 5, 2, 1, 55, 0.1, 2, 4, 2, 1],
#                   [11, 22, 33, 44, 55, 66, 77, 88, 99, 11]])
# Y = torch.Tensor([[1],
#                  [0],
#                  [1]]
#                  )
# # 自定义数据集和标签集
#
# epoch = 10
# # 训练轮次
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # 加载优化器
# for i in range(epoch):
#     for j in range(X.shape[0]):
#         y_hat = model(X[j])
#         result_loss = loss(y_hat, Y[j])
#         print(result_loss)
#         result_loss.backward()
#         optimizer.step()
# # 训练




