import Frameworks.level_three_stream as model3
import Frameworks.level_two_stream as model2
import torch
import Tools.Data as dataloader
import matplotlib.pyplot as plt

model = model2.LevelTwoStream()
model.load_state_dict(torch.load('../checkpoints/model_3.pth', map_location='cpu'))
model.eval()

datasets = dataloader.MyData('../LOLdataset/our485/low', './LOLdataset/our485/high')
valid_img = datasets.get_by_index(5)


def f(y_hat):
    numpy_image = y_hat.cpu().numpy()
    # 将数值范围规范到 0-1 之间
    numpy_image = numpy_image.transpose(1, 2, 0)  # 转换通道顺序
    numpy_image = (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min())
    # 显示图像
    plt.imshow(numpy_image)
    plt.axis('off')  # 可选，关闭坐标轴
    plt.show()


with torch.no_grad():
    y_hat = model(valid_img[0])
    y_hat = y_hat.unsqueeze(0)
    y_hat = torch.nn.functional.interpolate(y_hat, size=[400, 600], mode='bilinear', align_corners=False)
    y_hat = y_hat.squeeze(0)
    f(y_hat)
    f(valid_img[1])
    f(valid_img[0])










