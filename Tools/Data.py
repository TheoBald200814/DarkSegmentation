import os
from torchvision import transforms
from PIL import Image
"""
    @description: 图像数据加载类，专门用于加载图像增强模型需要使用的数据集和标签集。
    @author: ZhouRenjie
    @Date: 2023/12/03
"""


class MyData():
    def __init__(self, low_image_dir, high_image_dir):

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.low_image_dir = low_image_dir
        self.high_image_dir = high_image_dir
        self.low_image_datasets = []
        self.high_image_datasets = []

        # 检查文件夹是否存在
        if os.path.exists(low_image_dir):
            # 获取文件夹内的所有文件名
            low_image_file_names = os.listdir(low_image_dir)

            # 构建文件的完整路径
            self.low_image_file_paths = [os.path.join(low_image_dir, file_name) for file_name in low_image_file_names]

            for path in self.low_image_file_paths:
                temp_img = Image.open(path)
                self.low_image_datasets.append(transform(temp_img))
        else:
            print(f"文件夹 '{low_image_dir}' 不存在")


        if os.path.exists(high_image_dir):
            high_image_file_names = os.listdir(high_image_dir)
            self.high_image_file_paths = [os.path.join(high_image_dir, file_name) for file_name in high_image_file_names]

            for path in self.high_image_file_paths:
                temp_img = Image.open(path)
                self.high_image_datasets.append(transform(temp_img))
        else:
            print(f"文件夹 '{high_image_dir}' 不存在")

    def size(self):
        len_low = len(self.low_image_datasets)
        len_high = len(self.high_image_datasets)
        if len_low == len_high:
            return len_low
        else:
            return -1

    def get_by_index(self, i):
        n = self.size()
        if n != -1 and i >= 0 and i < n:
            return self.low_image_datasets[i], self.high_image_datasets[i]
        else:
            return [[], []]








