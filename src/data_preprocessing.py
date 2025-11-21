import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

""" Private Class, just for data preprocessing"""
class CustomMNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 数据集所在的文件夹，包含子文件夹 'TrainingSet' 和 'TestSet'
            transform (callable, optional): 用于数据预处理的转换
        """
        self.data_dir = data_dir
        self.transform = transform

        # 获取所有图片路径和标签
        self.image_paths = []
        self.labels = []

        # 遍历所有文件，文件名格式为 {数字}_{编号}.bmp
        for img_name in os.listdir(data_dir):
            if img_name.endswith(".bmp"):
                # 解析文件名中的数字部分作为标签
                label = int(img_name.split('_')[0])  # 获取文件名的第一个部分（数字）

                # 添加图像路径和标签
                self.image_paths.append(os.path.join(data_dir, img_name))
                self.labels.append(label)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('L')  # 转为灰度图像

        if self.transform:
            img = self.transform(img)

        return img, label


def load_data(batch_size=64, data_dir='./data'):
    # 定义图像预处理操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 加载训练集和测试集
    train_dataset = CustomMNISTDataset(data_dir=os.path.join(data_dir, 'TrainingSet'), transform=transform)
    test_dataset = CustomMNISTDataset(data_dir=os.path.join(data_dir, 'TestSet'), transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
