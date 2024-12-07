import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.data_preprocessing import read_images, read_labels
from PIL import Image
import numpy as np

class DigitDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        super().__init__()
        self.transform = transform
        
        # 读取图像和标签
        self.images, self.rows, self.cols = read_images(image_path)
        self.labels = read_labels(label_path)
        
        # 转换为 NumPy 数组 -> Torch 张量
        self.images = torch.tensor(self.images, dtype=torch.float32).reshape(-1, 1, 28, 28)  # 转换为 (N, C, H, W)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]  # (1, 28, 28)
        label = self.labels[index]

        # 如果有 transform，应用在 image 上
        if self.transform:
            image = self.transform(image)

        return image, label

# 定义数据增强/变换
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))  # 对单通道图像进行归一化
])

# 创建数据集和数据加载器
train_dataset = DigitDataset(
    image_path='./Dataset/MNIST/train-images-idx3-ubyte', 
    label_path='./Dataset/MNIST/train-labels-idx1-ubyte', 
    transform=transform
)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

eval_dataset = DigitDataset(
    image_path='./Dataset/MNIST/t10k-images-idx3-ubyte', 
    label_path='./Dataset/MNIST/t10k-labels-idx1-ubyte', 
    transform=transform
)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=True)



if __name__ == "__main__":
    for inputs, labels in train_dataloader:
        print(f"Input batch shape: {inputs.shape}")  # 修正为正确的属性调用
        print(f"Label batch shape: {labels.shape}")
