import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1通道，输出32通道，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入32通道，输出64通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输入64通道，输出128通道
        
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化
        
        # 全连接层，展平后的尺寸将在 forward 方法中动态计算
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)  # 根据 MNIST (28x28) 的输入手动计算
        self.fc2 = nn.Linear(1024, 10)  # 输出10类（数字 0-9）

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 动态展平张量
        x = torch.flatten(x, 1)  # 从第1维开始展平，动态计算形状
        x = F.relu(self.fc1(x))  # 全连接 + 激活
        x = self.fc2(x)          # 输出层
        return x

# 测试模型结构
if __name__ == "__main__":
    model = DigitRecognitionCNN()
    print(model)

    # 模拟输入测试，MNIST 数据是 (N, 1, 28, 28)
    sample_input = torch.randn(1, 1, 28, 28)  # Batch size = 1, 单通道 28x28
    output = model(sample_input)
    print("Output shape:", output.shape)  # 应该是 (1, 10)
