import matplotlib as plt
import torch
from models.RecNet import DigitRecognitionCNN
from trainers.trainRec import train
import torch.optim as optim
from data.dataset import train_dataloader
from utils.utils import save_model


model = DigitRecognitionCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using {device} to train the model")

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_dataloader, criterion, optimizer, epochs=10)
save_model(model, 'model.pth')

# 训练模型并获取损失值
loss_values = train(model, train_dataloader, criterion, optimizer, epochs=10)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.show()
