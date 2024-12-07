import torch
import torch.optim as optim
from models.RecNet import DigitRecognitionCNN
from data.dataset import train_dataloader
import utils.utils as utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, epochs=10):
    """训练模型的函数"""
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {running_loss/len(train_loader):.4f}, "
            f"Accuracy: {100 * correct/total:.2f}%"
        )

if __name__ == "__main__":
    model = DigitRecognitionCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using {device} to train the model")

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_dataloader, criterion, optimizer, epochs=10)
    utils.save_model(model, 'model.pth')
