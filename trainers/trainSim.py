import torch
from models.SimpleNet import SimpleNN
from data.dataset import train_dataloader, eval_dataloader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, eval_loader, learning_rate, epochs=10, writer=None):
    """训练模型的函数"""
    totalloss = []
    totalcorrect = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 将标签转为 one-hot 编码
            Y = torch.zeros(labels.size(0), 10).to(device)
            Y.scatter_(1, labels.view(-1, 1), 1)

            outputs = model.forward(inputs)
            loss = model.compute_loss(Y, outputs)
            # 反向传播并计算梯度
            model.backward(inputs, Y, outputs, learning_rate)

            totalloss.append(loss.item())
            running_loss += loss.item()


            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            totalcorrect.append(correct)

            if writer:
                writer.add_scalar('Training Loss', running_loss / total, epoch * len(train_loader) + total)
                writer.add_scalar('Training Accuracy', 100 * correct / total, epoch * len(train_loader) + total)

            if total % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"CurrentLoss: {running_loss/total:.4f}, "
                    f"CurrentAccuracy: {100 * correct/total:.2f}%"
                )

        eval_loss = 0
        eval_correct = 0
        eval_total = 0
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 将标签转为 one-hot 编码
            Y = torch.zeros(labels.size(0), 10).to(device)
            Y.scatter_(1, labels.view(-1, 1), 1)

            outputs = model.forward(inputs)
            loss = model.compute_loss(Y, outputs)

            eval_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {running_loss/len(train_loader):.4f}, "
            f"Accuracy: {100 * correct/total:.2f}%"
        )
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"EvalLoss: {eval_loss/len(eval_loader):.4f}, "
            f"EvalAccuracy: {100 * eval_correct/eval_total:.2f}%"
        )

    return totalloss, totalcorrect

if __name__ == "__main__":
    writer = SummaryWriter(log_dir='runs/experiment_1')

    # 初始化模型
    model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
    print(f"Using {device} to train the model")

    # 训练模型
    learning_rate = 0.001
    loss, correct = train(model, train_dataloader, eval_dataloader, learning_rate, epochs=5, writer=writer)

    writer.close()

    # 保存模型时使用时间戳生成唯一文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"./checkpoints/model_{timestamp}.pth"
    model.save_model(model_filename)