import torch
import torch.nn.functional as F

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 权重初始化
        self.W1 = torch.randn(input_size, hidden_size) * 0.01
        self.b1 = torch.zeros(1, hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.01
        self.b2 = torch.zeros(1, output_size)

        # 将模型移动到GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W1, self.b1, self.W2, self.b2 = self.W1.to(self.device), self.b1.to(self.device), self.W2.to(self.device), self.b2.to(self.device)

    # 激活函数 ReLU
    def relu(self, Z):
        return torch.maximum(torch.tensor(0.0, device=self.device), Z)

    # Softmax 激活函数
    def softmax(self, Z):
        exp_Z = torch.exp(Z - torch.max(Z, dim=1, keepdim=True).values)
        return exp_Z / torch.sum(exp_Z, dim=1, keepdim=True)

    # 前向传播
    def forward(self, X):
        X = X.view(X.size(0), -1)
        self.Z1 = torch.matmul(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = torch.matmul(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    # 交叉熵损失函数
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        loss = -torch.sum(Y * torch.log(Y_hat + 1e-9)) / m  # 防止 log(0) 的数值问题
        return loss

    # 反向传播
    def backward(self, X, Y, Y_hat, learning_rate):
        X = X.view(X.size(0), -1)
        m = X.shape[0]
        
        # 计算输出层的误差
        dZ2 = Y_hat - Y
        dW2 = torch.matmul(self.A1.T, dZ2) / m
        db2 = torch.sum(dZ2, dim=0, keepdim=True) / m

        # 计算隐藏层的误差
        dA1 = torch.matmul(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0).float()  # ReLU 激活函数的导数

        dW1 = torch.matmul(X.T, dZ1) / m
        db1 = torch.sum(dZ1, dim=0, keepdim=True) / m

        # 更新权重和偏置
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # 保存模型
    def save_model(self, filename):
        torch.save({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }, filename)

    # 加载模型
    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.W1 = checkpoint['W1']
        self.b1 = checkpoint['b1']
        self.W2 = checkpoint['W2']
        self.b2 = checkpoint['b2']
        
        # 将加载的模型移动到相同的设备上
        self.W1, self.b1, self.W2, self.b2 = self.W1.to(self.device), self.b1.to(self.device), self.W2.to(self.device), self.b2.to(self.device)