import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import CNNModel
from src.data_preprocessing import load_data
from src.evaluate import evaluate_model
import os

def train_model():
    # 加载数据
    train_loader, test_loader = load_data(batch_size=64)

    # 初始化模型
    model = CNNModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # 每 100 个 mini-batch 输出一次损失
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    # 保存模型
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')
    torch.save(model.state_dict(), 'outputs/models/mnist_cnn_model.pth')

if __name__ == '__main__':
    # train_model()
    evaluate_model()
