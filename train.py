import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import CNNModel
from src.data_preprocessing import load_data
from src.evaluate import evaluate_model,evaluate_model_fail_list,evaluate_with_analysis
import os
import matplotlib.pyplot as plt


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

    # 用于记录 loss
    epoch_losses = []

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

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
            batch_count += 1

            if i % 100 == 99:  # 每 100 个 mini-batch 输出一次损失
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}")

                # 记录每个 epoch 的平均 loss

        avg_loss = running_loss / batch_count
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    # 保存模型
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')
    torch.save(model.state_dict(), 'outputs/models/mnist_cnn_model.pth')

    # ==============================
    #       绘制 loss 曲线
    # ==============================
    if not os.path.exists('outputs/img'):
        os.makedirs('outputs/img')

    plt.figure()
    plt.plot(epoch_losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputs/img/loss_curve.png", dpi=300)
    plt.close()
    print("Loss 曲线已保存到 outputs/img/loss_curve.png")



if __name__ == '__main__':
    # train_model()
    # evaluate_model_fail_list()
    evaluate_with_analysis()