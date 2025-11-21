import torch
from torch.utils.data import DataLoader
from src.model import CNNModel
from src.data_preprocessing import load_data

def evaluate_model():
    # 加载数据
    _, test_loader = load_data(batch_size=64)

    # 加载训练好的模型
    model = CNNModel()
    model.load_state_dict(torch.load('outputs/models/mnist_cnn_model.pth'))
    model.eval()  # 设置模型为评估模式

    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    evaluate_model()
