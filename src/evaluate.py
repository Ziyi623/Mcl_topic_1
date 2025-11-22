import torch
from torch.utils.data import DataLoader
from src.model import CNNModel
from src.data_preprocessing import load_data
import os
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

def evaluate_model_fail_list():
    #     加载数据
    _, test_loader = load_data(batch_size=64)

    #     加载模型
    model = CNNModel()
    model.load_state_dict(torch.load('outputs/models/mnist_cnn_model.pth'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建保存失败案例目录
    fail_dir = 'outputs/fail_cases'
    if not os.path.exists(fail_dir):
        os.makedirs(fail_dir)

    fail_list = []

    #     评估模型
    correct = 0
    total = 0
    idx_global = 0  # 用于记录全局样本编号

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # ============================
            #    保存预测错误的样本
            # ============================
            for i in range(labels.size(0)):
                idx_global += 1
                if predicted[i] != labels[i]:
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()

                    # 保存图像文件
                    filename = f"true_{true_label}_pred_{pred_label}_idx_{idx_global}.png"
                    filepath = os.path.join(fail_dir, filename)

                    # 单张图像保存（需要保持图像 1×28×28 格式）
                    vutils.save_image(inputs[i].cpu(), filepath)

                    # 写入记录
                    fail_list.append(f"{filename}  true={true_label}, pred={pred_label}")

    # ===================
    #   写入失败清单文件
    # ===================
    list_path = os.path.join(fail_dir, "fail_list.txt")
    with open(list_path, "w") as f:
        for line in fail_list:
            f.write(line + "\n")

    print(f"测试集准确率: {100 * correct / total:.2f}%")
    print(f"错误样本已保存至: {fail_dir}")
    print(f"失败清单已保存至: {list_path}")

def evaluate_with_analysis():
    # ===================
    #     加载数据
    # ===================
    _, test_loader = load_data(batch_size=64)

    # ===================
    #     加载模型
    # ===================
    model = CNNModel()
    model.load_state_dict(torch.load('outputs/models/mnist_cnn_model.pth'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_labels = []
    all_preds = []

    # ===================
    #   逐批评估
    # ===================
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # ===================
    #     准确率
    # ===================
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = (all_labels == all_preds).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # ===============================
    #      ① 混淆矩阵计算
    # ===============================
    cm = confusion_matrix(all_labels, all_preds)

    # 保存路径
    if not os.path.exists("outputs/img"):
        os.makedirs("outputs/img")

    # ===============================
    #     ② 绘制混淆矩阵图
    # ===============================
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (MNIST)")

    plt.savefig("outputs/img/confusion_matrix.png", dpi=300)
    plt.close()
    print("混淆矩阵已保存到 outputs/img/confusion_matrix.png")

    # ===============================
    #   ③ Top-10 最容易混淆的数字对
    # ===============================
    # 去掉对角线（正确预测）
    mixed_errors = cm.copy().astype(float)
    np.fill_diagonal(mixed_errors, 0)

    # 计算最常见的混淆对
    confusions = []
    for true_label in range(10):
        for pred_label in range(10):
            if mixed_errors[true_label, pred_label] > 0:
                confusions.append((true_label, pred_label, mixed_errors[true_label, pred_label]))

    # 根据混淆次数排序
    confusions.sort(key=lambda x: x[2], reverse=True)

    # 取前 10 名
    top10 = confusions[:10]

    # 保存到文本文件
    with open("outputs/img/top_confusions.txt", "w") as f:
        for true_label, pred_label, count in top10:
            f.write(f"真实 {true_label} → 预测 {pred_label} ： {int(count)} 次\n")

    print("Top-10 容易混淆数字对已保存到 outputs/img/top_confusions.txt")
    print("Done.")

if __name__ == '__main__':
    evaluate_model()
