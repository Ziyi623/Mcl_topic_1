# README

该项目为BUAA机器学习课程作业一 CNN手写数字识别

CNN网络架构已经完成，训练的模型样例在output文件夹中，准确率为98.87%

其实手写数字本身有MNIST数据集可以直接download，但课程给出了 TestSet 和 TrainingSet ,就以课程给出的数据集为训练集和测试集。在 data_preprocessing.py 中写了继承 Dataset 的类 CustomMNISTDataset 用于数据预处理。

该版本暂时不包含图像绘制，我会在考虑好实验报告需要哪些指标数据后进行该部分的更新。