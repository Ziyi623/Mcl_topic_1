# README

该项目为BUAA机器学习课程作业一 CNN手写数字识别

CNN网络架构已经完成，训练的模型样例在output文件夹中，准确率为98.87%

其实手写数字本身有MNIST数据集可以直接download，但课程给出了 TestSet 和 TrainingSet ,就以课程给出的数据集为训练集和测试集。在 data_preprocessing.py 中写了继承 Dataset 的类 CustomMNISTDataset 用于数据预处理。

数据处理结果已经生成，分析内容可参考report.md（课程要上交的实验报告，懒得在readme里再写一遍了）

**注意:** report.md 中的部分图像链接来源于outputs文件夹中的输出结果，如果重新训练和评估结果生成新的图像可能会导致链接失效。可自行先导出保存为pdf文件。