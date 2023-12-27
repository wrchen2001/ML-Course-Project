# 基于LSTM的多变量时序序列预测模型



## Note

在项目中我们使用了两种衡量预测误差的指标：平均绝对误差 (Mean Absolute Error，MAE)和均方误差 (Mean Squared Error，MSE)。

## Dateset

项目使用的数据集来自于[code](https://github.com/zhouhaoyi/Informer2020)，选择里面的ETTh1.csv作为数据集，该数据集包含17420条数据，每条数据均包含8维特征，包括数据点的记录日期 (date)、预测值“油温 (OT)”以及6个不同类型的外部载值，Train/Val/Test划分为6:2:2，即训练集有10452条数据，验证集和测试集分别有3484条数据。

## Libarary

```
pytorch == 1.7.1 + cu11.0
numpy == 1.24.3
pillow == 9.5.0
scikit-learn == 1.3.2
scipy == 1.10.1
matplotlib == 3.7.4
```

## Training resources

我们在八张NVIDIA GeForce RTX 3090上并行训练，默认训练轮数为250轮，总共训练5次，取最终结果的平均值，训练时长大概需要10-15分钟左右。
