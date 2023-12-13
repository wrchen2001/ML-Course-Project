# coding=UTF-8

import torch
import random
import numpy as np
import pandas as pd
 
from Error_calculation import MAE, MSE, NRMSE, RMSE
 

# 设置随机数种子，保证每次运行结果相同
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True 
 

# 用滑动窗口来制作数据集
def sliding_window(normalized_data, train_length, val_length, time_step, forecast_step, 
                    feature_size=1, sample_feature_compression=True):
    
    """
    normalized_data: 经过归一化处理的时间序列数据
    train_length: 训练集的长度
    val_length: 验证集的长度
    time_step: 滑动窗口的时间步长, 即用多少个时间步去构造输入序列
    forecast_step: 预测步长, 即从当前步起, 预测未来的时间步数
    feature_size: 特征维度
    sample_feature_compression: 是否将特征压缩到一维数组中
    """

    inputs = []
    outputs = []
    for i in range(len(normalized_data) - time_step - forecast_step + 1): 
        package = []
        for j in range(feature_size):
            package.append(normalized_data[i:i + time_step][:, j])
        if sample_feature_compression == True:
            inputs.append(np.array(package).reshape(1, -1)[0, :])
        else:
            inputs.append(np.array(package).T)
        #outputs.append(normalized_data[i + time_step][-1])
        outputs.append(normalized_data[i + time_step][:])
    

    inputs = np.array(inputs)
    #outputs = np.array(outputs).reshape(-1, 1)
    outputs = np.array(outputs)

    train_input = inputs[:train_length - time_step - forecast_step + 1]
    train_output = outputs[:train_length - time_step - forecast_step + 1]
    val_input = inputs[train_length - time_step - forecast_step + 1: train_length + val_length - time_step - forecast_step + 1]
    val_output = outputs[train_length - time_step - forecast_step + 1: train_length + val_length - time_step - forecast_step + 1]
    test_input = inputs[train_length + val_length - time_step - forecast_step + 1:]
    test_output = outputs[train_length + val_length - time_step - forecast_step + 1:]
 
    return [train_input, train_output, val_input, val_output, test_input, test_output]
 

# 计算预测值和目标值之间的误差
def cmpt_error(predicted, target):
    contrast = pd.DataFrame(np.hstack((predicted, target)), columns=['预测值', '目标值'])
    print(contrast)

    mae = MAE(predicted, target)
    mse = MSE(predicted, target)
    rmse = RMSE(predicted, target)
    nrmse = NRMSE(predicted, target)
 
    print('MAE误差: ', mae)
    print('MSE误差: ', mse)
    print('RMSE误差: ', rmse)
    print(f'NRMSE误差: {"%.4f" % (nrmse * 100)}%')
 
    return [mae, mse, rmse, nrmse]