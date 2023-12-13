# coding=UTF-8

import math
 
# 计算平均绝对误差 MAE
def MAE(predicted, target):
    return (abs(target - predicted)).mean()

# 计算均方误差 MSE
def MSE(predicted, target):
    return ((target - predicted) ** 2).mean()
 
# 计算均方根误差 RMSE
def RMSE(predicted, target):
    return math.sqrt(MSE(predicted, target))
 
# 计算正规化均方根误差 NRMSE
def NRMSE(predicted, target):
    return RMSE(predicted, target) / (target.max() - target.min())

