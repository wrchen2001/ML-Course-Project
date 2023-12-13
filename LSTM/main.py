# coding=UTF-8

import torch.nn
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import os

#import matplotlib as plot

from torch.nn.parallel import DataParallel
 
from Function import setup_seed, sliding_window, cmpt_error  # 这部分自己写的函数

torch.cuda.set_device(4)
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'

if torch.cuda.device_count() > 1:
    print("使用多个GPU.")

device_ids = [4, 5, 6, 7]  # 选择要使用的GPU设备编号
 
 
# 定义LSTM主模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) 
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

setup_seed(10)
 
# 读取数据
df = pd.read_csv("ETTh1.csv", parse_dates=["date"])

#print(df)

# 数据集划分
boundary_date_train = pd.to_datetime("2017/9/9 11:00")  # train/val/test=6:2:2
mask_train = df["date"] <= boundary_date_train
train = df.loc[mask_train].iloc[:, 1:]  # 得到训练集，其中iloc[:, 1:]表示排除掉第一列date

#print(train)

boundary_date_val = pd.to_datetime("2018/2/1 15:00")
mask_val = (df['date'] > boundary_date_train) & (df['date'] <= boundary_date_val)
val = df.loc[mask_val].iloc[:, 1:] 

#print(val)


# 对数据集进行归一化
scaler = MinMaxScaler()
scaler_train = MinMaxScaler()
scaler.fit(train)
scaler_train.fit(train.iloc[:, :1])
normalized_data = scaler.transform(df.iloc[:, 1:])  # 用训练集作模板归一化整个数据集

# print(normalized_data)

# file = "normalized_data.txt"

# np.savetxt(file, normalized_data)

# 基础参数设置
time_step = 96  # 时间步长，就是利用多少组历史数据进行预测
forecast_step = 96  # 预测步长，即预测未来第几步的数据
# forecast_step = 336
feature_size = 7  # 输入特征数


# 构造训练集和测试集
[train_input, train_output, val_input, val_output, test_input, test_output] = sliding_window(normalized_data, len(train), len(val), time_step,
                                                                      forecast_step, feature_size,
                                                                       sample_feature_compression=False)
#print(train_input)
# print(train_output)

# print(val_input.shape)
# print(val_output.shape)

# print(test_input.shape)
# print(test_output.shape)


# 输入、输出维度
input_dim = len(train_input[0, 0, :])
output_dim = 7
hidden_dim = 20  # 炼丹
torch.set_default_tensor_type(torch.DoubleTensor)
 
# 使用GPU运行
#device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
 
# 将输入转换为tensor, 并转移到多个GPU上
train_inputs_tensor = torch.from_numpy(train_input).cuda()
train_labels = torch.from_numpy(train_output).cuda()
val_inputs_tensor = torch.from_numpy(val_input).cuda()
val_labels = torch.from_numpy(val_output).cuda()
test_inputs_tensor = torch.from_numpy(test_input).cuda()




# 将输入数据移到多个GPU上
# train_inputs_tensor = train_inputs_tensor.cuda()
# train_labels = train_labels.cuda()
# val_inputs_tensor = val_inputs_tensor.cuda()
# val_labels = val_labels.cuda()
# test_inputs_tensor = test_inputs_tensor.cuda()

# 将模型移到多个GPU上
#model = DataParallel(model, device_ids=device_ids)

# print(train_inputs_tensor.size())
# print(train_labels.size())
# print(val_inputs_tensor.size())
# print(val_labels.size())
# print(test_inputs_tensor.size())

# model = LSTM(input_dim, hidden_dim, output_dim).to(device)
# train_outputs_tensor = model(train_inputs_tensor)
# print(train_outputs_tensor.size())


# 指定参数和损失函数
epochs = 8000  # 迭代次数
learning_rate = 0.003  # 学习率
 
# 多次运行，方便求误差平均值
train_prediction_set = []
prediction_set = []
error = []
start = time.perf_counter()  # 运行开始时间

save_dir = "checkpoints"

# 多次运行取平均值
multi_times = 5   # 运行次数
for times in range(multi_times):
    model = LSTM(input_dim, hidden_dim, output_dim)
    model = DataParallel(model, device_ids = device_ids).cuda()
    if times == 0:
        print(model)   # 查看神经网络模型
        print()
    
    print("第"+str(times+1)+"次运行: ")
    print("-------------------------")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 损失函数

    # 训练模型
    train_predicted = 0  # 用来保存训练集预测数据

    for epoch in range(epochs):
        optimizer.zero_grad()
        train_outputs_tensor = model(train_inputs_tensor)
        loss = criterion(train_outputs_tensor, train_labels)
        # #print(loss)

        # loss_scalar = loss.item()
        # loss_str = str(loss_scalar)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:    # 每100次训练输出一次损失值
            print(f'Training loss at epoch {epoch + 1}, loss {loss}')
            print("-------------------------------")
        if epoch == epochs - 1:
            train_predicted = train_outputs_tensor.detach().cpu().numpy()
        
        # 验证集上验证
        val_outputs_tensor = model(val_inputs_tensor)
        val_loss = criterion(val_outputs_tensor, val_labels)

        # val_loss_scalar = val_loss.item()
        # val_loss_str = str(val_loss_scalar)

        if (epoch + 1) % 100 == 0:
            print(f'Validation loss at epoch {epoch + 1}: {val_loss}')
            print()
            print()
    
    save_path = os.path.join(save_dir, f'model_{times+1}.pth')
    torch.save(model.state_dict(), save_path)

    print()

 
# 预测结果
predicted = model(test_inputs_tensor).detach().cpu().numpy()

# 逆缩放
train_predicted = scaler_train.inverse_transform(train_predicted)  # 训练集预测数据
predicted = scaler_train.inverse_transform(predicted)  # 预测值
target = scaler_train.inverse_transform(test_output)  # 目标值

# 计算误差
error.append(cmpt_error(predicted, target))
# 保存每次预测结果
train_prediction_set.append(train_predicted)
prediction_set.append(predicted)

end = time.perf_counter()  # 运行结束时间
runTime = end - start
print("Run time: ", runTime)  # 输出运行时间
 
# 数据排序
train_prediction_set = np.array(train_prediction_set)[:, :, 0].T
prediction_set = np.array(prediction_set)[:, :, 0].T
error = np.array(error).T
prediction_set = np.vstack([train_prediction_set, prediction_set])
error_prediction = pd.DataFrame(np.vstack([error, prediction_set]))  # 将误差和预测数据堆叠起来，方便排序
error_prediction = error_prediction.sort_values(by=2, axis=1)  # NRMSE在第三行，以NRMSE从小到大排序
 
# 保存数据
# error_prediction.iloc[3:, :]是因为前三行是误差，如果用了更多的误差指标记得修改
prediction_set = pd.DataFrame(np.array(error_prediction.iloc[3:, :]), columns=[i for i in range(1, multi_times + 1)])
error = pd.DataFrame(np.array(error_prediction.iloc[:3, :]), columns=[i for i in range(1, multi_times + 1)],
                     index=['MAE', 'RMSE', 'NRMSE'])
prediction_set.to_excel('LSTM.xlsx', index=False, sheet_name='LSTM')
with pd.ExcelWriter('LSTM.xlsx', mode='a', engine='openpyxl') as writer:
    error.to_excel(writer, sheet_name='error')
