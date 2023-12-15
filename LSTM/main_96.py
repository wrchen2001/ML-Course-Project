import torch.nn
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
 
from Function import setup_seed, sliding_window

 
torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'


device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # 选择要使用的GPU设备编号
 
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

# 数据集划分
boundary_date_train = pd.to_datetime("2017/9/9 11:00")  # train/val/test=6:2:2
mask_train = df["date"] <= boundary_date_train
train = df.loc[mask_train].iloc[:, 1:]  # 得到训练集，其中iloc[:, 1:]表示排除掉第一列date

boundary_date_val = pd.to_datetime("2018/2/1 15:00")
mask_val = (df['date'] > boundary_date_train) & (df['date'] <= boundary_date_val)
val = df.loc[mask_val].iloc[:, 1:] 


# 对数据集进行归一化
scaler = MinMaxScaler()
scaler_train = MinMaxScaler()
scaler.fit(train)
scaler_train.fit(train.iloc[:, :1])
normalized_data = scaler.transform(df.iloc[:, 1:])  # 用训练集作模板归一化整个数据集


# 基础参数设置
time_step = 96  # 时间步长，就是利用多少组历史数据进行预测
forecast_step = 96  # 预测步长，即预测未来第几步的数据
feature_size = 7  # 输入特征数


# 构造训练集和测试集
[train_input, train_output, val_input, val_output, test_input, test_output] = sliding_window(normalized_data, len(train), len(val), time_step,
                                                                      forecast_step, feature_size,
                                                                       sample_feature_compression=False)


# 输入、输出维度
input_dim = len(train_input[0, 0, :])
output_dim = 7
hidden_dim = 20  # 炼丹
torch.set_default_tensor_type(torch.DoubleTensor)

 
# 将输入转换为tensor, 并转移到多个GPU上
train_inputs_tensor = torch.from_numpy(train_input).cuda()
train_labels = torch.from_numpy(train_output).cuda()
val_inputs_tensor = torch.from_numpy(val_input).cuda()
val_labels = torch.from_numpy(val_output).cuda()
test_inputs_tensor = torch.from_numpy(test_input).cuda()


# 指定参数和损失函数
epochs = 4000  # 迭代次数
learning_rate = 0.003  # 学习率
 
# 多次运行，方便求误差平均值
train_prediction_set = []
prediction_set = []
error = []
start = time.perf_counter()  # 运行开始时间

save_dir = "checkpoints_96"


multi_times = 5   # 运行次数
for times in range(multi_times):

    train_losses_96 = []  #存储训练集的损失
    val_losses_96 = []   # 存储验证集的损失

    model = LSTM(input_dim, hidden_dim, output_dim)
    model = DataParallel(model, device_ids = device_ids).cuda()
    
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

        train_losses_96.append(loss.item())

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

        val_losses_96.append(val_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Validation loss at epoch {epoch + 1}: {val_loss}')
            print()
            print()

    plt.figure(figsize=(50, 35))
    plt.plot(train_losses_96, label='Train Loss')
    plt.plot(val_losses_96, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'Figure_96/figure{times+1}.png', format='png')
    #plt.show()
    
    save_path = os.path.join(save_dir, f'model_96_{times+1}.pth')
    torch.save(model.state_dict(), save_path)

    print()

 
# 预测结果
predicted = model(test_inputs_tensor).detach().cpu().numpy()
predicted = scaler_train.inverse_transform(predicted)  # 预测值
target = scaler_train.inverse_transform(test_output)  # 目标值

predicted = np.array(predicted)
target = np.array(target)

# 计算误差

mse = ((predicted-target)**2).mean()
print("MSE:", mse)

mae = np.abs(predicted - target).mean()
print("MAE:", mae)

rmse = np.sqrt(((predicted - target) ** 2).mean())
print("RMSE:", rmse)

nrmse = rmse / (np.max(predicted) - np.min(predicted))
print("NRMSE:", nrmse)


end = time.perf_counter()  # 运行结束时间
runTime = end - start
print("Run time: ", runTime)  # 输出运行时间
 
