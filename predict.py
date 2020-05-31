import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('股票数据.xlsx', sheet_name='Sheet1')
df.set_index('时间', inplace=True)  # 索引
df.drop(['开盘', '最高', '最低', '成交量'], axis=1, inplace=True)
close_robust = RobustScaler()  # 按比例调整收盘价用于反向转换
close_robust.fit(df[['收盘']])
scaler_robust = RobustScaler()  # 归一化
df = pd.DataFrame(scaler_robust.fit_transform(df),
                  columns=df.columns,
                  index=df.index)


def split_series(seq, in_num, out_num):
    X, y = [], []
    for i in range(len(seq)):
        # 查找当前序列的结尾
        end = i + in_num
        o_end = end + out_num
        # 超出长度则跳出循环
        if o_end > len(seq):
            break
        # x 过去的价格和指标，y 未来的价格
        seq_x, seq_y = seq[i:end, :], seq[end:o_end, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# 现实的真实天数
real_day = 100
# 要预测的天数
predict_day = 15
# 特征
shape = df.shape[1]
# 将数据分成适当的顺序
X, y = split_series(df.to_numpy(), real_day, predict_day)

# 构建模型

# 实例化模型
model = Sequential()
# 输入层
model.add(LSTM(90,
         activation="tanh",
         return_sequences=True,
         input_shape=(real_day, shape)))
# 隐藏层
model.add(LSTM(30, activation="tanh", return_sequences=True))
model.add(LSTM(60, activation="tanh"))
# 输出层
model.add(Dense(predict_day))
# 模型汇总
model.summary()
# 使用选定的规范编译数据
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# 适配与训练
model.fit(X, y, epochs=50, batch_size=128, validation_split=0.1)

# 进行预测
y_pre = model.predict(np.array(df.tail(real_day)).reshape(1, real_day, shape))
# 预测值转回原始格式
y_pre = close_robust.inverse_transform(y_pre)[0]
# 创建预测价格
preds = pd.DataFrame(y_pre,
                     index=pd.date_range(start=df.index[-1] +
                                         timedelta(days=1),
                                         periods=len(y_pre),
                                         freq="B"),
                     columns=[df.columns[0]])
# 实际值转为原始价格
actual_data = pd.DataFrame(close_robust.inverse_transform(df[["收盘"]].tail(real_day)),
                           index=df.收盘.tail(real_day).index,
                           columns=[df.columns[0]]).append(preds.head(1))

#绘图
plt.figure(figsize=(16, 6))
plt.plot(actual_data, label="真实值")
plt.plot(preds, label="预测值")
plt.ylabel("价格")
plt.xlabel("时间")
plt.title(f"未来 {len(y_pre)} 天")
plt.legend()
plt.show()