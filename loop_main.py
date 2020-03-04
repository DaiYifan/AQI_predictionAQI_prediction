#LSTM 特征temp ，dew SO2
#利用过去1小时数居预测后1小时数据
#利用过去3 6 12 小时数据分别预测后3 6 12 24 小时的AQI
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
dataset = read_csv('new_pollution(1).csv')
examDf = DataFrame(dataset)
new_examDf = DataFrame(examDf.drop('date',axis=1))
print(new_examDf.head())

values = new_examDf.values #数据转换为数组
# series = Series.from_array(examDf)
# series.plot()
# pyplot.show()


# # for i in range(4800,8490):
# #     if new_examDf.AQI[i+1] > new_examDf.AQI[i]+20 or new_examDf.AQI[i+1] < new_examDf.AQI[i]-20:
# #         print(i)
#
#
#时间转换

#convert series to supvised learning
def series_to_supervised(data,n_in=1,n_out=1,dropnan= True):
    n_vars = 1 if type(data) is list else data.shape[1] #
    df = DataFrame(data)
    cols, names = list(), list()

    #输入前n行数据 input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))  # 通过位移得到时间序列
        # 添加列名
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    #预测后n行数据 forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    print(agg)
    return agg

# 1: 归一化
values = values.astype('float32') #数据转换为浮点数
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

# 2: 指定特征数目和输入时序长度
n_hours = 1
n_features = 14
#frame as supervised learning
reframed = series_to_supervised(scaled,n_hours,1)
print(reframed.shape)
print(reframed)

# 3: 划分训练集和测试集
values = reframed.values
n_train_hours = 200 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 4: 划分训练集和测试集中的输入输出
n_obs = n_hours * n_features #观察值 n X 14
train_X, train_y = train[:, :n_obs], train[:,-n_features]
test_X, test_y = test[:,:n_obs], test[:,-n_features]

#5: 将输入转换为n_hour维形式
train_X = train_X.reshape((train_X.shape[0],n_hours,n_features))
test_X = test_X.reshape((test_X.shape[0],n_hours,n_features))

#LSTM
model = Sequential()
# LSTM参数依次为：输出维度，输入维度
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))  # LSTM模型
# 全连接层
model.add(Dense(1))
# 优化方法选择adam，损失函数选择均方差
model.compile(loss='mae', optimizer='adam')  # 确定代价函数及梯度下降法

#into loop
epochs_loop = [10,20]
# fit network
f = open("main.txt","w")
for ep in epochs_loop:
    history = model.fit(train_X, train_y, epochs=ep, batch_size=72,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -13:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    print(inv_yhat.shape)

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -13:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    print(inv_y.shape)
    # 计算均方差
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    abse = mean_absolute_error(inv_y,inv_yhat)
    print('show mae: %.3f' % abse)
    f.write(rmse)
    f.write("\n")
f.close()

