#LSTM 特征temp ，dew SO2
#利用过去1小时数居预测后1小时数据
#利用过去3 6 12 小时数据分别预测后3 6 12 24 小时的AQI
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import LSTM
dataset = read_csv('new_pollution(1).csv')
examDf = DataFrame(dataset)
new_examDf = DataFrame(examDf.drop('date',axis=1))
print(new_examDf.head())
values = new_examDf.values #数据转换为数组
print(new_examDf.AQI[0])
for i in range(4800,8490):
    if new_examDf.AQI[i+1] > new_examDf.AQI[i]+20 or new_examDf.AQI[i+1] < new_examDf.AQI[i]-20:
        print(i)