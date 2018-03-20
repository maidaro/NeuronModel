import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import *
from keras.callbacks import *

def cal_pyx(kernel_x_y1, kernel_x_y0, py, xs):
	px_y1 = kernel_x_y1(xs) * py
	px_noty1 = kernel_x_y0(xs) * (1 - py)
	return px_y1 / (px_y1 + px_noty1)

df_raw=pd.DataFrame({
    'A':np.random.normal(size = 4096),
    'T':np.random.randint(2, size=4096)},
    columns=['A','T'])

df_sample=df_raw.sample(n=1000)

data_xy = np.vstack([df_sample.iloc[:,0].ravel(), df_sample.iloc[:,1].ravel()])
data_x = df_sample.iloc[:,0]
#print(data_xy)
kernel_xy = stats.gaussian_kde(data_xy)
kernel_xy.set_bandwidth(0.3)
kernel_x = stats.gaussian_kde(data_x.ravel())
kernel_x.set_bandwidth(0.3)

yeval = lambda x : x == 1
data_y1 = df_sample[yeval(df_sample.iloc[:,1])].iloc[:,0]
data_y0 = df_sample[~yeval(df_sample.iloc[:,1])].iloc[:,0]
py = len(data_y1)/(len(data_y1) + len(data_y0))
kernel_x_y1 = stats.gaussian_kde(data_y1.ravel())
kernel_x_y1.set_bandwidth(0.3)
kernel_x_y0 = stats.gaussian_kde(data_y0.ravel())
kernel_x_y0.set_bandwidth(0.3)

print(py, kernel_xy.integrate_box([-np.inf, df_sample.iloc[:,1].values.mean()], [np.inf, np.inf]))

plt.figure()
xs = np.linspace(df_sample.iloc[:,0].min(), df_sample.iloc[:,0].max(), 500)
ys = np.linspace(df_sample.iloc[:,1].min(), df_sample.iloc[:,1].max(), 50)
# multivarian density
plt.subplot(2, 2, 1)
xx, yy = np.meshgrid(xs, ys)
pos_xy = np.vstack([xx.ravel(), yy.ravel()])
pdf_xy = kernel_xy(pos_xy).reshape(xx.shape)
#print(pos_xy.shape, pos_xy)
#print(pdf_xy.shape, pdf_xy)
sns.heatmap(pdf_xy, cbar=False)
# pyx density computed by fxy(x,y)/fy(y)
# py computed by integration p(y) for (0.5, 1.0)
plt.subplot(2, 2, 2)
plt.plot(xs, kernel_x_y1(xs))
plt.title('f(T=1)')
# pyx method2
plt.subplot(2, 2, 3)
pyx = cal_pyx(kernel_x_y1, kernel_x_y0, py, xs)
#print(pyx.shape, pyx)
plt.plot(xs, pyx.reshape(xs.shape))
plt.title('P(T=1)')
plt.subplot(2, 2, 4)
data_learn = df_sample.iloc[:,0].values
x_data = np.array([(0, x) for x in data_learn]).reshape([-1,2])
y_data = cal_pyx(kernel_x_y1, kernel_x_y0, py, data_learn)
print(y_data)
model = Sequential()
model.add(Dense(2, activation='sigmoid', input_dim=2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape', 'acc'])
history=model.fit(x_data, y_data, epochs=20, verbose = 0)
print(history.history)
plt.plot(history.history['loss'], 'b', label='loss')
plt.twinx()
plt.plot(history.history['mean_absolute_percentage_error'], 'r', label='mape')
plt.legend()

# df_test = df_raw.sample(n=10)
# data_test = df_test.iloc[:,0].values
# x_test = np.array([(0, x) for x in data_test]).reshape([-1,2])
# y_test = cal_pyx(kernel_x_y1, kernel_x_y0, py, data_test)
# print(x_test)
# print(y_test)
# print(model.predict(x_test))

plt.show()