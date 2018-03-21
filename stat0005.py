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

def transform_xs(x):
	return [0, x]

df_raw=pd.DataFrame({
    'A':np.random.normal(size=10000),
    'T':np.random.randint(2, size=10000)},
    columns=['A','T'])

df_sample=df_raw.sample(n=2048)

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

plt.figure(figsize=[5.5, 8])
plt.suptitle('Learning Norm Classifier', fontsize=16)
xs = np.linspace(df_sample.iloc[:,0].min(), df_sample.iloc[:,0].max(), 500)
ys = np.linspace(df_sample.iloc[:,1].min(), df_sample.iloc[:,1].max(), 50)
# multivarian density
plt.subplot(3, 2, 1)
xx, yy = np.meshgrid(xs, ys)
pos_xy = np.vstack([xx.ravel(), yy.ravel()])
pdf_xy = kernel_xy(pos_xy).reshape(xx.shape)
#print(pos_xy.shape, pos_xy)
#print(pdf_xy.shape, pdf_xy)
sns.heatmap(pdf_xy, cbar=False, xticklabels=False, yticklabels=False)
# pyx density computed by fxy(x,y)/fy(y)
# py computed by integration p(y) for (0.5, 1.0)
plt.subplot(3, 2, 2)
plt.plot(xs, kernel_x_y1(xs))
plt.title('f(T=1)')
# pyx method2
learn_data = df_sample
x_data = np.array([transform_xs(x) for x in learn_data.iloc[:,0].values])
#y_data = cal_pyx(kernel_x_y1, kernel_x_y0, py, learn_data.iloc[:,0].values)
y_data = np.array([(0, 1) if x==1 else (1, 0) for x in learn_data.iloc[:,1].values])
print(y_data)
test_data = df_raw.sample(n=2048)
x_test = np.array([transform_xs(x) for x in test_data.iloc[:,0].values])
y_test = np.array([(0, 1) if x==1 else (1, 0) for x in test_data.iloc[:,1].values])
model = Sequential()
model.add(Dense(x_data.shape[1] * 5, activation='relu', input_dim=x_data.shape[1]))
model.add(Dense(x_data.shape[1] * 5, activation='relu'))
model.add(Dense(y_data.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history=model.fit(x_data, y_data, epochs=30, verbose = 0, validation_data=(x_test, y_test))
x_summary = np.array([transform_xs(x) for x in xs])
y_summary = model.predict(x_summary)
plt.subplot(3, 2, 3)
pyx = cal_pyx(kernel_x_y1, kernel_x_y0, py, xs)
#print(pyx.shape, pyx)
plt.scatter(xs, pyx.reshape(xs.shape), s=1, c='b', label='sample')
plt.scatter(xs, y_summary[:,1], s=1, c='r', label='predict')
plt.legend()
plt.title('P(T=1)')
plt.subplot(3, 2, 4)
xs_s = np.linspace(df_sample.iloc[:,0].min(), df_sample.iloc[:,0].max(), 100)
ys_s = np.array([1 if 2 * x - 1 >= 0 else 0 for x in cal_pyx(kernel_x_y1, kernel_x_y0, py, xs_s)])
ys_summary = np.array([1 if x[0] <= x[1] else 0 for x in model.predict(np.array([transform_xs(x) for x in xs_s]))])
plt.scatter(xs_s, ys_s, s=2, c='b', label='sample')
plt.scatter(xs_s, ys_summary, s=2, c='r', label='predict')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='loss(test)')
plt.legend()
plt.subplot(3, 2, 6)
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='accuracy(test)')
plt.legend()
plt.show()
