import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

df_raw=pd.DataFrame({
    'A':np.random.random_sample(size = 4096) * 1000,
    'T':np.random.randint(2, size=4096)},
    columns=['A','T'])

df_sample=df_raw.sample(n=1000)
r = np.vstack([df_sample.iloc[:,0].ravel(), df_sample.iloc[:,1].ravel()])
#print(r)
kernel_xy = stats.gaussian_kde(r)
kernel_xy.set_bandwidth(0.3)

data_x = df_sample.iloc[:,0]
kernel_x = stats.gaussian_kde(data_x.ravel())
kernel_x.set_bandwidth(0.3)
#print(kernel_x(xx.ravel()))

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
print(pos_xy.shape, pos_xy)
print(pdf_xy.shape, pdf_xy)
sns.heatmap(pdf_xy, cbar=False)
# pyx density
plt.subplot(2, 2, 3)
pdf_x = kernel_x(xs).reshape([1,-1])
print(pdf_x.shape, pdf_x)
pdf_pyx = pdf_xy / pdf_x
print(pdf_pyx.T.shape, pdf_pyx.T)
sns.heatmap(pdf_pyx, cbar=False)
# pyx method1
plt.subplot(2, 2, 2)
pyx=pdf_pyx.T[:,25:].sum(axis=1)
plt.plot(xs, pyx)
plt.title('P(T=1)')
# pyx method2
plt.subplot(2, 2, 4)
px_y1 = kernel_x_y1(xs) * py
px_noty1 = kernel_x_y0(xs) * (1 - py)
pyx = px_y1 / (px_y1 + px_noty1)
print(pyx.shape, pyx)
plt.plot(xs, pyx.reshape(xs.shape))
plt.title('P(T=1)')
plt.show()
