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

kernel_x = stats.gaussian_kde(df_sample.iloc[:,0].ravel())
kernel_x.set_bandwidth(0.3)
#print(kernel_x.pdf(xx.ravel()))

yeval = lambda x : x == 1
data_y1 = df_sample[yeval(df_sample.iloc[:,1])].iloc[:,0]
data_y0 = df_sample[~yeval(df_sample.iloc[:,1])].iloc[:,0]
py = len(data_y1)/(len(data_y1) + len(data_y0))

kernel_x_y1 = stats.gaussian_kde(data_y1.ravel())
kernel_x_y1.set_bandwidth(0.3)
kernel_x_y0 = stats.gaussian_kde(data_y0.ravel())
kernel_x_y0.set_bandwidth(0.3)

print(py, kernel_xy.integrate_box([-np.inf, df_sample.iloc[:,1].values.mean()], [np.inf, np.inf]))

xx, yy = np.mgrid[df_sample.iloc[:,0].min():df_sample.iloc[:,0].max():500j, df_sample.iloc[:,1].min():df_sample.iloc[:,1].max():50j]
pos = np.vstack([xx.ravel(), yy.ravel()])
pdf_xy = kernel_xy.pdf(pos)
#print(pdf_xy)
xs = np.linspace(df_sample.iloc[:,0].min(), df_sample.iloc[:,0].max(), 500)
plt.figure()
plt.subplot(2, 3, 1)
sns.heatmap(pdf_xy.reshape(xx.shape))
plt.subplot(2, 3, 2)
plt.plot(xx, pdf_xy.reshape(xx.shape) / kernel_x.pdf(xs))
plt.show()
