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
print(r)

#(s, bins) = np.histogramdd(r, bins=[50, 10])
#print(s)
#print(bins)

#plt.figure()
#plt.hist2d(df_sample.iloc[:,0], df_sample.iloc[:,1], bins=50)

xx, yy = np.mgrid[df_sample.iloc[:,0].min():df_sample.iloc[:,0].max():500j, df_sample.iloc[:,1].min():df_sample.iloc[:,1].max():50j]
pos = np.vstack([xx.ravel(), yy.ravel()])
#print(pos)
kernel = stats.gaussian_kde(r)
kernel.set_bandwidth(0.3)
pdf = kernel.pdf(pos)
print(pdf)

plt.figure()
sns.heatmap(pdf.reshape(xx.shape))
plt.show()
