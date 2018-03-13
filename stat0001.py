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

print(df_sample)
# Draw joint probability P(A,T)
plt.figure()
plt.subplot(151)
sns.distplot(df_sample['A'], kde=True, rug=True, fit=stats.norm)
plt.subplot(152)
sns.distplot(df_sample['A'], bins=50, kde=True, rug=True, fit=stats.norm)
plt.subplot(153)
sns.kdeplot(df_sample['A'], bw=0.5)
plt.subplot(154)
sns.kdeplot(df_sample['A'], bw=0.3)
plt.subplot(155)
sns.kdeplot(df_sample['A'], bw=0.1)

kernel = stats.gaussian_kde(df_sample['A'])
xs = np.linspace(df_sample['A'].min(), df_sample['A'].max(), 1000)
kernel.set_bandwidth(0.337)
pdf=kernel.pdf(xs)
print('Samples')
print(xs)
print('Density')
print(pdf, pdf.sum(), kernel.integrate_box_1d(-np.inf, np.inf))

plt.figure()
plt.subplot(161)
sns.distplot(df_sample['A'], bins=50, kde=True, rug=True, fit=stats.norm)
plt.subplot(162)
sns.distplot(df_sample['T'], bins=50, kde=True, rug=True, fit=stats.norm)
plt.subplot(163)
sns.distplot(df_sample[df_sample['T'] != 0].iloc[:,0], bins=50, kde=True, rug=True, fit=stats.norm)
plt.subplot(164)
sns.distplot(df_sample[df_sample['T'] == 0].iloc[:,0], bins=50, kde=True, rug=True, fit=stats.norm)
plt.subplot(165)
plt.bar(xs, pdf)

plt.show()
