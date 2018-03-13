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
r = np.hstack((df_sample.iloc[:,0].reshape((-1,1)), df_sample.iloc[:,1].reshape((-1,1))))
print(r)
(s, bins) = np.histogramdd(r, bins=(50, 10))
print(s)
print(bins)

plt.figure()
#plt.subplot(151)
plt.hist2d(df_sample.iloc[:,0], df_sample.iloc[:,1], bins=50)
plt.show()
