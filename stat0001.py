import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def df_scale(x, scale):
    return x.mul(pow(10, scale)).round()

df=pd.DataFrame({
    'A':np.random.uniform(size = 1024),
    'B':np.random.uniform(size = 1024),
    'C':np.random.uniform(size = 1024),
    'D':np.random.uniform(size = 1024),
    'E':np.random.uniform(size = 1024),
    'F':np.random.uniform(size = 1024),
    'G':np.random.uniform(size = 1024),
    'H':np.random.uniform(size = 1024),
    'T':np.random.randint(2, size=1024)},
    columns=['A','B','C','D','E','F','G','H','T'])

df_sample=pd.DataFrame({
    'A':df_scale(df.A, 2),
    'B':df_scale(df.B, 2),
    'C':df_scale(df.C, 2),
    'D':df_scale(df.D, 2),
    'E':df_scale(df.E, 2),
    'F':df_scale(df.F, 2),
    'G':df_scale(df.G, 2),
    'H':df_scale(df.H, 2),
    'T':np.random.randint(2, size=1024)},
    columns=['A','B','C','D','E','F','G','H','T'])

#df_tbl=df_sample.pivot_table(values=['T'], index=['A'], aggfunc=np.sum)
df_tbl_A=df_sample.groupby(by='A')
df_tbl=df_tbl_A['T'].sum()/df_sample['T'].count()
df_tbl.plot.bar()
plt.xticks([])
#plt.scatter(df_sample['A'], df_sample['T'])
#df_sample['A'].plot.hist()
plt.show()
