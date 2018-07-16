import csv
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import norm
from scipy import stats
# %matplotlib inline

df_train = pd.read_csv('filename.csv')

le = preprocessing.LabelEncoder()


(i,j)=df_train.shape
h=np.zeros((i,j),dtype=np.float32)
a=df_train.columns

for k in range(j):
    if k > 0:z
        z = np.array(df_train[a[k]])
        if type(z[0]) == str:
            df = df_train[a[k]].fillna('0')
            le.fit(df)
            c = le.transform(df)
            for v in range(i):
                if z[v]==np.nan:
                    h[v][k] = np.nan
                else:
                    h[v][k] = c[v]
        else:
            for v in range(i):
                h[v][k] = z[v]

# h=np.array(h).reshape(j,i)
# np.save("filename.npy",a)
# b = np.load("filename.npy")













































