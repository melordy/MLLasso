
# coding: utf-8

# In[3]:


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame, read_fwf

from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

df = pd.read_excel('ecs171.dataset.xlsx',sep=',',header=0)

dataset=df.values
Xinput = dataset[:,6:4501]
Youtput = dataset[:,5]

np.set_printoptions(threshold=sys.maxsize)
model = Lasso(alpha=1e-3,max_iter=100000).fit(Xinput,Youtput)
    
Xvalues = Xinput.mean(0)
Xvalues = np.asarray([Xvalues])



print("Y output:", model.predict(Xvalues))

