
# coding: utf-8

# In[1]:


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

stats = list()
n_iterations = 500
for i in range(n_iterations):
	train_X, test_X, train_Y, test_Y = train_test_split(Xinput,Youtput,test_size=0.50)
	model = Lasso(alpha=1e-3, max_iter=100000).fit(train_X,train_Y)
	prediction = model.predict(test_X)

	score = mean_squared_error(test_Y,prediction)

	stats.append(score)
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1-alpha)/2))*100
upper = min(1.0,np.percentile(stats, p))
print((alpha*100), "confidence interval ", (lower), " and ", (upper))

