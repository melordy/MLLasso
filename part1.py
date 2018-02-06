
# coding: utf-8

# In[2]:


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


alpha_lasso = [1e-3,1e-2,1e-1, 1]
np.set_printoptions(threshold=sys.maxsize)


for i in range(len(alpha_lasso)):
	count = 0
	model = Lasso(alpha=alpha_lasso[i],max_iter=100000).fit(Xinput,Youtput)

	for i in range(0,4495):
		if model.coef_[i] != 0:
			count=count+1
	scores = cross_val_score(model,Xinput,Youtput,cv=10,scoring='neg_mean_squared_error')
	print("Cross Score: ", np.abs(scores))
	sum = 0
	for j in range(0, 10):
		sum = sum + np.abs(scores[j])
	msescore = sum/10
	print ("MSE:", msescore)

	print("Number of non zero coefficients: ",count)

