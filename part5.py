
# coding: utf-8

# In[44]:


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
 
from scipy import interp

from pandas import DataFrame, read_fwf

from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from sklearn import datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import label_binarize

from random import randrange, sample

def random_insert(lst, item):
    lst.insert(randrange(len(lst)+1), item)

from itertools import cycle
import random

def ROCCurve(Xinput,Yinput, graphType):
	
	random_state = np.random.RandomState(0)
	#newlist= [item for sublist in y_score for item in sublist]

	Yinput = Yinput.astype(float)

	if graphType == 0:
		n_classes = 18
	if graphType == 1:
		n_classes = 8


	cv = StratifiedKFold(n_splits=10)

	classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=random_state))

	fpr = dict()
	tpr =dict()
	roc_auc = dict()

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	recall = dict()
	precision = dict()
	pr_auc = dict()
	i = 0 #initial fold


    
	for train, test in cv.split(Xinput, Yinput):
		Yinput = label_binarize(Yinput, classes=list(range(n_classes)))
		
		y_score = classifier.fit(Xinput[train], Yinput[train]).decision_function(Xinput[test])

		flat_list = [item for sublist in y_score for item in sublist]
      
		fpr, tpr, _ = roc_curve(Yinput[test][1], flat_list[:n_classes])



		tprs.append(interp(mean_fpr, fpr, tpr))

		roc_auc = auc(fpr,tpr)
		aucs.append(roc_auc)
		plt.plot(fpr, tpr, lw=1, label='ROC curve for fold {0}, AUC:{1:0.2f}'''.format((i+1),roc_auc))
        
		i+=1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title("Medium and Environmental Perbutations")
	plt.legend(loc='lower right')
	plt.show()



df = pd.read_excel('ecs171.dataset.xlsx',sep=',',header=0)


dataset=df.values

Xinput = dataset[:,6:4501]
Youtput = dataset[:,5]


alphas_lasso = [1e-5,1e-3]
np.set_printoptions(threshold=sys.maxsize)


Yvalues = []


med = dataset[:,2]
for i in range(len(med)):
	if med[i] == 'MD001':
		med[i] = 1
	elif med[i] == 'MD002':
		med[i] = 2
	elif med[i] == 'MD003':
		med[i] = 3
	elif med[i] == 'MD004':
		med[i] = 4
	elif med[i] == 'MD005':
		med[i] = 5
	elif med[i] == 'MD006':
		med[i] = 6
	elif med[i] == 'MD007':
		med[i] = 7
	elif med[i] == 'MD008':
		med[i] = 8
	elif med[i] == 'MD009':
		med[i] = 9
	elif med[i] == 'MD010':
		med[i] = 10
	elif med[i] == 'MD011':
		med[i] = 11
	elif med[i] == 'MD012':
		med[i] = 12
	elif med[i] == 'MD013':
		med[i] = 13
	elif med[i] == 'MD014':
		med[i] = 14
	elif med[i] == 'MD015':
		med[i] = 15
	elif med[i] == 'MD016':
		med[i] = 16
	elif med[i] == 'MD017':
		med[i] = 17
	elif med[i] == 'MD018':
		med[i] = 18


enviro_per = dataset[:,3]
for i in range(len(enviro_per)):
	if enviro_per[i] == 'Indole':
		enviro_per[i] = 1
	elif enviro_per[i] == 'O2-starvation':
		enviro_per[i] = 2
	elif enviro_per[i] == 'RP-overexpress':
		enviro_per[i] = 3
	elif enviro_per[i] == 'antibacterial':
		enviro_per[i] = 4
	elif enviro_per[i] == 'carbon-limitation':
		enviro_per[i] = 5
	elif enviro_per[i] == 'dna-damage':
		enviro_per[i] = 6
	elif enviro_per[i] == 'zinc-limitation':
		enviro_per[i] = 7
	elif enviro_per[i] == 'none':
		enviro_per[i] = 8

combined = med + enviro_per



Yvalues.append(combined)

estimator = SVR(kernel='linear')
selector = RFE(estimator, 50, step = 50) #create new X input with selector
selector.fit(Xinput,Youtput)
Xinput = selector.transform(Xinput)

print((Yvalues))
for i in range(len(Yvalues)):
	ROCCurve(Xinput,Yvalues[i], i)

