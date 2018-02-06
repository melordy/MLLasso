
# coding: utf-8

# In[2]:


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


from itertools import cycle

import warnings
warnings.filterwarnings("ignore")



def PRCurve(Xinput, Yinput, graphType):
	random_state = np.random.RandomState(0)
	Yinput = Yinput.astype(float)
	
	titles = ['Strain Type', 'Medium Type','Environmental Perturbation','Gene Perturbation']
	
	
	if graphType == 0:
		n_classes = 10
		title = titles[0]
	if graphType == 1:
		n_classes = 18
		title = titles[1]
	if graphType == 2:
		n_classes = 8
		title = titles[2]
	if graphType == 3:
		n_classes = 12
		title=titles[3]

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

		pca = PCA(n_components = 3)
		pca.fit(Xinput[train])
		X_pca_train = pca.transform(Xinput[train])
		X_pca_test = pca.transform(Xinput[test])

		y_score = classifier.fit(X_pca_train, Yinput[train]).decision_function(X_pca_test)
		flat_list = [item for sublist in y_score for item in sublist]

		precision, recall, _ = precision_recall_curve(Yinput[test][1], flat_list[:n_classes])
		pr_auc = auc(recall,precision)
		plt.plot(recall, precision, lw=1, label='PR curve for fold {0}, AUPRC:{1:0.2f}'''.format((i+1),pr_auc))

		tprs.append(interp(mean_fpr, precision, recall))


		i+=1 #fold count


	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
     label=r'Mean PR (AUPRC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
       lw=2, alpha=.8)

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(titles[graphType])
	plt.legend(loc='lower right')
	plt.show()


df = pd.read_excel('ecs171.dataset.xlsx',sep=',',header=0)


dataset=df.values

Xinput = dataset[:,6:4501]
Youtput = dataset[:,5]


alphas_lasso = [1e-5,1e-3]
np.set_printoptions(threshold=sys.maxsize)


Yvalues = []


strain_type = dataset[:,1]
for i in range(len(strain_type)):
	if strain_type[i] == 'BW25113':
		strain_type[i] = 1
	elif strain_type[i] == 'CG2':
		strain_type[i] = 2
	elif strain_type[i] == 'DH5alpha':
		strain_type[i] = 3
	elif strain_type[i] == 'MG1655':
		strain_type[i] = 4
	elif strain_type[i] == 'P2':
		strain_type[i] = 5
	elif strain_type[i] == 'P4X':
		strain_type[i] = 6
	elif strain_type[i] == 'W3110':
		strain_type[i] = 7
	elif strain_type[i] == 'rpoA14':
		strain_type[i] = 8
	elif strain_type[i] == 'rpoA27':
		strain_type[i] = 9
	elif strain_type[i] == 'rpoD3':
		strain_type[i] = 10

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


gene_per = dataset[:,4]
for i in range(len(gene_per)):
	if gene_per[i] == 'appY_KO':
		gene_per[i] = 1
	elif gene_per[i] == 'arcA_KO':
		gene_per[i] = 2
	elif gene_per[i] == 'argR_KO':
		gene_per[i] = 3
	elif gene_per[i] == 'cya_KO':
		gene_per[i] = 4
	elif gene_per[i] == 'fis_OE':
		gene_per[i] = 5
	elif gene_per[i] == 'fnr_KO':
		gene_per[i] = 6
	elif gene_per[i] == 'frdC_KO':
		gene_per[i] = 7
	elif gene_per[i] == 'na_WT':
		gene_per[i] = 8
	elif gene_per[i] == 'oxyR_KO':
		gene_per[i] = 9
	elif gene_per[i] == 'rpoS_KO':
		gene_per[i] = 10
	elif gene_per[i] == 'soxS_KO':
		gene_per[i] = 11
	elif gene_per[i] == 'tnaA_KO':
		gene_per[i] = 12
        
        
Yvalues.append(strain_type)
Yvalues.append(med)
Yvalues.append(enviro_per)
Yvalues.append(gene_per)

estimator = SVR(kernel='linear')
selector = RFE(estimator, 50, step = 50) #create new X input with selector
selector.fit(Xinput,Youtput)
Xinput = selector.transform(Xinput)

for i in range(len(Yvalues)):
	PRCurve(Xinput,Yvalues[i],i)

