import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()
preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/decomp')
dfList = preprocess.standardPreprocess('data/decomp')
X = dfList[0]

#Preprocess targets
thomasDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Mark unnecessary columns and append to targets list
Y = [x for x in thomasDf.iloc[3, :].tolist()]
Y.pop(0)

#Clean up targets
for index in range(len(Y)):
	if Y[index] == 'adenoma':
		X = X.drop(thomasDf.columns.tolist()[index], axis=0)
		continue
	if Y[index] == 'control':
		Y[index] = 'CTR'

Y = [x for x in Y if x != 'adenoma']


#PCA
ml = ML()
ml.pca(X, Y)