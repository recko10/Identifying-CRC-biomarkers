import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out= 'data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

#Declare feature dfs
X_italian = dfList[0]
X_chinese = dfList[1]


#Preprocess Italian targets
italianDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Mark unnecessary columns and append to targets list
Y_italian = [x for x in italianDf.iloc[3, :].tolist()]
Y_italian.pop(0)

#Clean up Italian targets
for index in range(len(Y_italian)):
	if Y_italian[index] == 'adenoma':
		X_italian = X_italian.drop(italianDf.columns.tolist()[index], axis=0)
		continue

Y_italian = [x for x in Y_italian if x != 'adenoma']


#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Extract Chinese targets
Y_chinese = chineseDf.iloc[3, :].tolist()
Y_chinese.pop(0)

#Create logistic classifier to predict on Yu
ml = ML()
ml.logisticRegeression(X_italian,X_chinese,Y_italian,Y_chinese)

#Create logistic classifier to predict on Italian
#ml.logisticRegeression(X_chinese, X_italian,Y_chinese,Y_italian)

#PCA
#ml.pca(X_chinese,Y_chinese)



