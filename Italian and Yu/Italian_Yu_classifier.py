import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out= 'data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

#Declare feature dfs
X_train = dfList[0]
X_test = dfList[1]


#Preprocess Italian targets
italianDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Mark unnecessary columns and append to targets list
Y_train = [x for x in italianDf.iloc[3, :].tolist()]
Y_train.pop(0)

#Clean up Italian targets
for index in range(len(Y_train)):
	if Y_train[index] == 'adenoma':
		X_train = X_train.drop(italianDf.columns.tolist()[index], axis=0)
		continue

Y_train = [x for x in Y_train if x != 'adenoma']


#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Extract Chinese targets
Y_test = chineseDf.iloc[3, :].tolist()
Y_test.pop(0)

#Create logistic classifier to predict on Yu
ml = ML()
#ml.logisticRegeression(X_train,X_test,Y_train,Y_test)

#Create logistic classifier to predict on Italian
ml.logisticRegeression(X_test, X_train,Y_test,Y_train)



