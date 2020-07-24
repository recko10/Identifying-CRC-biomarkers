import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/FengQ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Austrian')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

X_austrian = dfList[0]
X_chinese = dfList[1]

X_chinese.to_csv('X_chinese.csv')


#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

chineseDf.to_csv('chineseDf.csv')

#Extract Chinese targets
Y_chinese = chineseDf.iloc[3, :].tolist()
Y_chinese.pop(0)

#Preprocess Austrian targets
austrianDf = pd.read_csv('data/FengQ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

austrianDf = austrianDf.drop('Unnamed: 0', axis=1)
Y_austrian = austrianDf.iloc[3, :].tolist()

for index in range(len(Y_austrian)):
	if Y_austrian[index] != 'CRC' and Y_austrian[index] != 'control':
		X_austrian = X_austrian.drop(austrianDf.columns.tolist()[index], axis=0)
		continue

Y_austrian = [x for x in Y_austrian if x == 'CRC' or x == 'control']

#Classifier
ml = ML()
ml.logisticRegeression(X_chinese, X_austrian, Y_chinese, Y_austrian)

#PCA
#ml.pca(X_austrian, Y_austrian)

