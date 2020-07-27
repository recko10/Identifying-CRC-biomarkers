import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/German')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

X_german = dfList[0]
X_chinese = dfList[1]

#Select for only German samples
for index in X_german.index.tolist():
	if 'CCIS' in index:
		X_german = X_german.drop(index, axis=0)

print(X_german)
print(X_chinese)

#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Fix the scrambled IDs issue
idToTarget = {}
for sample in chineseDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	idToTarget[sample] = chineseDf.at[3, sample]

Y_chinese = []
for index in X_chinese.index.tolist():
	Y_chinese.append(idToTarget[index])

#Preprocess German targets
germanDf = pd.read_csv('data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', sep='\t')

#Select for only German samples
for header in germanDf.columns.tolist():
	if 'CCIS' in header:
		germanDf = germanDf.drop(header, axis=1)

#Fix the scrambled IDs issue
idToTarget = {}
for sample in germanDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	idToTarget[sample] = germanDf.at[3, sample]

Y_german = []
for index in X_german.index.tolist():
	Y_german.append(idToTarget[index])

#Classifier
ml = ML()
ml.logisticRegeression(X_chinese, X_german, Y_chinese, Y_german)


