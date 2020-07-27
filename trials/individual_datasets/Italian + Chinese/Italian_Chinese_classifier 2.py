import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

X_italian = dfList[0]
X_chinese = dfList[1]

print(X_italian)
print(X_chinese)

#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Fix the scrambled IDs issue
idToTarget = {}
for sample in chineseDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	#Remove all unrelated targets and their corresponding samples
	if chineseDf.at[3, sample] != 'CRC' and chineseDf.at[3, sample] != 'control':
		X_chinese = X_chinese.drop(sample, axis=0)
		continue
	idToTarget[sample] = chineseDf.at[3, sample]

Y_chinese = []
for index in X_chinese.index.tolist():
	Y_chinese.append(idToTarget[index])

#Preprocess Italian targets
italianDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Fix the scrambled IDs issue
idToTarget = {}
for sample in italianDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	#Remove all unrelated targets and their corresponding samples
	if italianDf.at[3, sample] != 'CRC' and italianDf.at[3, sample] != 'control':
		X_italian = X_italian.drop(sample, axis=0)
		continue
	idToTarget[sample] = italianDf.at[3, sample]

Y_italian = []
for index in X_italian.index.tolist():
	Y_italian.append(idToTarget[index])

#Classifier
ml = ML()
#ml.logisticRegeression(X_chinese, X_italian, Y_chinese, Y_italian)
#ml.logisticRegeression(X_italian, X_chinese, Y_italian, Y_chinese)
