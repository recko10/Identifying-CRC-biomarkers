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

print(X_austrian)
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

#Preprocess Austrian targets
austrianDf = pd.read_csv('data/FengQ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Fix the scrambled IDs issue
idToTarget = {}
for sample in austrianDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	#Remove all unrelated targets and their corresponding samples
	if austrianDf.at[3, sample] != 'CRC' and austrianDf.at[3, sample] != 'control':
		X_austrian = X_austrian.drop(sample, axis=0)
		continue
	idToTarget[sample] = austrianDf.at[3, sample]

Y_austrian = []
for index in X_austrian.index.tolist():
	Y_austrian.append(idToTarget[index])

#Classifier
ml = ML()
ml.logisticRegeression(X_chinese, X_austrian, Y_chinese, Y_austrian)


