import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/French')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

X_french = dfList[0]
X_chinese = dfList[1]

#Select for only french samples
for index in X_french.index.tolist():
	if 'CCIS' not in index:
		X_french = X_french.drop(index, axis=0)

print(X_french)
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



#Preprocess french targets
frenchDf = pd.read_csv('data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', sep='\t')

#Select for only french samples
for header in frenchDf.columns.tolist():
	if 'CCIS' not in header:
		frenchDf = frenchDf.drop(header, axis=1)

#Fix the scrambled IDs issue
idToTarget = {}
for sample in frenchDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	#Remove all unrelated targets and their corresponding samples
	if frenchDf.at[3, sample] != 'CRC' and frenchDf.at[3, sample] != 'control':
		X_french = X_french.drop(sample, axis=0)
		continue
	idToTarget[sample] = frenchDf.at[3, sample]

Y_french = []
for index in X_french.index.tolist():
	Y_french.append(idToTarget[index])

#Classifier
ml = ML()
ml.logisticRegeression(X_chinese, X_french, Y_chinese, Y_french)


