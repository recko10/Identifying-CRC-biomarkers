import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/VogtmannE_2016.metaphlan_bugs_list.stool.tsv', out='data/filedump/USA')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

X_usa = dfList[0]
X_chinese = dfList[1]

print(X_usa)
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


#Preprocess USA targets
usaDf = pd.read_csv('data/VogtmannE_2016.metaphlan_bugs_list.stool.tsv', sep='\t')

#Fix the scrambled IDs issue
idToTarget = {}
for sample in usaDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	#Remove all unrelated targets and their corresponding samples
	if usaDf.at[3, sample] != 'CRC' and usaDf.at[3, sample] != 'control':
		X_usa = X_usa.drop(sample, axis=0)
		continue
	idToTarget[sample] = usaDf.at[3, sample]

Y_usa = []
for index in X_usa.index.tolist():
	Y_usa.append(idToTarget[index])

#Classifier
ml = ML()
#ml.logisticRegeression(X_chinese, X_usa, Y_chinese, Y_usa)
ml.logisticRegeression(X_usa, X_chinese, Y_usa, Y_chinese)

