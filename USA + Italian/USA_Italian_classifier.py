import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/VogtmannE_2016.metaphlan_bugs_list.stool.tsv', out='data/filedump/USA')

dfList = preprocess.standardPreprocess('data/filedump')

X_usa = dfList[0]
X_italian = dfList[1]

print(X_italian)
print(X_usa)

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
#ml.logisticRegeression(X_usa, X_italian, Y_usa, Y_italian)
ml.logisticRegeression(X_italian, X_usa, Y_italian, Y_usa)


