import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/FengQ_austrian.tsv', out='data/filedump/Austrian')
preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/German')

dfList = preprocess.standardPreprocess('data/filedump', keepFiles=False)

X_german = dfList[0]
X_austrian = dfList[1]

#Select for only German samples
for index in X_german.index.tolist():
	if 'CCIS' in index:
		X_german = X_german.drop(index, axis=0)

print(X_german)
print(X_austrian)

#Preprocess targets
austrianDf = pd.read_csv('data/FengQ_austrian.tsv', sep='\t')

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


#Preprocess targets
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

	#Remove all unrelated targets and their corresponding samples
	if germanDf.at[3, sample] != 'CRC' and germanDf.at[3, sample] != 'control':
		X_german = X_german.drop(sample, axis=0)
		continue
	idToTarget[sample] = germanDf.at[3, sample]

Y_german = []
for index in X_german.index.tolist():
	Y_german.append(idToTarget[index])


#Classifier
ml = ML()
#ml.randomForest(X_german, X_austrian, Y_german, Y_austrian)
ml.randomForest(X_austrian, X_german, Y_austrian, Y_german)


