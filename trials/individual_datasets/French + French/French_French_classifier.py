import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/French')

dfList = preprocess.standardPreprocess('data/filedump', keepFiles=False)

X_french = dfList[0]

#Select for only french samples
for index in X_french.index.tolist():
	if 'CCIS' not in index:
		X_french = X_french.drop(index, axis=0)

print(X_french)

#Preprocess targets
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

#Cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X_french, Y_french, test_size = 0.33)

#Classifier
ml = ML()
ml.randomForest(X_train, X_test, Y_train, Y_test)


