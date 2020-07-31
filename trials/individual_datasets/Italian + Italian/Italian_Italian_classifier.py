import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features
preprocess.decompose(path='data/ThomasAM_italian.tsv', out='data/filedump/Italian')

dfList = preprocess.standardPreprocess('data/filedump', keepFiles=False)

X_italian = dfList[0]

print(X_italian)

#Preprocess Chinese targets
italianDf = pd.read_csv('data/ThomasAM_italian.tsv', sep='\t')

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

#Cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X_italian, Y_italian, test_size = 0.33)

#Classifier
ml = ML()
ml.randomForest(X_train, X_test, Y_train, Y_test)


