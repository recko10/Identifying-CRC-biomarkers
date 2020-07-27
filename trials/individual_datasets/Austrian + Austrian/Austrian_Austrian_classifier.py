import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/FengQ_austrian.tsv', out='data/filedump/Austrian')

dfList = preprocess.standardPreprocess('data/filedump', keepFiles=False)

X_austrian = dfList[0]

print(X_austrian)

#Preprocess Chinese targets
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

#Cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X_austrian, Y_austrian, test_size = 0.33)

#Classifier
ml = ML()
ml.randomForest(X_train, X_test, Y_train, Y_test)


