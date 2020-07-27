import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/Vogtmann_USA.tsv', out='data/filedump/USA')

dfList = preprocess.standardPreprocess('data/filedump')

X_usa = dfList[0]

print(X_usa)

#Preprocess USA targets
usaDf = pd.read_csv('data/Vogtmann_USA.tsv', sep='\t')

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

#Cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X_usa, Y_usa, test_size = 0.33)

#Classifier
ml = ML()
ml.logisticRegeression(X_train, X_test, Y_train, Y_test)


