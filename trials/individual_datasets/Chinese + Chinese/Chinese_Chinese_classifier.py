import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features

preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump', keepFiles=False, onlyVirus=True)

X_chinese = dfList[0]

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

#Cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X_chinese, Y_chinese, test_size = 0.33)

#Classifier
ml = ML()
#ml.randomForest(X_train, X_test, Y_train, Y_test)
#ml.logisticRegression(X_train, X_test, Y_train, Y_test)

#ml.pca(X_chinese, Y_chinese)



