import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/FengQ_austrian.tsv', out='data/filedump/Austrian')
preprocess.decompose(path='data/ThomasAM_italian.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/French_German')
preprocess.decompose(path='data/Yu_china.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump', keepFiles=False)

X_austrian = dfList[0]
X_italian = dfList[1]
X_chinese = dfList[2]
X_french_german = dfList[3]

print(X_austrian)
print(X_italian)
print(X_chinese)
print(X_french_german)

#Preprocess Austrian targets
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


#Preprocess Italian targets
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


#Preprocess French and German targets
french_germanDf = pd.read_csv('data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', sep='\t')

#Fix the scrambled IDs issue
idToTarget = {}
for sample in french_germanDf.columns.tolist():
	if sample == 'Unnamed: 0':
		continue
	#Remove all unrelated targets and their corresponding samples
	if french_germanDf.at[3, sample] != 'CRC' and french_germanDf.at[3, sample] != 'control':
		X_french_german = X_french_german.drop(sample, axis=0)
		continue
	idToTarget[sample] = french_germanDf.at[3, sample]

Y_french_german = []
for index in X_french_german.index.tolist():
	Y_french_german.append(idToTarget[index])


#Preprocess Chinese targets
chineseDf = pd.read_csv('data/Yu_china.tsv', sep='\t')

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


# ##PCA with disease + geography preprocessing
# for index in range(len(Y_italian)):
# 	Y_italian[index] = Y_italian[index] + ' Italian'

# for index in range(len(Y_french_german)):
# 	Y_french_german[index] = Y_french_german[index] + ' French_German'

# for index in range(len(Y_austrian)):
# 	Y_austrian[index] = Y_austrian[index] + ' Austrian'


#Combine all European datasets
X_european = X_austrian.append([X_italian, X_french_german])
Y_european = Y_austrian + Y_italian + Y_french_german


#LOSO Austrian
# X_european = X_italian.append([X_french_german])
# Y_european = Y_italian + Y_french_german

#LOSO Italian
# X_european = X_austrian.append([X_french_german])
# Y_european = Y_austrian + Y_french_german

#LOSO French and German
# X_european = X_austrian.append([X_italian])
# Y_european = Y_austrian + Y_italian

#Cross validation
#X_train, X_test, Y_train, Y_test = train_test_split(X_european, Y_european, test_size = 0.33)

#Classifier
ml = ML()
#ml.randomForest(X_train, X_test, Y_train, Y_test)
#ml.randomForest(X_european, X_french_german, Y_european, Y_french_german)
#l.logisticRegeression(X_european, X_french_german, Y_european, Y_french_german)

#Feature selection
#selectedFeatures = ml.selectFromModel(X_european, Y_european)

#Create and plot a diagonal correlation matrix
#ml.correlationMatrix(X_european, Y_european)

#PCA
#ml.pca(X_european, Y_european, targets=['CRC Italian', 'control Italian', 'CRC French_German', 'control French_German', 'CRC Austrian', 'control Austrian'], colors=['r','b','g','y','c','m'])
ml.pca(X_european, Y_european)

