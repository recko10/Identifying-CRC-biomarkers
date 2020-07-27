import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split


#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/FengQ_austrian.tsv', out='data/filedump/Austrian')
preprocess.decompose(path='data/ThomasAM_italian.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/Vogtmann_USA.tsv', out='data/filedump/USA')
preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/French_German')

dfList = preprocess.standardPreprocess('data/filedump')

X_usa = dfList[0]
X_austrian = dfList[1]
X_italian = dfList[2]
X_french_german = dfList[3]

print(X_usa)
print(X_austrian)
print(X_italian)
print(X_french_german)


# #Preprocess Austrian targets
# austrianDf = pd.read_csv('data/FengQ_austrian.tsv', sep='\t')

# #Fix the scrambled IDs issue
# idToTarget = {}
# for sample in austrianDf.columns.tolist():
# 	if sample == 'Unnamed: 0':
# 		continue
# 	#Remove all unrelated targets and their corresponding samples
# 	if austrianDf.at[3, sample] != 'CRC' and austrianDf.at[3, sample] != 'control':
# 		X_austrian = X_austrian.drop(sample, axis=0)
# 		continue
# 	idToTarget[sample] = austrianDf.at[3, sample]

# Y_austrian = []
# for index in X_austrian.index.tolist():
# 	Y_austrian.append(idToTarget[index])

# #Cross validation
# X_train, X_test, Y_train, Y_test = train_test_split(X_austrian, Y_austrian, test_size = 0.33)


# #Classifier
# ml = ML()
# ml.logisticRegeression(X_train, X_test, Y_train, Y_test)


