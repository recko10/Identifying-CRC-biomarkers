import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features
preprocess = preprocess()

dfList = preprocess.standardPreprocess('data/filedump')

X_japanese = dfList[0]
X_austrian = dfList[1]
X_chinese = dfList[2]

###Preprocess Japanese

#Load in metadata
japaneseMetadataDf = pd.read_csv('data/japanese_metadata.csv')

#Create a dictionary with key:value = sample:run
sampleToRun = {}
index=0
for sample in japaneseMetadataDf['BioSample']:
	sampleToRun[sample] = japaneseMetadataDf.at[index, 'Run']
	index+=1

#Create new indices for features
newIndexFeatures=[]
for index in X_japanese.index.tolist():
	if index in sampleToRun:
		newIndexFeatures.append(sampleToRun[index])

#Change indicies
X_japanese.index = newIndexFeatures

#Create new indices for metadata
newIndexMetadata=[]
for sample in japaneseMetadataDf['BioSample'].tolist():
	if sample in sampleToRun:
		newIndexMetadata.append(sampleToRun[sample])

japaneseMetadataDf.index = newIndexMetadata

#Preprocess targets

#Fix the scrambled IDs issue
idToTarget = {}
for sample in japaneseMetadataDf.index.tolist():
	#Remove all unrelated targets and their corresponding samples
	if 'CRC' in japaneseMetadataDf.at[sample,'host_disease_stat']:
		idToTarget[sample] = japaneseMetadataDf.at[sample, 'host_disease_stat'].split(' (', 1)[0]

	if 'Healthy control' in japaneseMetadataDf.at[sample, 'host_disease_stat']:
		idToTarget[sample] = japaneseMetadataDf.at[sample, 'host_disease_stat']

#Create targets list with matching id:target
Y_japanese = []
for index in X_japanese.index.tolist():
	if index in idToTarget:
		Y_japanese.append(idToTarget[index])

#Change all "Healthy control" to "control"
for index in range(len(Y_japanese)):
	if Y_japanese[index] == 'Healthy control':
		Y_japanese[index] = 'control'

###Preprocess Chinese

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




# ##PCA with disease + geography preprocessing
# for index in range(len(Y_japanese)):
# 	Y_japanese[index] = Y_japanese[index] + ' Japanese'

# for index in range(len(Y_chinese)):
# 	Y_chinese[index] = Y_chinese[index] + ' Chinese'


X_eastasian = X_japanese.append(X_chinese)
Y_eastasian = Y_japanese + Y_chinese

print(X_eastasian)
print(Y_eastasian)

#Train test split
#X_train, X_test, Y_train, Y_test = train_test_split(X_eastasian, Y_eastasian, test_size=0.33)

#Classifier
ml = ML()
#ml.randomForest(X_train, X_test, Y_train, Y_test)
#ml.logisticRegeression(X_train, X_test, Y_train, Y_test)
#ml.randomForest(X_eastasian, X_austrian, Y_eastasian, Y_austrian)
#ml.logisticRegeression(X_eastasian, X_austrian, Y_eastasian, Y_austrian)


#Create diagonal correlation matrix
ml.correlationMatrix(X_eastasian, Y_eastasian)

#PCA
#ml.pca(X_eastasian, Y_eastasian, targets=['control Japanese', 'CRC Japanese', 'control Chinese', 'CRC Chinese'], colors=['r','b','g','y'])
#ml.pca(X_eastasian, Y_eastasian)


