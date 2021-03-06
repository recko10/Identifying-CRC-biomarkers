import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Preprocess features
preprocess.decompose(path='data/FengQ_austrian.tsv', out='data/filedump/Austrian')
preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/French_German')
preprocess.decompose(path='data/Yu_china.tsv', out='data/filedump/Chinese')
preprocess.decompose(path='data/ThomasAM_italian.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/Vogtmann_USA.tsv', out='data/filedump/USA')


dfList = preprocess.standardPreprocess('data/filedump', onlyVirus=True)


X_usa = dfList[0]
X_austrian = dfList[1]
X_italian = dfList[2]
X_chinese = dfList[3]
X_french_german = dfList[4]
X_japanese = dfList[5]

print(X_austrian)
print(X_chinese)
print(X_french_german)
print(X_italian)
print(X_usa)
print(X_japanese)

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

#Preprocess Japanese targets
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




# ##PCA with disease + geography preprocessing
# for index in range(len(Y_italian)):
# 	Y_italian[index] = Y_italian[index] + ' Italian'

# for index in range(len(Y_french_german)):
# 	Y_french_german[index] = Y_french_german[index] + ' French_German'

# for index in range(len(Y_austrian)):
# 	Y_austrian[index] = Y_austrian[index] + ' Austrian'


#Combine all datasets
X_virome = X_austrian.append([X_chinese, X_french_german, X_italian, X_usa, X_japanese])
Y_virome = Y_austrian + Y_chinese + Y_french_german + Y_italian + Y_usa + Y_japanese

#LOSO American
# X_virome = X_austrian.append([X_chinese, X_french_german, X_italian, X_japanese])
# Y_virome = Y_austrian + Y_chinese + Y_french_german + Y_italian + Y_japanese

#LOSO Austrian
# X_virome = X_chinese.append([X_french_german, X_italian, X_usa, X_japanese])
# Y_virome = Y_chinese + Y_french_german + Y_italian + Y_usa + Y_japanese

#LOSO Italian
# X_virome = X_austrian.append([X_chinese, X_french_german, X_usa, X_japanese])
# Y_virome = Y_austrian + Y_chinese + Y_french_german + Y_usa + Y_japanese

#LOSO French and German
# X_virome = X_austrian.append([X_chinese, X_italian, X_usa, X_japanese])
# Y_virome = Y_austrian + Y_chinese + Y_italian + Y_usa + Y_japanese

#Cross validation
#_train, X_test, Y_train, Y_test = train_test_split(X_virome, Y_virome, test_size = 0.33)

#Classifier
ml = ML()
#ml.randomForest(X_train, X_test, Y_train, Y_test)
#ml.logisticRegression(X_virome, X_italian, Y_virome, Y_italian)

#Feature selection
#selectedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_virome, Y_virome), X_virome, Y_virome)

#Create and plot a diagonal correlation matrix
#ml.correlationMatrix(X_virome, Y_virome)

#PCA
#ml.pca(X_european, Y_european, targets=['CRC Italian', 'control Italian', 'CRC French_German', 'control French_German', 'CRC Austrian', 'control Austrian'], colors=['r','b','g','y','c','m'])
#ml.pca(X_virome, Y_virome)

#TSNE
#ml.tsne(X_virome, Y_virome)

