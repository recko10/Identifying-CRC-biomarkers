import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *
from sklearn.model_selection import train_test_split

#Preprocess features

preprocess.decompose(path='data/FengQ_austrian.tsv', out='data/filedump/CMD/Austrian')
preprocess.decompose(path='data/ThomasAM_italian.tsv', out='data/filedump/CMD/Italian')
preprocess.decompose(path='data/Yu_china.tsv', out='data/filedump/CMD/Chinese')
preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/CMD/French_German')

dfList = preprocess.standardPreprocess('data/filedump')

X_austrian = dfList[0]
X_italian = dfList[1]
X_chinese = dfList[2]
X_french_german = dfList[3]
X_japanese = dfList[4]

print(X_austrian)
print(X_italian)
print(X_chinese)
print(X_french_german)
print(X_japanese)

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

japID = idToTarget

###Preprocess Austrian

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

ausID = idToTarget

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

chiID = idToTarget
###Preprocess Italian

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

itaID = idToTarget

##Preprocess just French
X_french = X_french_german

#Select for only french samples
for index in X_french.index.tolist():
	if 'CCIS' not in index:
		X_french = X_french.drop(index, axis=0)

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

freID = idToTarget

##Preprocess just German
X_german = X_french_german

#Select for only German samples
for index in X_german.index.tolist():
	if 'CCIS' in index:
		X_german = X_german.drop(index, axis=0)

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

gerID = idToTarget

####MISC PREPROCESSING METHODS BELOW

# ###Preprocess French and German
# #Preprocess French and German targets
# french_germanDf = pd.read_csv('data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', sep='\t')

# #Fix the scrambled IDs issue
# idToTarget = {}
# for sample in french_germanDf.columns.tolist():
# 	if sample == 'Unnamed: 0':
# 		continue
# 	#Remove all unrelated targets and their corresponding samples
# 	if french_germanDf.at[3, sample] != 'CRC' and french_germanDf.at[3, sample] != 'control':
# 		X_french_german = X_french_german.drop(sample, axis=0)
# 		continue
# 	idToTarget[sample] = french_germanDf.at[3, sample]

# Y_french_german = []
# for index in X_french_german.index.tolist():
# 	Y_french_german.append(idToTarget[index])


# ###Preprocess geography + disease PCA
# for index in range(len(Y_austrian)):
# 	Y_austrian[index] = Y_austrian[index] + ' Austrian'

# for index in range(len(Y_italian)):
# 	Y_italian[index] = Y_italian[index] + ' Italian'

# for index in range(len(Y_french_german)):
# 	Y_french_german[index] = Y_french_german[index] + ' French or German'

# for index in range(len(Y_chinese)):
# 	Y_chinese[index] = Y_chinese[index] + ' Chinese'

# for index in range(len(Y_japanese)):
# 	Y_japanese[index] = Y_japanese[index] + ' Japanese'

# ###Preprocess geography PCA
# for index in range(len(Y_austrian)):
# 	Y_austrian[index] = 'Austrian'

# for index in range(len(Y_italian)):
# 	Y_italian[index] = 'Italian'

# for index in range(len(Y_french)):
# 	Y_french[index] = 'French'

# for index in range(len(Y_german)):
# 	Y_german[index] = 'German'

# for index in range(len(Y_chinese)):
# 	Y_chinese[index] = 'Chinese'

# for index in range(len(Y_japanese)):
# 	Y_japanese[index] = 'Japanese'

# ##Preprocess geography only controls PCA
# for index in X_austrian.index.tolist():
# 	if ausID[index] == 'CRC':
# 		X_austrian = X_austrian.drop(index, axis=0)

# Y_austrian = ['Austrian' for x in Y_austrian if x != 'CRC']

# for index in X_italian.index.tolist():
# 	if itaID[index] == 'CRC':
# 		X_italian = X_italian.drop(index, axis=0)

# Y_italian = ['Italian' for x in Y_italian if x != 'CRC']

# for index in X_french.index.tolist():
# 	if freID[index] == 'CRC':
# 		X_french = X_french.drop(index, axis=0)

# Y_french = ['French' for x in Y_french if x != 'CRC']

# for index in X_german.index.tolist():
# 	if gerID[index] == 'CRC':
# 		X_german = X_german.drop(index, axis=0)

# Y_german = ['German' for x in Y_german if x != 'CRC']

# for index in X_chinese.index.tolist():
# 	if chiID[index] == 'CRC':
# 		X_chinese = X_chinese.drop(index, axis=0)

# Y_chinese = ['Chinese' for x in Y_chinese if x != 'CRC']

# for index in X_japanese.index.tolist():
# 	if japID[index] == 'CRC':
# 		X_japanese = X_japanese.drop(index, axis=0)

# Y_japanese = ['Japanese' for x in Y_japanese if x != 'CRC']

# ##Preprocess geography only CRC PCA
# for index in X_austrian.index.tolist():
# 	if ausID[index] == 'control':
# 		X_austrian = X_austrian.drop(index, axis=0)

# Y_austrian = ['Austrian' for x in Y_austrian if x != 'control']

# for index in X_italian.index.tolist():
# 	if itaID[index] == 'control':
# 		X_italian = X_italian.drop(index, axis=0)

# Y_italian = ['Italian' for x in Y_italian if x != 'control']

# for index in X_french.index.tolist():
# 	if freID[index] == 'control':
# 		X_french = X_french.drop(index, axis=0)

# Y_french = ['French' for x in Y_french if x != 'control']

# for index in X_german.index.tolist():
# 	if gerID[index] == 'control':
# 		X_german = X_german.drop(index, axis=0)

# Y_german = ['German' for x in Y_german if x != 'control']

# for index in X_chinese.index.tolist():
# 	if chiID[index] == 'control':
# 		X_chinese = X_chinese.drop(index, axis=0)

# Y_chinese = ['Chinese' for x in Y_chinese if x != 'control']

# for index in X_japanese.index.tolist():
# 	if japID[index] == 'Healthy control':
# 		X_japanese = X_japanese.drop(index, axis=0)

# Y_japanese = ['Japanese' for x in Y_japanese if x != 'control']


##ML

# #Create European and East Asian dataset
# X_european_eastasian = X_austrian.append([X_italian, X_chinese, X_french, X_german, X_japanese])
# Y_european_eastasian = Y_austrian + Y_italian + Y_chinese + Y_french + Y_german + Y_japanese

# #Create East Asian dataset
X_eastasian = X_chinese.append([X_japanese])
Y_eastasian = Y_chinese + Y_japanese
Y_eastasian = ['East Asian' for x in Y_eastasian]

# #Create European dataset
X_european = X_austrian.append([X_french, X_german, X_italian])
Y_european = Y_austrian + Y_french + Y_german + Y_italian
Y_european = ['European' for x in Y_european]

#Train test split
#X_train, X_test, Y_train, Y_test = train_test_split(X_european_eastasian, Y_european_eastasian, test_size=0.33)

###Classifiers
ml = ML()
#ml.randomForest(X_train, X_test, Y_train, Y_test)
#ml.logisticRegression(X_train, X_test, Y_train, Y_test)
#ml.randomForest(X_eastasian, X_french, Y_eastasian, Y_french)
#ml.logisticRegression(X_eastasian, X_french, Y_eastasian, Y_french)

#Scree plot
#ml.scree(X_eastasian.append(X_european))

#Create diagonal correlation matrix
#ml.correlationMatrix(X_european_eastasian, Y_european_eastasian)

#PCA
#ml.pca(X_eastasian.append(X_european), Y_eastasian + Y_european)

#Geography + disease PCA
#ml.pca(X_european_eastasian, Y_european_eastasian, targets=['control Japanese', 'CRC Japanese', 'control Chinese', 'CRC Chinese', 'control Italian', 'CRC Italian', 'control Austrian', 'CRC Austrian','control French or German', 'CRC French or German'], colors=['r','b','g','y', 'k','c','m','#894850', '#33FFA8', '#F29A12'])

#Geography PCA
#ml.pca(X_eastasian.append(X_european), Y_eastasian+Y_european, targets=['East Asian', 'European'], colors=['r','g'])

#TSNE
#ml.tsne(X_eastasian.append(X_european), Y_eastasian + Y_european)

#Geography + disease TSNE
#ml.tsne(X_european_eastasian, Y_european_eastasian, targets=['control Japanese', 'CRC Japanese', 'control Chinese', 'CRC Chinese', 'control Italian', 'CRC Italian', 'control Austrian', 'CRC Austrian','control French or German', 'CRC French or German'], colors=['r','b','g','y', 'k','c','m','#894850', '#33FFA8', '#F29A12'])

#Geography TSNE
#ml.tsne(X_japanese.append(X_chinese), Y_japanese+Y_chinese, targets=['Japanese', 'Chinese'], colors=['r','g'])










# ###Create weights table
# japaneseTopFeatures, japaneseRankedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_japanese, Y_japanese), X_japanese, Y_japanese)
# austrianTopFeatures, austrianRankedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_austrian, Y_austrian), X_austrian, Y_austrian)
# italianTopFeatures, italianRankedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_italian, Y_italian), X_italian, Y_italian)
# chineseTopFeatures, chineseRankedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_chinese, Y_chinese), X_chinese, Y_chinese)
# frenchTopFeatures, frenchRankedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_french, Y_french), X_french, Y_french)
# germanTopFeatures, germanRankedFeatures = ml.selectFromModel(RandomForestClassifier().fit(X_german, Y_german), X_german, Y_german)


# #Bacteria: list of weights
# selectedRankedFeatures = {}
# expectedLoop = 1


# #Create bacterial superset in dictionary
# for item in japaneseTopFeatures + austrianTopFeatures + italianTopFeatures + chineseTopFeatures + frenchTopFeatures + germanTopFeatures:
# 	if item not in selectedRankedFeatures:
# 		selectedRankedFeatures[item] = []


# #Append to lists
# for item in japaneseTopFeatures:
# 	if item in selectedRankedFeatures:
# 		selectedRankedFeatures[item].append(japaneseRankedFeatures[X_japanese.columns.tolist().index(item)])

# for bacteria in selectedRankedFeatures:
# 	if len(selectedRankedFeatures[bacteria]) < expectedLoop:
# 		selectedRankedFeatures[bacteria].append('NA')

# for item in chineseTopFeatures:
# 	if item in selectedRankedFeatures:
# 		selectedRankedFeatures[item].append(chineseRankedFeatures[X_chinese.columns.tolist().index(item)])

# expectedLoop+=1
# for bacteria in selectedRankedFeatures:
# 	if len(selectedRankedFeatures[bacteria]) < expectedLoop:
# 		selectedRankedFeatures[bacteria].append('NA')

# for item in austrianTopFeatures:
# 	if item in selectedRankedFeatures:
# 		selectedRankedFeatures[item].append(austrianRankedFeatures[X_austrian.columns.tolist().index(item)])

# expectedLoop+=1
# for bacteria in selectedRankedFeatures:
# 	if len(selectedRankedFeatures[bacteria]) < expectedLoop:
# 		selectedRankedFeatures[bacteria].append('NA')


# for item in italianTopFeatures:
# 	if item in selectedRankedFeatures:
# 		selectedRankedFeatures[item].append(italianRankedFeatures[X_italian.columns.tolist().index(item)])

# expectedLoop+=1
# for bacteria in selectedRankedFeatures:
# 	if len(selectedRankedFeatures[bacteria]) < expectedLoop:
# 		selectedRankedFeatures[bacteria].append('NA')

# for item in frenchTopFeatures:
# 	if item in selectedRankedFeatures:
# 		selectedRankedFeatures[item].append(frenchRankedFeatures[X_french_german.columns.tolist().index(item)])

# expectedLoop+=1
# for bacteria in selectedRankedFeatures:
# 	if len(selectedRankedFeatures[bacteria]) < expectedLoop:
# 		selectedRankedFeatures[bacteria].append('NA')


# for item in germanTopFeatures:
# 	if item in selectedRankedFeatures:
# 		selectedRankedFeatures[item].append(germanRankedFeatures[X_german.columns.tolist().index(item)])

# expectedLoop+=1
# for bacteria in selectedRankedFeatures:
# 	if len(selectedRankedFeatures[bacteria]) < expectedLoop:
# 		selectedRankedFeatures[bacteria].append('NA')

# #Create table
# weightsDf = pd.DataFrame(selectedRankedFeatures)
# weightsDf.index = ['Japanese', 'Austrian', 'Italian', 'Chinese', 'French', 'German']
# weightsDf =  weightsDf.T
# print(weightsDf)
# weightsDf.to_csv('weights.csv')
