import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler

#Preprocess data
featuresDf = pd.read_csv('taxonomic_abundances.csv') #Load in df

#indexList = [entry.split('[',1)[0] for entry in featuresDf['Unnamed: 0']] #List of bacteria names 
indexList = [entry for entry in featuresDf['Unnamed: 0']] #List of bacteria names 

featuresDf.index = indexList #Change indices to values in this column
featuresDf = featuresDf.drop(columns='Unnamed: 0') #Drop the column

featuresDf = featuresDf.T #Transpose featuresDf (switch rows and columns and adjust values accordingly)

featuresDf['Experiment'] = ''

#Create df with metadata
targetDf = pd.read_csv('metadata.csv')

#Set the appropriate rows in the Experiment column to be equal to the appropriate rows of the Group column
index = 0
for sample in targetDf['Sample_ID']:
	featuresDf.loc[sample,'Experiment'] = targetDf.loc[index, 'Group'] 
	index+=1

#Remove rows without any metadata
featuresDf = featuresDf[featuresDf.Experiment != '']

#This new dataframe represents featuresDf but with the metadata column in place
finalFeaturesDf = featuresDf

#Remove the metadata column
X = featuresDf.drop(columns = 'Experiment')

#Convert the metadata column into a list of labels
Y = finalFeaturesDf['Experiment'].tolist()

#CONVERT ALL NUMBERS INTO 'yes' or 'no' values indicating the presence of the bacteria (yes is 1 and no is 0)
#This method is a little slow, but it works
def binaryData(X, threshold):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) #Train-test split
	for column in X.columns.tolist():
		for index in range(len(X)):
			if X[column].iloc[index] < threshold:
				X[column].iloc[index] = 0
			else:
				X[column].iloc[index] = 1

def kneighbors(X, Y):
	X = StandardScaler().fit_transform(X) #Scale the data

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) #Train-test split

	#Initialize classifier
	kn = KNeighborsClassifier(n_neighbors=3)
	print(kn)
	kn.fit(X_train, y_train)

	#Predict
	y_pred = kn.predict(X_test)

	print(accuracy_score(y_test,y_pred))
	print(confusion_matrix(y_test,y_pred))

def logisticRegeression(X, Y):
	X = StandardScaler().fit_transform(X) #Scale the data

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) #Train-test split

	#Initialize classifier
	logReg = LogisticRegression(C=10, max_iter=200)
	print(logReg)
	logReg.fit(X_train, y_train)

	#Predict
	y_pred = logReg.predict(X_test)

	print(accuracy_score(y_test,y_pred))
	print(confusion_matrix(y_test,y_pred))

def lassoRegression(X, Y):
	X = StandardScaler().fit_transform(X) #Scale the data

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) #Train-test split

	#Make targets 1s and 0s
	for index in range(len(y_train)):
		if y_train[index] == 'CRC':
			y_train[index] = 1
		if y_train[index] == 'CTR':
			y_train[index] = 0

	#Make targets 1s and 0s
	for index in range(len(y_test)):
		if y_test[index] == 'CRC':
			y_test[index] = 1
		if y_test[index] == 'CTR':
			y_test[index] = 0

	print(y_test)
	print(y_train)

	lasso = Lasso()
	print(lasso)
	lasso.fit(X_train, y_train)

	#Predict
	y_pred = lasso.predict(X_test)

	print(accuracy_score(y_test,y_pred.round()))
	print(confusion_matrix(y_test,y_pred.round()))

def LOSO(X,Y):

	#Add targets back to features dataframe so that they can also be deleted when rows are left out
	X['Experiment'] = Y

	CCISList = [item for item in X.index.tolist() if 'CCIS' in item] #Get all rows with CCIS in the index
	X_temp_1 = X.drop(CCISList, axis=0) #Feature training set
	y_temp_1 = X_temp_1['Experiment'].tolist() #Target training set
	X_temp_1 = X_temp_1.drop('Experiment',axis=1)

	CCIS = X.iloc[0:114] 
	CCIS_targets = CCIS['Experiment'] #Target prediction set
	CCIS = CCIS.drop('Experiment', axis=1) #Feature prediction set

	# CCMDList = [item for item in X.index.tolist() if 'CCMD' in item] #Get all rows with CCMD in the index
	# X_temp_2 =  X.drop(CCMDList, axis=0)
	# y_temp_2 = X_temp_2['Experiment'].tolist()
	# CCMD = X.iloc[114:234]
	# CCMD = CCMD.drop(['Experiment'], axis=1)

	# ERRList = [item for item in X.index.tolist() if 'ERR' in item] #Get all rows with ERR in the index
	# X_temp_3 = X.drop(ERRList, axis=0)
	# y_temp_3 = X_temp_3['Experiment'].tolist()
	# ERR = X.iloc[234:362]
	# ERR = ERR.drop(['Experiment'], axis=1)

	# MMRSList = [item for item in X.index.tolist() if 'MMRS' in item] #Get all rows with MMRS in the index
	# X_temp_4 = X.drop(MMRSList, axis=0)
	# y_temp_4 = X_temp_4['Experiment'].tolist()
	# MMRS = X.iloc[362:466]
	# MMRS = MMRS.drop(['Experiment'], axis=1)


	# SAMEAList = [item for item in X.index.tolist() if 'SAMEA' in item] #Get all rows with SAMEA in the index
	# X_temp_5 = X.drop(SAMEAList, axis=0)
	# y_temp_5 = X_temp_5['Experiment'].tolist()
	# SAMEA = X.iloc[466:575]
	# SAMEA = SAMEA.drop(['Experiment'], axis=1)

	#Scale the data
	CCIS = StandardScaler().fit_transform(CCIS)
	# CCMD = StandardScaler().fit_transform(CCMD)
	# ERR = StandardScaler().fit_transform(ERR) 
	# MMRS = StandardScaler().fit_transform(MMRS) 
	# SAMEA = StandardScaler().fit_transform(SAMEA) 
	
	#Initialize classifier
	logReg = LogisticRegression(C=10, max_iter=200)
	print(logReg)
	logReg.fit(X_temp_1, y_temp_1)

	#Predict
	y_pred = logReg.predict(CCIS)

	print(accuracy_score(CCIS_targets,y_pred))
	print(confusion_matrix(CCIS_targets,y_pred))


#Call methods and do final processing
#binaryData(X,0.001) #Make features binary (1 if above threshold 0 if below)

LOSO(X,Y)

#kneighbors(X, Y)
#logisticRegeression(X,Y)
#lassoRegression(X, Y)






















# #Preprocess dataset to predict on

# predictFeaturesDf = pd.read_csv('predict_abundances.csv')
# predictFeaturesDf = predictFeaturesDf.T

# newColumns = []
# for index in range(1753):
# 	newColumns.append(predictFeaturesDf.at['Unnamed: 0', index])

# predictFeaturesDf = predictFeaturesDf.drop('Unnamed: 0',axis=0)
# newColumns = [x.split('[',1)[0] for x in newColumns]
# predictFeaturesDf.columns = newColumns


# #Preprocess metadata for dataset to predict on
# predictTargetsDf = pd.read_csv('predict_metadata.csv')

# #Remove samples with no metadata and samples with metadata not regarding cancer or normal
# index=0
# for entry in predictTargetsDf['Alias']:
# 	if entry not in predictFeaturesDf.index.tolist() or predictTargetsDf.at[index,'diagnosis'] == '' or (predictTargetsDf.at[index,'diagnosis'] != 'Normal' and predictTargetsDf.at[index,'diagnosis'] != 'Cancer'):
# 		predictTargetsDf = predictTargetsDf.drop(index, axis=0)
# 	index+=1

# #Change 'Cancer' to CRC and normal to 'CTR'
# for entry in predictTargetsDf['diagnosis']




