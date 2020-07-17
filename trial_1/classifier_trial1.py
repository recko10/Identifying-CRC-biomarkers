import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFdr, chi2
from sklearn.feature_selection import SelectKBest

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

#Identify important features--takes a list of coefficients, a list of all the bacteria, and a prescaled feature dataframe as input
def featureImportanceRegression(model, bacteria, X_prescale):
	importantBacteria = []
	coefficientList = model.coef_.tolist()[0] 
	#Identify bacteria with most impact on the model by identifying coefficients of high magnitude
	for index in range(len(coefficientList)):
		if coefficientList[index] < -0.40:
			importantBacteria.append(bacteria[index])

	print(f'Most impactful bacteria: {importantBacteria}\n')
	print(f'Number of most impactful bacteria: {len(importantBacteria)}')
	# X_feature_selected = SelectKBest(chi2, k=20).fit_transform(X, Y)
	# importantFeatureData = X_feature_selected.tolist()

	# importantBacteria = []
	# for item in importantFeatureData:
	# 	for column in X.columns.tolist():
	# 		if item == X[column].tolist():
	# 			importantBacteria.append(column)
	# print(importantBacteria)


#Scales features and then maps them onto a new vector space
def pca(X):

	indices = X.index #Save X inidces to import into principalDf once it is created
	X = StandardScaler().fit_transform(X) #Scale the data

	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(X) #Transform the scaled data onto a new vector space
	principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2']) #Create new dataframe with principal components as the data

	principalDf.index = indices
	return principalDf

#Convert all numbers into 'yes' or 'no' values indicating the presence of the bacteria (yes is 1 and no is 0)
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

def logisticRegeression(X, Y, loso=False, losoFeatureTrain = pd.DataFrame(['Empty']), losoTargetTrain = [], losoFeaturePredict = pd.DataFrame(['Empty']), losoTargetPredict = []):
	if loso == False:
		X_prescale = X
		bacteria = X_prescale.columns.tolist()
		X = StandardScaler().fit_transform(X) #Scale the data
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) #Train-test split

	#If it is being called from the LOSO function
	if loso == True:
		#Initialize classifier
		logReg = LogisticRegression(C=10, max_iter=200)
		logReg.fit(losoFeatureTrain, losoTargetTrain)

		#Predict
		y_pred = logReg.predict(losoFeaturePredict)
		y_pred_roc = logReg.decision_function(losoFeaturePredict)

		#Print out accuracy metrics
		print(f'Accuracy score: {accuracy_score(losoTargetPredict,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(losoTargetPredict,y_pred)}')
		print(f'AUROC score: {roc_auc_score(losoTargetPredict, y_pred_roc)}\n')
		return

	#Initialize classifier
	logReg = LogisticRegression(C=10, max_iter=200)
	logReg.fit(X_train, y_train)

	#Predict
	y_pred = logReg.predict(X_test)
	y_pred_roc = logReg.decision_function(X_test)

	print(f'Accuracy score: {accuracy_score(y_test,y_pred)}')
	print(f'Confusion matrix: {confusion_matrix(y_test,y_pred)}')
	print(f'AUROC score: {roc_auc_score(y_test, y_pred_roc)}\n')

	#Identify important features
	featureImportanceRegression(logReg, bacteria, X_prescale)

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

#Leave-one-study-out validation
def LOSO(X,Y):

	#Add targets back to features dataframe so that they can also be deleted when rows are left out
	X['Experiment'] = Y

	#CCIS processing
	CCISList = [item for item in X.index.tolist() if 'CCIS' in item] #Get all rows with CCIS in the index
	X_temp_1 = X.drop(CCISList, axis=0) #Feature training set
	y_temp_1 = X_temp_1['Experiment'].tolist() #Target training set
	X_temp_1 = X_temp_1.drop('Experiment',axis=1)

	CCIS = X.iloc[0:114] 
	CCIS_targets = CCIS['Experiment'] #Target prediction set
	CCIS = CCIS.drop('Experiment', axis=1) #Feature prediction set

	#CCMD processing
	CCMDList = [item for item in X.index.tolist() if 'CCMD' in item] #Get all rows with CCMD in the index
	X_temp_2 = X.drop(CCMDList, axis=0) #Feature training set
	y_temp_2 = X_temp_2['Experiment'].tolist() #Target training set
	X_temp_2 = X_temp_2.drop('Experiment',axis=1)

	CCMD = X.iloc[114:234] 
	CCMD_targets = CCMD['Experiment'] #Target prediction set
	CCMD = CCMD.drop('Experiment', axis=1) #Feature prediction set

	#ERR processing
	ERRList = [item for item in X.index.tolist() if 'ERR' in item] #Get all rows with ERR in the index
	X_temp_3 = X.drop(ERRList, axis=0) #Feature training set
	y_temp_3 = X_temp_3['Experiment'].tolist() #Target training set
	X_temp_3 = X_temp_3.drop('Experiment',axis=1)

	ERR = X.iloc[234:362] 
	ERR_targets = ERR['Experiment'] #Target prediction set
	ERR = ERR.drop('Experiment', axis=1) #Feature prediction set

	#MMRS processing
	MMRSList = [item for item in X.index.tolist() if 'MMRS' in item] #Get all rows with MMRS in the index
	X_temp_4 = X.drop(MMRSList, axis=0) #Feature training set
	y_temp_4 = X_temp_4['Experiment'].tolist() #Target training set
	X_temp_4 = X_temp_4.drop('Experiment',axis=1)

	MMRS = X.iloc[362:466] 
	MMRS_targets = MMRS['Experiment'] #Target prediction set
	MMRS = MMRS.drop('Experiment', axis=1) #Feature prediction set


	#SAMEA processing
	SAMEAList = [item for item in X.index.tolist() if 'SAMEA' in item] #Get all rows with SAMEA in the index
	X_temp_5 = X.drop(SAMEAList, axis=0) #Feature training set
	y_temp_5 = X_temp_5['Experiment'].tolist() #Target training set
	X_temp_5 = X_temp_5.drop('Experiment',axis=1)

	SAMEA = X.iloc[466:575] 
	SAMEA_targets = SAMEA['Experiment'] #Target prediction set
	SAMEA = SAMEA.drop('Experiment', axis=1) #Feature prediction set

	#Scale the data
	CCIS = StandardScaler().fit_transform(CCIS)
	CCMD = StandardScaler().fit_transform(CCMD)
	ERR = StandardScaler().fit_transform(ERR) 
	MMRS = StandardScaler().fit_transform(MMRS) 
	SAMEA = StandardScaler().fit_transform(SAMEA) 

	#Call return logistic regression function calls
	return logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_1, losoTargetTrain=y_temp_1, losoFeaturePredict=CCIS, losoTargetPredict=CCIS_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_2, losoTargetTrain=y_temp_2, losoFeaturePredict=CCMD, losoTargetPredict=CCMD_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_3, losoTargetTrain=y_temp_3, losoFeaturePredict=ERR, losoTargetPredict=ERR_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_4, losoTargetTrain=y_temp_4, losoFeaturePredict=MMRS, losoTargetPredict=MMRS_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_5, losoTargetTrain=y_temp_5, losoFeaturePredict=SAMEA, losoTargetPredict=SAMEA_targets)

#Leave-one-study-out validation but the US (MMRS) dataset is disregarded as an outlier
def LOSONoUS(X,Y):

	#Add targets back to features dataframe so that they can also be deleted when rows are left out
	X['Experiment'] = Y

	#Remove US dataset
	X = X[~X.index.str.contains('MMRS')]

	#CCIS processing
	CCISList = [item for item in X.index.tolist() if 'CCIS' in item] #Get all rows with CCIS in the index
	X_temp_1 = X.drop(CCISList, axis=0) #Feature training set
	y_temp_1 = X_temp_1['Experiment'].tolist() #Target training set
	X_temp_1 = X_temp_1.drop('Experiment',axis=1)

	CCIS = X.iloc[0:114] 
	CCIS_targets = CCIS['Experiment'] #Target prediction set
	CCIS = CCIS.drop('Experiment', axis=1) #Feature prediction set

	#CCMD processing
	CCMDList = [item for item in X.index.tolist() if 'CCMD' in item] #Get all rows with CCMD in the index
	X_temp_2 = X.drop(CCMDList, axis=0) #Feature training set
	y_temp_2 = X_temp_2['Experiment'].tolist() #Target training set
	X_temp_2 = X_temp_2.drop('Experiment',axis=1)

	CCMD = X.iloc[114:234] 
	CCMD_targets = CCMD['Experiment'] #Target prediction set
	CCMD = CCMD.drop('Experiment', axis=1) #Feature prediction set

	#ERR processing
	ERRList = [item for item in X.index.tolist() if 'ERR' in item] #Get all rows with ERR in the index
	X_temp_3 = X.drop(ERRList, axis=0) #Feature training set
	y_temp_3 = X_temp_3['Experiment'].tolist() #Target training set
	X_temp_3 = X_temp_3.drop('Experiment',axis=1)

	ERR = X.iloc[234:362] 
	ERR_targets = ERR['Experiment'] #Target prediction set
	ERR = ERR.drop('Experiment', axis=1) #Feature prediction set

	#SAMEA processing
	SAMEAList = [item for item in X.index.tolist() if 'SAMEA' in item] #Get all rows with SAMEA in the index
	X_temp_4 = X.drop(SAMEAList, axis=0) #Feature training set
	y_temp_4 = X_temp_4['Experiment'].tolist() #Target training set
	X_temp_4 = X_temp_4.drop('Experiment',axis=1)

	SAMEA = X.iloc[362:466] 
	SAMEA_targets = SAMEA['Experiment'] #Target prediction set
	SAMEA = SAMEA.drop('Experiment', axis=1) #Feature prediction set

	#Scale the data
	CCIS = StandardScaler().fit_transform(CCIS)
	CCMD = StandardScaler().fit_transform(CCMD)
	ERR = StandardScaler().fit_transform(ERR) 
	SAMEA = StandardScaler().fit_transform(SAMEA) 


	#Call return logistic regression function calls
	return logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_1, losoTargetTrain=y_temp_1, losoFeaturePredict=CCIS, losoTargetPredict=CCIS_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_2, losoTargetTrain=y_temp_2, losoFeaturePredict=CCMD, losoTargetPredict=CCMD_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_3, losoTargetTrain=y_temp_3, losoFeaturePredict=ERR, losoTargetPredict=ERR_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_4, losoTargetTrain=y_temp_4, losoFeaturePredict=SAMEA, losoTargetPredict=SAMEA_targets)

def LOSONoFrance(X,Y):

	#Add targets back to features dataframe so that they can also be deleted when rows are left out
	X['Experiment'] = Y

	#Remove US dataset
	X = X[~X.index.str.contains('CCIS')]

	#CCMD processing
	CCMDList = [item for item in X.index.tolist() if 'CCMD' in item] #Get all rows with CCMD in the index
	X_temp_2 = X.drop(CCMDList, axis=0) #Feature training set
	y_temp_2 = X_temp_2['Experiment'].tolist() #Target training set
	X_temp_2 = X_temp_2.drop('Experiment',axis=1)

	CCMD = X.iloc[0:114] 
	CCMD_targets = CCMD['Experiment'] #Target prediction set
	CCMD = CCMD.drop('Experiment', axis=1) #Feature prediction set

	#ERR processing
	ERRList = [item for item in X.index.tolist() if 'ERR' in item] #Get all rows with ERR in the index
	X_temp_3 = X.drop(ERRList, axis=0) #Feature training set
	y_temp_3 = X_temp_3['Experiment'].tolist() #Target training set
	X_temp_3 = X_temp_3.drop('Experiment',axis=1)

	ERR = X.iloc[114:234] 
	ERR_targets = ERR['Experiment'] #Target prediction set
	ERR = ERR.drop('Experiment', axis=1) #Feature prediction set

	#MMRS processing
	MMRSList = [item for item in X.index.tolist() if 'MMRS' in item] #Get all rows with MMRS in the index
	X_temp_4 = X.drop(MMRSList, axis=0) #Feature training set
	y_temp_4 = X_temp_4['Experiment'].tolist() #Target training set
	X_temp_4 = X_temp_4.drop('Experiment',axis=1)

	MMRS = X.iloc[234:362] 
	MMRS_targets = MMRS['Experiment'] #Target prediction set
	MMRS = MMRS.drop('Experiment', axis=1) #Feature prediction set


	#SAMEA processing
	SAMEAList = [item for item in X.index.tolist() if 'SAMEA' in item] #Get all rows with SAMEA in the index
	X_temp_5 = X.drop(SAMEAList, axis=0) #Feature training set
	y_temp_5 = X_temp_5['Experiment'].tolist() #Target training set
	X_temp_5 = X_temp_5.drop('Experiment',axis=1)

	SAMEA = X.iloc[362:466] 
	SAMEA_targets = SAMEA['Experiment'] #Target prediction set
	SAMEA = SAMEA.drop('Experiment', axis=1) #Feature prediction set

	#Scale the data
	CCMD = StandardScaler().fit_transform(CCMD)
	ERR = StandardScaler().fit_transform(ERR) 
	MMRS = StandardScaler().fit_transform(MMRS) 
	SAMEA = StandardScaler().fit_transform(SAMEA) 

	#Call return logistic regression function calls
	return logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_2, losoTargetTrain=y_temp_2, losoFeaturePredict=CCMD, losoTargetPredict=CCMD_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_3, losoTargetTrain=y_temp_3, losoFeaturePredict=ERR, losoTargetPredict=ERR_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_4, losoTargetTrain=y_temp_4, losoFeaturePredict=MMRS, losoTargetPredict=MMRS_targets), logisticRegeression(X,Y, loso=True,losoFeatureTrain=X_temp_5, losoTargetTrain=y_temp_5, losoFeaturePredict=SAMEA, losoTargetPredict=SAMEA_targets)



#Call methods and do final processing
#X = pca(X)
#binaryData(X,0.001) #Make features binary (1 if above threshold 0 if below)

#LOSO(X,Y)
#LOSONoUS(X,Y)
#LOSONoFrance(X,Y)
#kneighbors(X, Y)
logisticRegeression(X,Y)
#lassoRegression(X, Y)

