import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import *

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
	for column in X.columns.tolist():
		for index in range(len(X)):
			if X[column].iloc[index] < threshold:
				X[column].iloc[index] = 0
			else:
				X[column].iloc[index] = 1

def kneighbors(X_train, X_test, y_train, y_test):
	#Initialize classifier
	kn = KNeighborsClassifier(n_neighbors=3)
	print(kn)
	kn.fit(X_train, y_train)

	#Predict
	y_pred = kn.predict(X_test)

	print(accuracy_score(y_test,y_pred))
	print(confusion_matrix(y_test,y_pred))

def logisticRegeression(X_train, X_test, y_train, y_test):
	#Initialize classifier
	logReg = LogisticRegression(C=10, max_iter=200)
	print(logReg)
	logReg.fit(X_train, y_train)

	#Predict
	y_pred = logReg.predict(X_test)

	print(accuracy_score(y_test,y_pred))
	print(confusion_matrix(y_test,y_pred))

#Call methods and do final processing
binaryData(X,0.001) #Make data binary (1 if above threshold 0 if below)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) #Train-test split

#kneighbors(X_train, X_test, y_train, y_test)
logisticRegeression(X_train, X_test, y_train, y_test)

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




