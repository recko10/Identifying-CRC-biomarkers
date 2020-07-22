import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.base import clone
from taxonomicPreprocess_trial_1 import *

#TRAIN DATA

#Load in data
featuresDf = pd.read_csv('data/taxonomic_abundances.csv') #Load in df


indexList = [entry for entry in featuresDf['Unnamed: 0']] #List of bacteria names 

featuresDf.index = indexList #Change indices to values in this column
featuresDf = featuresDf.drop(columns='Unnamed: 0') #Drop the column

X_train = featuresDf

#Transpose
X_train = X_train.T

#Empty column for metadata
X_train['Experiment'] = ''

#Create df with metadata
targetDf = pd.read_csv('data/metadata.csv')

#Convert the metadata column into a list of labels
Y_train = targetDf['Group'].tolist()

#Metadata sample IDs
xIndex = targetDf['Sample_ID']

#Add metadata to applicable samples
index=0
for element in xIndex:
	if element in X_train.index.tolist():
		X_train.at[element, 'Experiment'] = Y_train[index]
	index+=1

#Remove samples with no metadata
X_train = X_train[X_train.Experiment != '']

#Remove unnecessary features from the train data
index=0
for element in X_train.columns.tolist():
	if 'sp.' in element:
		X_train = X_train.drop(element, axis=1)
		continue
	if 'unknown' in element:
		X_train = X_train.drop(element, axis=1)
		continue
	if '[' in element:
		X_train = X_train.rename(columns={element:element.split(' [', 1)[0]})
		continue
	if '/' in element:
		X_train = X_train.rename(columns={element:element.split('/', 1)[0]+ ' ' + element.split('/', 1)[1]})
	index+=1

for element in X_train.columns.tolist():
	if '/' in element:
		X_train = X_train.rename(columns={element:element.split('/', 1)[0]})

for element in X_train.columns.tolist():		
	if ' ' in element:
		X_train = X_train.rename(columns={element:element.split()[0] + '_' +element.split()[1]})

#Create list of targets
Y_train = X_train['Experiment'].tolist() 

#Remove repeats
X_train = X_train.iloc[:,~X_train.columns.duplicated()]

#Remove targets column
X_train = X_train.drop('Experiment', axis=1)


#TEST DATA

#Preprocess test features
preprocess = preprocess()
preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/decomp')

dfList = preprocess.standardPreprocess('data/decomp')
X_test = dfList[0]


#Preprocess test targets
thomasDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Mark unnecessary columns and append to targets list
Y_test = [x for x in thomasDf.iloc[3, :].tolist()]
Y_test.pop(0)

#Clean up targets
for index in range(len(Y_test)):
	if Y_test[index] == 'adenoma':
		X_test = X_test.drop(thomasDf.columns.tolist()[index], axis=0)
		continue
	if Y_test[index] == 'control':
		Y_test[index] = 'CTR'

Y_test = [x for x in Y_test if x != 'adenoma']


#Find common columns
for element in X_test.columns.tolist():
	if element not in X_train.columns.tolist():
		X_test = X_test.drop(element,axis=1)

for element in X_train.columns.tolist():
	if element not in X_test.columns.tolist():
		X_train = X_train.drop(element,axis=1)


#Identify important features--takes a list of coefficients, a list of all the bacteria, and a prescaled feature dataframe as input
def featureImportanceRegression(model, bacteria, X_prescale, Y):
	importantBacteria = []
	coefficientList = model.coef_.tolist()[0] 
	#Identify bacteria with most impact on the model by identifying coefficients of high magnitude
	for index in range(len(coefficientList)):
		if coefficientList[index] < -0.40:
			importantBacteria.append(bacteria[index])

	print(f'Most impactful bacteria (coef): {importantBacteria}\n')
	print(f'Number of most impactful bacteria (coef): {len(importantBacteria)}\n')

	#Clone the model (create duplicate with same paramters but that is not fit to data)
	model = clone(model)

	#Create the RFE model and select the top 'n_features_to_select' features
	rfe = RFE(model, n_features_to_select=50)
	rfe.fit(X_prescale,Y)

	#Get all of the features under a threshold 
	selectedFeatures = []
	index=0
	for index in range(len(X_prescale.columns.tolist())):
		if rfe.ranking_[index] < 10:
			selectedFeatures.append(X_prescale.columns.tolist()[index])
	print(f'Most impactful bacteria (RFE): {selectedFeatures}\n')
	print(f'Number of most impactful bacteria (RFE): {len(selectedFeatures)}')

#Logistic classifier-- takes in featuers (not scaled) and targets for both the train and test set
def logisticRegeression(X_train, X_test, Y_train, Y_test):

	#For feature selection
	X_prescale = X_train
	bacteria = X_prescale.columns.tolist() 

	#Scale the data
	X_train = StandardScaler().fit_transform(X_train)
	X_test = StandardScaler().fit_transform(X_test)


	#Initialize classifier
	logReg = LogisticRegression(C=10, max_iter=200)
	logReg.fit(X_train, Y_train)

	#Predict
	Y_pred = logReg.predict(X_test)
	Y_pred_roc = logReg.decision_function(X_test)

	print(f'Accuracy score: {accuracy_score(Y_test,Y_pred)}')
	print(f'Confusion matrix: {confusion_matrix(Y_test,Y_pred)}')
	print(f'AUROC score: {roc_auc_score(Y_test, Y_pred_roc)}\n')

	#Identify important features
	#featureImportanceRegression(logReg, bacteria, X_prescale,Y)


logisticRegeression(X_train, X_test, Y_train, Y_test)


