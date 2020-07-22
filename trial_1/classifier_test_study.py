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

#Preprocess data
featuresDf = pd.read_csv('data/taxonomic_abundances.csv') #Load in df

#indexList = [entry.split('[',1)[0] for entry in featuresDf['Unnamed: 0']] #List of bacteria names 
indexList = [entry for entry in featuresDf['Unnamed: 0']] #List of bacteria names 

featuresDf.index = indexList #Change indices to values in this column
featuresDf = featuresDf.drop(columns='Unnamed: 0') #Drop the column

featuresDf = featuresDf.T #Transpose featuresDf (switch rows and columns and adjust values accordingly)

featuresDf['Experiment'] = ''

#Create df with metadata
targetDf = pd.read_csv('data/metadata.csv')

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
X_train = featuresDf.drop(columns = 'Experiment')

#Convert the metadata column into a list of labels
Y_train = finalFeaturesDf['Experiment'].tolist()

#Preprocess test features
preprocess = preprocess()
dfList = preprocess.standardPreprocess('data/ThomasAM_2018a')
X_test = dfList[0]


# #Remove unnecessary features from the train set
# newHeaders = [x for x in X_train.columns.tolist()]
# for element in X_train.columns.tolist():
# 	if 'sp.' in element:
# 		newHeaders.remove(element)
# 		continue
# 	if 'unknown' in element:
# 		newHeaders.remove(element)
# 		continue
# 	if len(element.split('\s')) > 3: ##FIX THIS
# 		newHeaders.remove(element)
# 		continue
# 	if '[' in element:
# 		newHeaders[newHeaders.index(element)] = newHeaders[newHeaders.index(element)].split('[', 1)[0]


# print(newHeaders)




#Preprocess test targets
thomasDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Mark unnecessary columns and append to targets list
Y_test = [x for x in thomasDf.iloc[3, :].tolist()]
Y_test.pop(0)

for index in range(len(Y_test)):
	if Y_test[index] == 'adenoma':
		X_test = X_test.drop(thomasDf.columns.tolist()[index], axis=0)
		continue
	if Y_test[index] == 'control':
		Y_test[index] = 'CTR'

Y_test = [x for x in Y_test if x != 'adenoma']





