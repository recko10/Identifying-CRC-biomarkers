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
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from string import ascii_letters
from sklearn.feature_selection import SelectFromModel

class ML:
	#Identify important features--takes a list of coefficients, a list of all the bacteria, a prescaled feature dataframe, and its corresponding targets as input
	def featureImportanceRegression(self, model, X_prescale, Y):

		#Clone the model (create duplicate with same paramters but that is not fit to data)
		model = clone(model)

		#Create the RFE model and select the top 'n_features_to_select' features
		rfe = RFE(model, n_features_to_select=30)
		X_scale = StandardScaler().fit_transform(X_prescale)

		rfe.fit(X_scale,Y)

		#Get all of the features under a threshold 
		selectedFeatures = []
		index=0
		for index in range(len(X_prescale.columns.tolist())):
			if rfe.ranking_[index] < 10:
				selectedFeatures.append(X_prescale.columns.tolist()[index])
		print(f'Most impactful bacteria (RFE): {selectedFeatures}')
		print(f'Number of most impactful bacteria (RFE): {len(selectedFeatures)}\n')

		return selectedFeatures

	#Takes as input a features df and a list of corresponding targets
	def pca(self, X, Y, targets=['CRC','control'], colors=['r','b']):

		#Scale
		indices = X.index #Save X inidces to import into principalDf once it is created
		X = StandardScaler().fit_transform(X) #Scale the data

		#PCA transform
		pca = PCA(n_components=2)
		principalComponents = pca.fit_transform(X) #Transform the scaled data onto a new vector space
		principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2']) #Create new dataframe with principal components as the data

		principalDf.index = indices

		finalDf = principalDf

		#Append targets to df before sending it to be plotted
		finalDf['target'] = Y

		#Plot the principal components
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		for target, color in zip(targets,colors):
		    indicesToKeep = finalDf['target'] == target
		    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
		               , finalDf.loc[indicesToKeep, 'principal component 2']
		               , c = color
		               , s = 50)
		ax.legend(targets)
		ax.grid()
		plt.show()
		return principalDf

	def selectFromModel(self, X_train, Y_train):
		selectedFeatures = []
		#Create model
		sfm = SelectFromModel(estimator=RandomForestClassifier().fit(X_train,Y_train), max_features=30)
		#Train the model to select the features
		sfm.fit(X_train, Y_train)
		# Print the names of the most important features
		for feature_list_index in sfm.get_support(indices=True):
			selectedFeatures.append(X_train.columns.tolist()[feature_list_index])

		return selectedFeatures

	#Takes a dataframe as input and a list of corresponding targets. Outputs a diagonal correlation matrix with the top features from the dataframe.
	def correlationMatrix(self, d, Y_train):

		ml = ML()
		#Find important features
		selectedFeatures = ml.selectFromModel(d, Y_train)

		#Remove unimportant features
		for header in d.columns.tolist():
			if header not in selectedFeatures:
				d = d.drop(header,axis=1)

		# Compute the correlation matrix
		corr = d.corr()
		# Generate a mask for the upper triangle
		mask = np.triu(np.ones_like(corr, dtype=np.bool))
		# Set up the matplotlib figure
		f, ax = plt.subplots(figsize=(15, 11)) #original figsize = (11,9)
		# Generate a custom diverging colormap
		cmap = sns.diverging_palette(220, 10, as_cmap=True)
		# Draw the heatmap with the mask and correct aspect ratio
		sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
		            square=True, linewidths=.5, cbar_kws={"shrink": .5})

		ax.figure.subplots_adjust(bottom = 0.3)
		plt.show()

	def randomForest(self, X_train, X_test, Y_train, Y_test):
		ml = ML()

		#Save the pandas dataframe before it gets scaled
		X_train_prescale = X_train

		#Scale the data
		X_train = StandardScaler().fit_transform(X_train)
		X_test = StandardScaler().fit_transform(X_test)

		#Initialize classifier
		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)

		#Predict
		y_pred = rf.predict(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')
		print(f'AUROC score: {roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1])}\n')

		#Get top features
		selectedFeatures = ml.selectFromModel(X_train_prescale, Y_train)
		print(selectedFeatures)

	def kneighbors(self, X_train, X_test, Y_train, Y_test):

		X_train = StandardScaler().fit_transform(X_train) #Scale the data
		X_test= StandardScaler().fit_transform(X_test) #Scale the data

		#Initialize classifier
		kn = KNeighborsClassifier(n_neighbors=3)
		kn.fit(X_train, Y_train)

		#Predict
		y_pred = kn.predict(X_test)
		y_pred_roc = kn.decision_function(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')
		print(f'AUROC score: {roc_auc_score(Y_test, y_pred_roc)}\n')

	def logisticRegeression(self, X_train, X_test, Y_train, Y_test):
		#Scale and create splits
		X_prescale_train = X_train
		X_prescale_test = X_test

		bacteriaTrain = X_prescale_train.columns.tolist()
		bacteriaTest = X_prescale_test.columns.tolist()

		X_train = StandardScaler().fit_transform(X_train) #Scale the data
		X_test= StandardScaler().fit_transform(X_test) #Scale the data

		#Initialize classifier
		logReg = LogisticRegression(C=10, max_iter=200)
		logReg.fit(X_train, Y_train)

		#Predict
		y_pred = logReg.predict(X_test)
		y_pred_roc = logReg.decision_function(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')
		print(f'AUROC score: {roc_auc_score(Y_test, y_pred_roc)}\n')

		#Identify important features for both the train and test sets

		ml = ML()

		print('Train data feature importance information: \n')
		ml.featureImportanceRegression(logReg, bacteriaTrain, X_prescale_train, Y_train)

		print('Test data feature importance information: \n')
		ml.featureImportanceRegression(logReg, bacteriaTest, X_prescale_test, Y_test)

	def lassoRegression(self, X_train, X_test, Y_train, Y_test):
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

		#Create model and fit data
		lasso = Lasso()
		lasso.fit(X_train, y_train)

		#Predict
		y_pred = lasso.predict(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')
		print(f'AUROC score: {roc_auc_score(Y_test, y_pred_roc)}\n')