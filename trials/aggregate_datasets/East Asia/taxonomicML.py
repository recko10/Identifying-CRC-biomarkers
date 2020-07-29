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
from sklearn.metrics import classification_report

class ML:

	#Takes as input a features df and a list of corresponding targets
	def pca(self, X, Y, targets=['CRC','control'], colors=['r','b']):

		#Scale
		indices = X.index #Save X inidces to import into principalDf once it is created
		X = StandardScaler().fit_transform(X) #Scale the data

		#PCA transform
		pca = PCA(n_components=2)
		principalComponents = pca.fit_transform(X) #Transform the scaled data onto a new vector space
		#principalComponents = principalComponents[:, [4, 5]]
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

	#Selects the top 30 features from a given fitted model
	def selectFromModel(self, model, X_train, Y_train):

		headers = X_train.columns.tolist()
		selectedFeatures = []

		#Create model
		sfm = SelectFromModel(estimator=model, max_features=30)

		#Scale the data
		X_train = StandardScaler().fit_transform(X_train)

		#Train the model to select the features
		sfm.fit(X_train, Y_train)

		# Print the names of the most important features
		for feature_list_index in sfm.get_support(indices=True):
			selectedFeatures.append(headers[feature_list_index])

		return selectedFeatures

	#Takes a dataframe as input and a list of corresponding targets. Outputs a diagonal correlation matrix with the top features from the dataframe.
	def correlationMatrix(self, d, Y_train):

		ml = ML()
		#Find important features
		selectedFeatures = ml.selectFromModel(RandomForestClassifier().fit(d,Y_train), d, Y_train)

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
		selectedFeatures = ml.selectFromModel(rf, X_train_prescale, Y_train)
		print(selectedFeatures)

		#Print classification report
		print(classification_report(Y_test, y_pred, target_names=['CRC','control']))

		return y_pred

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

		ml = ML()

		#Get top features
		selectedFeatures = ml.selectFromModel(logReg, X_prescale_train, Y_train)
		print(selectedFeatures)

		#Print classification report
		print(classification_report(Y_test, y_pred, target_names=['CRC','control']))

		return y_pred



	