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
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from string import ascii_letters
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import sklearn.metrics as metrics
from sklearn import preprocessing

class ML:

	#Takes as input a features df and a list of corresponding targets
	def pca(self, X, Y, targets=['CRC','control'], colors=['r','b']):
		ml = ML()

		#Scale
		X_prescale = X
		indices = X.index #Save X inidces to import into principalDf once it is created
		X = StandardScaler().fit_transform(X) #Scale the data

		#PCA transform
		pca = PCA(n_components=4)
		principalComponents = pca.fit_transform(X) #Transform the scaled data onto a new vector space
		principalComponents = principalComponents[:, [2,3]]
		principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2']) #Create new dataframe with principal components as the data

		principalDf.index = indices

		finalDf = principalDf

		#Generate heatmap to see influence of features on principal components
		ml.pcaHeatmap(pca, X_prescale, 0.12)

		#Append targets to df before sending it to be plotted
		finalDf['target'] = Y

		#Plot the principal components
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 3', fontsize = 15) ####REMEMBER TO CHANGE THESE WHEN CONSIDERING DIFFERENT PRINCIPAL COMPONENTS
		ax.set_ylabel('Principal Component 4', fontsize = 15) ####REMEMBER TO CHANGE THESE WHEN CONSIDERING DIFFERENT PRINCIPAL COMPONENTS
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

	def pcaHeatmap(self, pca, x, eigenThreshold):
		map = pd.DataFrame(pca.components_, columns=x.columns)
		map = map.tail(2) #Always select the last 2 principal components--assuming that the user always PCAs with only the PCs they need

		#Filter out columns with eigenvector components less than a certain threshold
		for column in map.columns.tolist():
			if abs(map.at[map.index.tolist()[0], column]) < eigenThreshold and abs(map.at[map.index.tolist()[1], column]) < eigenThreshold:
				map = map.drop(column, axis=1)

		map = map.T #Transpose to make the heatmap easier to read
		map.columns = [x+1 for x in map.columns]
		plt.figure(figsize=(12,6))
		plt.gcf().subplots_adjust(left=0.25)
		sns.heatmap(map, cmap='coolwarm')

	def tsne(self, X, Y, targets=['CRC','control'], colors=['r','b']):

		#Scale
		indices = X.index #Save X indices to import into principalDf once it is created
		X = StandardScaler().fit_transform(X) #Scale the data

		#TSNE transform
		tsne = TSNE(n_components=2)
		tsneComponents = tsne.fit_transform(X) #Transform the scaled data onto a new vector space
		tsneDf = pd.DataFrame(data=tsneComponents, columns = ['TSNE component 1', 'TSNE component 2']) #Create new dataframe with principal components as the data

		tsneDf.index = indices

		finalDf = tsneDf

		#Append targets to df before sending it to be plotted
		finalDf['target'] = Y

		#Plot the principal components
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('TSNE Component 1', fontsize = 15)
		ax.set_ylabel('TSNE Component 2', fontsize = 15)
		for target, color in zip(targets,colors):
		    indicesToKeep = finalDf['target'] == target
		    ax.scatter(finalDf.loc[indicesToKeep, 'TSNE component 1']
		               , finalDf.loc[indicesToKeep, 'TSNE component 2']
		               , c = color
		               , s = 50)
		ax.legend(targets)
		ax.grid()
		plt.show()
		return tsneDf

	#Create a scree plot
	def scree(self, X):

		num_vars = len(X.columns.tolist())
		num_obs = len(X.index.tolist())

		#Convert df to numpy array
		A = X.to_numpy()

		A = np.asmatrix(A.T) * np.asmatrix(A)
		U, S, V = np.linalg.svd(A) #Singular vector decomposition
		eigvals = S**2 / np.sum(S**2)  

		fig = plt.figure(figsize=(8,5))
		sing_vals = np.arange(num_vars) + 1
		plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
		plt.title('Scree Plot')
		plt.xlabel('Principal Component')
		plt.ylabel('Eigenvalue')

		leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
		                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
		                 markerscale=0.4)
		leg.get_frame().set_alpha(0.4)
		plt.show()

	#Selects the top 20 features from a given fitted model
	def selectFromModel(self, model, X_train, Y_train):

		headers = X_train.columns.tolist()
		selectedFeatures = []

		#Create model
		sfm = SelectFromModel(estimator=model, max_features=20)

		#Scale the data
		X_train = StandardScaler().fit_transform(X_train)

		#Train the model to select the features
		sfm.fit(X_train, Y_train)

		# Print the names of the most important features
		for feature_list_index in sfm.get_support(indices=True):
			selectedFeatures.append(headers[feature_list_index])

		#Get ranked feature importances by looking at impurities
		rankedImportances = model.feature_importances_

		return selectedFeatures, rankedImportances

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

	def randomForest(self, X_train, X_test, Y_train, Y_test, multi_class=False):
		ml = ML()

		#Save the pandas dataframe before it gets scaled
		X_train_prescale = X_train

		#Normalize the data
		X_train = preprocessing.normalize(X_train, norm='l1', axis=0)
		X_test = preprocessing.normalize(X_test, norm='l1', axis=0)

		#Initialize classifier
		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)

		#Predict
		y_pred = rf.predict(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')

		#Check if the classifcation task has multiple classes
		if multi_class == True:
			print(f'AUROC score: {roc_auc_score(Y_test, rf.predict_proba(X_test), multi_class="ovr")}\n')
		else:
			print(f'AUROC score: {roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1])}\n')

		#Get top features
		selectedFeatures, rankedImportances = ml.selectFromModel(rf, X_train_prescale, Y_train)
		print(selectedFeatures)

		#Print classification report
		print(classification_report(Y_test, y_pred))

		return y_pred

	def logisticRegression(self, X_train, X_test, Y_train, Y_test, multi_class=False):
		#Scale and create splits
		X_prescale_train = X_train
		X_prescale_test = X_test

		bacteriaTrain = X_prescale_train.columns.tolist()
		bacteriaTest = X_prescale_test.columns.tolist()
        
        #Standardize the data        
		X_train = StandardScaler().fit_transform(X_train) 
		X_test = StandardScaler().fit_transform(X_test) 

		#Initialize classifier
		logReg = LogisticRegression(C=10, max_iter=200)
		logReg.fit(X_train, Y_train)

		#Predict
		y_pred = logReg.predict(X_test)
		y_pred_roc = logReg.decision_function(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')

		if multi_class == True:
			print(f'AUROC score: {roc_auc_score(Y_test, logReg.predict_proba(X_test), multi_class="ovr")}\n')
		else:
			print(f'AUROC score: {roc_auc_score(Y_test, y_pred_roc)}\n')

		#Print classification report
		print(classification_report(Y_test, y_pred))

		return y_pred



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
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from string import ascii_letters
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import sklearn.metrics as metrics
from sklearn import preprocessing

class ML:

	#Takes as input a features df and a list of corresponding targets
	def pca(self, X, Y, targets=['CRC','control'], colors=['r','b']):
		ml = ML()

		#Scale
		X_prescale = X
		indices = X.index #Save X inidces to import into principalDf once it is created
		X = StandardScaler().fit_transform(X) #Scale the data

		#PCA transform
		pca = PCA(n_components=4)
		principalComponents = pca.fit_transform(X) #Transform the scaled data onto a new vector space
		principalComponents = principalComponents[:, [2,3]]
		principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2']) #Create new dataframe with principal components as the data

		principalDf.index = indices

		finalDf = principalDf

		#Generate heatmap to see influence of features on principal components
		ml.pcaHeatmap(pca, X_prescale, 0.12)

		#Append targets to df before sending it to be plotted
		finalDf['target'] = Y

		#Plot the principal components
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 3', fontsize = 15) ####REMEMBER TO CHANGE THESE WHEN CONSIDERING DIFFERENT PRINCIPAL COMPONENTS
		ax.set_ylabel('Principal Component 4', fontsize = 15) ####REMEMBER TO CHANGE THESE WHEN CONSIDERING DIFFERENT PRINCIPAL COMPONENTS
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

	def pcaHeatmap(self, pca, x, eigenThreshold):
		map = pd.DataFrame(pca.components_, columns=x.columns)
		map = map.tail(2) #Always select the last 2 principal components--assuming that the user always PCAs with only the PCs they need

		#Filter out columns with eigenvector components less than a certain threshold
		for column in map.columns.tolist():
			if abs(map.at[map.index.tolist()[0], column]) < eigenThreshold and abs(map.at[map.index.tolist()[1], column]) < eigenThreshold:
				map = map.drop(column, axis=1)

		map = map.T #Transpose to make the heatmap easier to read
		map.columns = [x+1 for x in map.columns]
		plt.figure(figsize=(12,6))
		plt.gcf().subplots_adjust(left=0.25)
		sns.heatmap(map, cmap='coolwarm')

	def tsne(self, X, Y, targets=['CRC','control'], colors=['r','b']):

		#Scale
		indices = X.index #Save X indices to import into principalDf once it is created
		X = StandardScaler().fit_transform(X) #Scale the data

		#TSNE transform
		tsne = TSNE(n_components=2)
		tsneComponents = tsne.fit_transform(X) #Transform the scaled data onto a new vector space
		tsneDf = pd.DataFrame(data=tsneComponents, columns = ['TSNE component 1', 'TSNE component 2']) #Create new dataframe with principal components as the data

		tsneDf.index = indices

		finalDf = tsneDf

		#Append targets to df before sending it to be plotted
		finalDf['target'] = Y

		#Plot the principal components
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('TSNE Component 1', fontsize = 15)
		ax.set_ylabel('TSNE Component 2', fontsize = 15)
		for target, color in zip(targets,colors):
		    indicesToKeep = finalDf['target'] == target
		    ax.scatter(finalDf.loc[indicesToKeep, 'TSNE component 1']
		               , finalDf.loc[indicesToKeep, 'TSNE component 2']
		               , c = color
		               , s = 50)
		ax.legend(targets)
		ax.grid()
		plt.show()
		return tsneDf

	#Create a scree plot
	def scree(self, X):

		num_vars = len(X.columns.tolist())
		num_obs = len(X.index.tolist())

		#Convert df to numpy array
		A = X.to_numpy()

		A = np.asmatrix(A.T) * np.asmatrix(A)
		U, S, V = np.linalg.svd(A) #Singular vector decomposition
		eigvals = S**2 / np.sum(S**2)  

		fig = plt.figure(figsize=(8,5))
		sing_vals = np.arange(num_vars) + 1
		plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
		plt.title('Scree Plot')
		plt.xlabel('Principal Component')
		plt.ylabel('Eigenvalue')

		leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
		                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
		                 markerscale=0.4)
		leg.get_frame().set_alpha(0.4)
		plt.show()

	#Selects the top 20 features from a given fitted model
	def selectFromModel(self, model, X_train, Y_train):

		headers = X_train.columns.tolist()
		selectedFeatures = []

		#Create model
		sfm = SelectFromModel(estimator=model, max_features=20)

		#Scale the data
		X_train = StandardScaler().fit_transform(X_train)

		#Train the model to select the features
		sfm.fit(X_train, Y_train)

		# Print the names of the most important features
		for feature_list_index in sfm.get_support(indices=True):
			selectedFeatures.append(headers[feature_list_index])

		#Get ranked feature importances by looking at impurities
		rankedImportances = model.feature_importances_

		return selectedFeatures, rankedImportances

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

	def randomForest(self, X_train, X_test, Y_train, Y_test, multi_class=False):
		ml = ML()

		#Save the pandas dataframe before it gets scaled
		X_train_prescale = X_train

		#Normalize the data
		X_train = preprocessing.normalize(X_train, norm='l1', axis=0)
		X_test = preprocessing.normalize(X_test, norm='l1', axis=0)

		#Initialize classifier
		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)

		#Predict
		y_pred = rf.predict(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')

		#Check if the classifcation task has multiple classes
		if multi_class == True:
			print(f'AUROC score: {roc_auc_score(Y_test, rf.predict_proba(X_test), multi_class="ovr")}\n')
		else:
			print(f'AUROC score: {roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1])}\n')

		#Get top features
		selectedFeatures, rankedImportances = ml.selectFromModel(rf, X_train_prescale, Y_train)
		print(selectedFeatures)

		#Print classification report
		print(classification_report(Y_test, y_pred))

		return y_pred

	def logisticRegression(self, X_train, X_test, Y_train, Y_test, multi_class=False):
		#Scale and create splits
		X_prescale_train = X_train
		X_prescale_test = X_test

		bacteriaTrain = X_prescale_train.columns.tolist()
		bacteriaTest = X_prescale_test.columns.tolist()
        
        #Standardize the data        
		X_train = StandardScaler().fit_transform(X_train) 
		X_test = StandardScaler().fit_transform(X_test) 

		#Initialize classifier
		logReg = LogisticRegression(C=10, max_iter=200)
		logReg.fit(X_train, Y_train)

		#Predict
		y_pred = logReg.predict(X_test)
		y_pred_roc = logReg.decision_function(X_test)

		print(f'Accuracy score: {accuracy_score(Y_test,y_pred)}')
		print(f'Confusion matrix: {confusion_matrix(Y_test,y_pred)}')

		if multi_class == True:
			print(f'AUROC score: {roc_auc_score(Y_test, logReg.predict_proba(X_test), multi_class="ovr")}\n')
		else:
			print(f'AUROC score: {roc_auc_score(Y_test, y_pred_roc)}\n')

		#Print classification report
		print(classification_report(Y_test, y_pred))

		return y_pred



	