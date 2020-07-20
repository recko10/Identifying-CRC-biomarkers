import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

# #Preprocess data
# featuresDf = pd.read_csv('taxonomic_abundances_test.csv') #Load in df
# featuresDf = featuresDf.T #Transpose featuresDf (switch rows and columns and adjust values accordingly)
# featuresDf = featuresDf.drop(0, axis=1)

# columnList = []
# for column in featuresDf.columns.tolist():
# 	columnList.append(featuresDf.at['Unnamed: 0', column])

# columnList = [x.split('[',1)[0] for x in columnList]

# featuresDf.columns = columnList #Change indices to values in this list
# featuresDf = featuresDf.drop('Unnamed: 0', axis=0) #Drop the column

# featuresDf.to_csv('results.csv')

# featuresDf['Experiment'] = '' #Create empty column to be filled with metadata items later

# #Create df with metadata
# targetDf = pd.read_csv('metadata_test.csv')

# #Set the appropriate rows in the Experiment column to be equal to the appropriate rows of the Group column
# index = 0
# for sample in targetDf['Sample ID']:
# 	featuresDf.loc[sample,'Experiment'] = targetDf.loc[index, 'Group'] 
# 	index+=1

# #Remove rows without any metadata
# featuresDf = featuresDf[featuresDf.Experiment != '']

# #This new dataframe represents featuresDf but with the metadata column in place
# finalFeaturesDf = featuresDf

# #Remove the metadata column to prepare for standard scaler
# featuresDf = featuresDf.drop(columns = 'Experiment')

# #Decompose the data
# x = StandardScaler().fit_transform(featuresDf) #Normalize the data

# #At this point our featuresDf has all of the raw data as well as an additional column called 'Experiment' which has the appropriate metadata

# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x) #Transform the scaled data onto a new vector space
# principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2']) #Create new dataframe with principal components as the data

# #Append list of targets to principalDf
# targetsTemp = [target for target in finalFeaturesDf.Experiment]
# principalDf['target'] = targetsTemp

#Plotting the principal components and assigning colors to the datapoints
#Takes as input a dataframe with principal components + targets, a list of targets (no repeats), and a list of colors to assign to each of the targets when plotting
def plotPCA(finalDf, targets, colors):

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

#plotPCA(principalDf, ['CRC','CTR'], ['r','b'])