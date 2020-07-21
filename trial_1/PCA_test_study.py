import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from taxonomicPreprocess import *

#Create object of preprocess class
preprocess = preprocess()

featuresDf, targets = preprocess.curatedMetagenomicDataFormatToTaxonomic('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv')

#Decompose the data
x = StandardScaler().fit_transform(featuresDf) #Normalize the data

#At this point our featuresDf has all of the raw data as well as an additional column called 'Experiment' which has the appropriate metadata
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x) #Transform the scaled data onto a new vector space
principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2']) #Create new dataframe with principal components as the data

#Append list of targets to principalDf
principalDf['target'] = targets

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

plotPCA(principalDf, ['CRC','CTR'], ['r','b'])