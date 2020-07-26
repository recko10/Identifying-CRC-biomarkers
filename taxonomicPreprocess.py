import pandas as pd
import os
import shutil

class preprocess:

	#Takes as input a curatedMetagenomicData abundance file/dataframe (with no modifications) and breaks it into separate files, each file representing 1 sample's abundances
	def decompose(self, path='', out='', dataframe=None):

		#Option to pass a dataframe as a parameter
		if dataframe != None:
			df = dataframe
		else:
			df = pd.read_csv(path, sep='\t')

		#Set indices
		df.index = df.iloc[:, 0].tolist()

		#Parse through list, extract targets, delete metadata
		for index in df.index.tolist():

			#End deletion when you arrive at the a bacteria index
			if '__' in index:
				break

			#Delete row
			df = df.drop(index, axis=0)

		#Drop unnamed column
		df = df.drop('Unnamed: 0', axis=1)

		#Make folder (titled with the file name without the metaphlan_bugs.stool.tsv) to store output files
		#folderPath = path.split('/')[-1].split('.')[0]
		
		folderPath=out 

		if os.path.exists(folderPath):
			shutil.rmtree(folderPath)
		os.makedirs(folderPath)
		
		#Create separate files
		for column in df.columns.tolist():
			df[column].to_csv(folderPath + os.sep + column + '.tsv', sep='\t')


	#Goes through directory with folders and returns multiple abundance dataframes all following the same superset and format
	def standardPreprocess(self, directory, keepFiles=False):

		speciesToWeights = {} #Dict that will have species as keys and a list taxonomic weights as values
		speciesNotPresent = []
		fileNames = []
		index = 0
		#Add all species in all files to dictionary
		for subdir, dirs, files in os.walk(directory):
			for file in files:
				filepath = subdir + os.sep + file
				if file == '.DS_Store':
					continue

				fileNames.append(file)
				#File should always be a tsv
				df = pd.read_csv(filepath, sep='\t', engine='python') #Import files into dataframe assuming 'tab' is the separator
				df.columns=['Microbes', 'Weights'] #Change column names
				#Iterate through species in microbes
				for species in df['Microbes']:
					if "s__" in species and "t__" not in species:
						#Log all species in dictionary
						if species not in speciesToWeights:
							speciesToWeights[species] = []

		numberOfLoops = 1
		subdirList = []
		#Append weights to dictionary
		for subdir, dirs, files in os.walk(directory):

			for file in files:
				filepath = subdir + os.sep + file

				if file == '.DS_Store':
					continue

				if subdir not in subdirList:
					subdirList.append(subdir)

				#File should always be a tsv
				df = pd.read_csv(filepath, sep='\t', engine='python') #Import files into dataframe assuming 'tab' is the separator
				df.columns=['Microbes', 'Weights'] #Change column names

				#Append weights to dictionary
				for species in df['Microbes']:
					if "s__" in species and "t__" not in species:
						speciesToWeights[species].append(float(df.at[index, 'Weights']))
					index+=1
				
				#Find which species were not present in a given file 
				for key in speciesToWeights:
					if len(speciesToWeights[key]) < numberOfLoops: #If a given key-value pair is not the expected length for this iteration
						speciesToWeights[key].append(0) #Append 0s to species who were not present in a given file

				numberOfLoops+=1
				index = 0


		#Create dataframe from the species : weights dictionary
		finalDf = pd.DataFrame.from_dict(speciesToWeights)

		#Change headers
		newHeaders = [key for key in speciesToWeights]
		for count in range(len(newHeaders)):
			newHeaders[count] = newHeaders[count].split('s__', 1)[1]
		finalDf.columns = newHeaders

		#Change indices
		sampleNames = [x.split('.',1)[0] for x in fileNames if x!= '.DS_Store']
		for count in range(len(sampleNames)):
			sampleNames[count] = sampleNames[count].split('_bugs',1)[0]
		finalDf.index=sampleNames

		#Split dataframe
		dfList = []
		subdirFileCount=0
		runOnce = False
		for subdir in subdirList:
			
			previous=subdirFileCount
			subdirFileCount=0

			for file in os.listdir(subdir):
				if file == '.DS_Store':
					continue
				subdirFileCount+=1

			if runOnce == False:
				dfList.append(finalDf.iloc[:subdirFileCount,:])
			if runOnce == True:
				dfList.append(finalDf.iloc[previous:previous+subdirFileCount, :])

			runOnce = True

		#Remove folder if user desires
		if keepFiles == False:
			if os.path.exists(directory):
				shutil.rmtree(directory)

		return dfList

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

# preprocess = preprocess()
# df = preprocess.decompose(path='trial_1/data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out = 'nani')

# dfList = preprocess.standardPreprocess('nani')
# print(dfList[0])


