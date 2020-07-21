import pandas as pd
import os

class preprocess:

	#Goes through a directory filled with prime-formatted files and converts them into a standard format
	def primeFormatToTaxonomic(self, directory):
		#directory = os.fsencode('taxonomic_profiles')

		speciesToWeights = {} #Dict that will have species as keys and a list taxonomic weights as values
		speciesNotPresent = []

		index = 0
		#Add all species in all files to dictionary
		for file in os.listdir(directory):
			if file == '.DS_Store':
				continue

			#File should always be a tsv
			df = pd.read_csv('taxonomic_profiles/' + file, sep='\t', engine='c') #Import files into dataframe assuming 'tab' is the separator
			#print(df)
			df.columns=['Microbes', 'Weights'] #Change column names

			#Iterate through species in microbes
			for species in df['Microbes']:
				if "s__" in species and "t__" not in species:
					#Log all species in dictionary
					if species not in speciesToWeights:
						speciesToWeights[species] = []
		
		numberOfLoops = 1

		#Append weights to dictionary
		for file in os.listdir(directory):
			
			if file == '.DS_Store':
				continue

			#File should always be a tsv
			df = pd.read_csv('taxonomic_profiles/' + file, sep='\t', engine='c') #Import files into dataframe assuming 'tab' is the separator
			#print(df)
			df.columns=['Microbes', 'Weights'] #Change column names

			for species in df['Microbes']:
				if "s__" in species and "t__" not in species:
					#Check if the current bacteria has already been logged or not
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
		finalDf.columns = [newHeaders]

		#Change indices
		sampleNames = os.listdir(directory)
		sampleNames.pop(0)
		for count in range(len(sampleNames)):
			sampleNames[count] = sampleNames[count].split('_bugs',1)[0]
		finalDf.index=sampleNames

		return finalDf

	#Goes through a file with both metadata and waterfall abundances in it and converts it to a standard format
	def curatedMetagenomicDataFormatToTaxonomic(self, path):

		#Preprocess data
		featuresDf = pd.read_csv(path, sep='\t') #Load in df

		#Changes indices
		featuresDf.index = featuresDf['Unnamed: 0'].tolist()
		featuresDf = featuresDf.drop(columns='Unnamed: 0')


		#Contains the targets
		targets = []
		#Parse through list, extract targets, delete metadata
		for index in featuresDf.index.tolist():

			#Create targets list
			if index == 'study_condition':
				for column in featuresDf.columns.tolist():
					if featuresDf.at[index, column] == 'control' or featuresDf.at[index, column] == 'CRC':
						targets.append(featuresDf.at[index,column])
						continue
					featuresDf = featuresDf.drop(column, axis=1) #Drop columns without appropriate metadata
				featuresDf = featuresDf.drop(index, axis=0) #Drop the "study_condition" row
				continue 

			#End deletion when you arrive at the k_Bacteria index
			if 's__' in index:
				break

			#Delete row
			featuresDf = featuresDf.drop(index, axis=0)

		#Delete all lower taxonomic levels
		for index in featuresDf.index.tolist():
			if 's__' not in index:
				featuresDf = featuresDf.drop(index, axis=0)

			if 't__' in index:
				featuresDf = featuresDf.drop(index, axis=0)

		#Transpose the dataframe
		featuresDf = featuresDf.T 

		#Changes all "controls" into "CTR"
		for index in range(len(targets)):
			if targets[index] == 'control':
				targets[index] = 'CTR'

		#Change headers
		newHeaders = [element.split('s__',1)[1] for element in featuresDf.columns.tolist()]

		featuresDf.columns = newHeaders

		return featuresDf, targets

preprocess = preprocess()
df = preprocess.primeFormatToTaxonomic('taxonomic_profiles')
print(df)
df.to_csv('results.csv')
# featuresDf, targets = preprocess.curatedMetagenomicDataFormatToTaxonomic('trial_1/data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv')
# print(featuresDf)
# print(targets)

