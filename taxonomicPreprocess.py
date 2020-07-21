import pandas as pd
import os

class preprocess:

	#Goes through a directory filled with prime-formatted files and converts them into a standard format
	def primeFormatToTaxonomic(self, directory):

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
		for subdir, dirs, files in os.walk(directory):
			for file in files:
				filepath = subdir + os.sep + file

				if file == '.DS_Store':
					continue

				#File should always be a tsv
				df = pd.read_csv(filepath, sep='\t', engine='python') #Import files into dataframe assuming 'tab' is the separator
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
		sampleNames = [x for x in fileNames if x!= '.DS_Store']
		for count in range(len(sampleNames)):
			sampleNames[count] = sampleNames[count].split('_bugs',1)[0]
		finalDf.index=sampleNames

		return finalDf

preprocess = preprocess()
df = preprocess.primeFormatToTaxonomic('taxonomic_profiles')
df.to_csv('new.csv')

