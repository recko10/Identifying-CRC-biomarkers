import pandas as pd
import os

def primeFormatToTaxonomic():
	#directory = os.fsencode('taxonomic_profiles')
	directory = 'taxonomic_profiles'

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

		#Iterate therough species in microbes
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
		
		#print(df['Microbes'])
		#Find which species were not present in a given file and store them in a list
		for key in speciesToWeights:
			if len(speciesToWeights[key]) < numberOfLoops:
				speciesToWeights[key].append(0) #Append 0s to species who were not present in a given file

		if numberOfLoops == len(os.listdir(directory)):
			break
		numberOfLoops+=1
		index = 0

	print(speciesToWeights)
	#Create dataframe from the species : weights dictionary
	finalDf = pd.DataFrame.from_dict(speciesToWeights)
	return finalDf

finalDf = primeFormatToTaxonomic()
#finalDf.to_csv('results.csv')
##BUG: Make sure to add 0 to the unaffected keys to make all the columns the same size