import pandas as pd
import os

#directory = os.fsencode('taxonomic_profiles')
directory = 'taxonomic_profiles'

speciesToWeights = {} #Dict that will have species as keys and a list taxonomic weights as values

index = 0
#Iterate through each file in the 'taxonomic_profiles' directory
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
			#Check if the bacteria has already been logged or not
			if species not in speciesToWeights:
				speciesToWeights[species] = [float(df.at[index, 'Weights'])]
			if species in speciesToWeights:
				speciesToWeights[species].append(df.at[index, 'Weights'])
		index+=1
	
	index=0
	continue
	
#Print out the resulting dictionarys
print(speciesToWeights)