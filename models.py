import pandas as pd

#Preprocess data
features = pd.read_csv('sample_data/taxonomic_abundances.csv') #Load in df

indexList = [entry.split('[',1)[0] for entry in features['Unnamed: 0']] #List of bacteria names 

features.index = indexList #Change indices to values in this column
features = features.drop(columns='Unnamed: 0') #Drop the column

features = features.T #Transpose features (switch rows and columns and adjust values accordingly)
print(features)


