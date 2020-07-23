import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', out='data/filedump/Italian')
preprocess.decompose(path='data/VogtmannE_2016.metaphlan_bugs_list.stool.tsv', out= 'data/filedump/USA')

dfList = preprocess.standardPreprocess('data/filedump')

#Declare feature dfs
X_italian = dfList[1]
X_usa = dfList[0]

print(X_italian)
print(X_usa)

#Preprocess Italian targets
italianDf = pd.read_csv('data/ThomasAM_2018a.metaphlan_bugs_list.stool.tsv', sep='\t')

#Mark unnecessary columns and append to targets list
Y_italian = [x for x in italianDf.iloc[3, :].tolist()]
Y_italian.pop(0)

#Clean up Italian targets
for index in range(len(Y_italian)):
	if Y_italian[index] == 'adenoma':
		X_italian = X_italian.drop(italianDf.columns.tolist()[index], axis=0)
		continue

Y_italian = [x for x in Y_italian if x != 'adenoma']


#Preprocess USA targets
usaDf = pd.read_csv('data/VogtmannE_2016.metaphlan_bugs_list.stool.tsv', sep='\t')

Y_usa = usaDf.iloc[3, :].tolist()
Y_usa.pop(0)


for index in range(len(Y_usa)):
	if Y_usa[index] != 'CRC' and Y_usa[index] != 'control':
		X_usa = X_usa.drop(usaDf.columns.tolist()[index], axis=0)
		continue

Y_usa = [x for x in Y_usa if x == 'CRC' or x == 'control']

#Classifier
ml = ML()
#ml.logisticRegeression(X_italian, X_usa, Y_italian, Y_usa)
#ml.logisticRegeression(X_usa, X_italian, Y_usa, Y_italian)


