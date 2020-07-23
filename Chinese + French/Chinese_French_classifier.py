import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', out='data/filedump/French')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')

X_french = dfList[0]
X_chinese = dfList[1]

#Select for only french samples
for index in X_french.index.tolist():
	if 'CCIS' not in index:
		X_french = X_french.drop(index, axis=0)

print(X_french)
print(X_chinese)


#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Extract Chinese targets
Y_chinese = chineseDf.iloc[3, :].tolist()
Y_chinese.pop(0)

#Preprocess French targets
frenchDf = pd.read_csv('data/ZellerG_2014.metaphlan_bugs_list.stool.tsv', sep='\t')

#Select for only french samples
for header in frenchDf.columns.tolist():
	if 'CCIS' not in header:
		frenchDf = frenchDf.drop(header, axis=1)


Y_french = frenchDf.iloc[3, :].tolist()
Y_french.pop(0)

for index in range(len(Y_french)):
	if Y_french[index] != 'CRC' and Y_french[index] != 'control':
		X_french = X_french.drop(frenchDf.columns.tolist()[index], axis=0)
		continue


Y_french = [x for x in Y_french if x == 'CRC' or x == 'control']


#Classifier
ml = ML()
ml.logisticRegeression(X_chinese, X_french, Y_chinese, Y_french)


