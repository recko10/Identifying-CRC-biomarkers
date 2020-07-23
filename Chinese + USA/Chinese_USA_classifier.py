import pandas as pd
from taxonomicPreprocess import *
from taxonomicML import *

#Preprocess features
preprocess = preprocess()

preprocess.decompose(path='data/VogtmannE_2016.metaphlan_bugs_list.stool.tsv', out='data/filedump/USA')
preprocess.decompose(path='data/YuJ_2015.metaphlan_bugs_list.stool.tsv', out='data/filedump/Chinese')

dfList = preprocess.standardPreprocess('data/filedump')
X_usa = dfList[0]
X_chinese = dfList[1]

#Preprocess Chinese targets
chineseDf = pd.read_csv('data/YuJ_2015.metaphlan_bugs_list.stool.tsv', sep='\t')

#Extract Chinese targets
Y_chinese = chineseDf.iloc[3, :].tolist()
Y_chinese.pop(0)

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
#ml.logisticRegeression(X_usa, X_chinese, Y_usa, Y_chinese)
#ml.logisticRegeression(X_chinese, X_usa, Y_chinese, Y_usa)

#PCA
#ml.pca(X_usa, Y_usa)