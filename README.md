# Identifying-CRC-biomarkers
Performed multiple analyses to show that CRC biomarkers remain consistent across geography and ethnicity despite differing baseline microbiome compositions.

All of the tests I ran to support my hypothesis can be found in the "trials" folder. Each trial has its own preprocessing and ML scripts. These scripts come with plenty of analysis and visualization tools baked in already, so there is no need to re-write any of the code I wrote for this project. Please refer to the docs for a comprehensive understanding of the code.

Quick start:

If one would like to work in an enviornment with all of the data imported and preprocessed, I recommend working in the "trials/aggregate_datasets/Europe + East Asia/european_eastasian_classifier.py" file. This file contains plenty of pre-written code which can be uncommented and modified depending on what you would like to test. For example, if you wanted to run a classification task where you cross-validate within the Chinese dataset, you would do the following:

1) Uncomment the train test split line
2) Uncomment the classifier method you would like to use (randomForest() or logisticRegression())
3) Pass the datasets into the method (if you wanted to use random forest then it would be: randomForest(X_train, X_test, Y_train, Y_test))
4) Run the code

If you have any questions feel free to PM me.


