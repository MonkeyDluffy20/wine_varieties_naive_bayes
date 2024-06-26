# -*- coding: utf-8 -*-
"""NaiveBayes

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14QgF_MFwtmG2qGzMncJkZVil2HlqsjwE

#**"Cheers to Classification: Analyzing Wine Varieties with Naive Bayes"**

# Import Necessary Libraries
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

"""# Load the wine dataset"""

wine=datasets.load_wine()

wine

"""#Identifying Features and Labels in the Wine Dataset"""

print("Features:",wine.feature_names)
print("Labels:", wine.target_names)

"""# Split data into features (X) and target variable (y)"""

x=pd.DataFrame(wine["data"])
y=print(wine.target)

"""# Split the dataset into training and testing sets"""

x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.30,random_state=109)

"""# Initialize the Naive Bayes classifier (Gaussian Naive Bayes for continuous features)"""

gnb=GaussianNB()

"""# Train the Naive Bayes classifier"""

gnb.fit(x_train,y_train)

"""# Make predictions on the testing set"""

x_predict=gnb.predict(x_test)
print(x_predict)

"""# Evaluate the classifier"""

print("Accuracy:",metrics.accuracy_score(y_test,x_predict))
cm=np.array(confusion_matrix(y_test,x_predict))
print(cm)