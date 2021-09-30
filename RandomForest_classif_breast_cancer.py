# -*- coding: utf-8 -*-

# libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE

# read the file
path="F:/aegis/4 ml/dataset/supervised/classification/breast_cancer/wisconsin/bc.csv"

data=pd.read_csv(path)

data.head()
data.tail()
data.shape

# remove the unwanted features
data.drop(columns='code',inplace=True)

# check the data types
data.dtypes

# identify the rows that have the ?
ndx = data[data.bn == "?"].index
print(len(ndx))

# use random numbers between 1-10 to replace ?
data.bn[ndx] = np.random.randint(1,11,len(ndx))

# verify the changes
data.bn[ndx]

# change the datatype of 'bn' into int64
data.bn = data.bn.astype(np.int64)
data.dtypes

# include the EDA code


# check the distribution of the y-variable
data["class"].value_counts()

# split the data into train and test
trainx,testx,trainy,testy = train_test_split(data.drop("class",1),
                                             data["class"],
                                             test_size=0.3)

trainx.shape,trainy.shape
testx.shape,testy.shape

# build the Random Forest model
len(trainx.columns)

'''
n_estimators: number of DT to build
max_features: features in tree
'''

m1 = RandomForestClassifier(n_estimators=50,max_features=2).fit(trainx,trainy)
m1.estimators_

# predict on the test data
p1 = m1.predict(testx)

# confusion matrix using cross tab method AND classification report
df1=pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(df1.actual,df1.predicted,margins=True)
print(classification_report(df1.actual,df1.predicted))

# Important features
# method 1
feat = pd.DataFrame({'feature':trainx.columns,
                     'score':m1.feature_importances_})

feat = feat.sort_values('score',ascending=False)
print(feat)

# method 2 : RFE
rfe = RFE(m1,n_features_to_select=6).fit(trainx,trainy)

feat = pd.DataFrame({'feature':trainx.columns,
                     'support':rfe.support_,
                     'rank':rfe.ranking_ })
feat = feat.sort_values('rank')
print(feat)

# next model
#   using the best features
#   HPT