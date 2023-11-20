import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv('D:\\task8\\classification_model\\healthcare_dataset.csv')
#print(df.head())

df=df.drop(['Doctor','Hospital','Name','Room Number','Date of Admission','Discharge Date','Billing Amount'],axis=1)
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
cols = ['Gender','Blood Type','Medical Condition','Insurance Provider','Admission Type','Medication','Test Results']
for i in cols:
  df[i]=lc.fit_transform(df[i])


X=df.drop(['Test Results'],axis=1)
y=df['Test Results']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)

pickle.dump(knn,open("model.pkl","wb"))
