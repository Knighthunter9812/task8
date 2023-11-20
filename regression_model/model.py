import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv('D:\\task8\\regression_model\\winequality-red.csv')

df=df.drop_duplicates()


X=df.drop(['quality'],axis=1)
y=df['quality']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

from sklearn.svm import SVR
svc = SVR(kernel='rbf')
svc.fit(x_train, y_train)

pickle.dump(svc,open("model.pkl","wb"))
