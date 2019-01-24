import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB



data = pd.read_csv("data.csv")

X = data.drop(['Gender','Dataset'], axis=1)

y = data['Dataset']

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

model = GaussianNB()  
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)  
  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test,y_pred))
