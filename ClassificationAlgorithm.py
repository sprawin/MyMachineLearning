#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Split the data into Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

#Scaling Data
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
X_train = scx.fit_transform(X_train)
X_test = scx.transform(X_test)

#Importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Creating Instance for the models
rfClassifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 0)
dtClassifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
lSVClassifier = SVC(kernel = 'linear', random_state = 0)
kernelSVClassifier = SVC(kernel = 'rbf', random_state = 0)
lrClassifier = LogisticRegression()
knnClassifier = KNeighborsClassifier(n_neighbors = 5)
nbClassifier = GaussianNB()

#Training the Model
rfClassifier.fit(X_train,y_train)
dtClassifier.fit(X_train,y_train)
lSVClassifier.fit(X_train,y_train)
kernelSVClassifier.fit(X_train,y_train)
lrClassifier.fit(X_train,y_train)
knnClassifier.fit(X_train,y_train)
nbClassifier.fit(X_train,y_train)

#Predicting the Test Set
y_pred_rf = rfClassifier.predict(X_test)
y_pred_dt = dtClassifier.predict(X_test)
y_pred_lSVC = lSVClassifier.predict(X_test)
y_pred_KSVC = kernelSVClassifier.predict(X_test)
y_pred_lr = lrClassifier.predict(X_test)
y_pred_knn = knnClassifier.predict(X_test)
y_pred_nb = nbClassifier.predict(X_test)

#Fetching Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
print('************** Accuracy Scores **************')
print('Random Forest      :\t',accuracy_score(y_test,y_pred_rf))
print('Decision Tree      :\t',accuracy_score(y_test,y_pred_dt))
print('Linear SVC         :\t',accuracy_score(y_test,y_pred_lSVC))
print('Kernel SVC         :\t',accuracy_score(y_test,y_pred_KSVC))
print('Logistic Regression:\t',accuracy_score(y_test,y_pred_lr))
print('KNN                :\t',accuracy_score(y_test,y_pred_knn))
print('Naive Bayes        :\t',accuracy_score(y_test,y_pred_nb))
