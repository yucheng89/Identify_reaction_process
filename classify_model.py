import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm
import pandas as pd
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('originaldata_4_fix.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5000].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#import sklearn.naive_bayes as nb
#model = nb.GaussianNB()

#import sklearn.svm as svm
#model = svm.SVC(kernel='poly', degree=2)

#import sklearn.neighbors as sn
#model = sn.KNeighborsClassifier(n_neighbors=10,weights='distance')

#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

import joblib
#joblib.dump(model, "class_steps_IR_RF.pkl")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)