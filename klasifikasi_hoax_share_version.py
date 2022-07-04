import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.model_selection import *
import seaborn as sns
from keras import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.neighbors import *
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from tensorflow.keras.utils import *

train = pd.read_csv('data hoax.csv')
train.info()
x = train.drop(labels=['Berita','Hoax'], axis=1)
y = train['Hoax']

plt.style.use('dark_background') 
plt.figure(figsize=(10,5))
sns.countplot('Hoax',data = train, palette = 'inferno')
plt.title("Distribusi Data Hoax",fontsize = 15)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101)

print("Banyak data latih setelah dilakukan Train-Validation Split: ", len(X_train))
print("Banyak data uji setelah dilakukan Train-Validation Split: ", len(X_test))

lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)

print("Skor Akurasi :",accuracy_score(y_test, pred)*100)
print("Skor Presisi:",precision_score(y_test, pred)*100)
print("Skor Recall :",recall_score(y_test, pred)*100)

knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)

print("Skor Akurasi :",accuracy_score(y_test, prediction)*100)

svc = svm.SVC()
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)

print("Skor Akurasi :",accuracy_score(y_test, svc_pred)*100)
print("Skor Presisi:",precision_score(y_test, svc_pred)*100)
print("Skor Recall :",recall_score(y_test, svc_pred)*100)

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(knn,param_grid,verbose = 3)

test = pd.read_csv('testing.csv')
test.info()

X_t = test.drop(labels=['Berita','Hoax'], axis=1) #variabel fitur
y_t = test['Hoax'] #variabel kelas

test_prediction = knn.predict(X_t)

print(X_t)
print (test_prediction)

plt.style.use('dark_background') 
plt.figure(figsize=(10,5))
sns.countplot(test_prediction,data = train, palette = 'inferno')
plt.title("Distribusi Data Hoax",fontsize = 15)

test['Hoax'] = test_prediction
test.to_csv('result.csv')