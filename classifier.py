# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:52:39 2017

@author: insanesac
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:42:16 2017

@author: insane
"""
import pickle
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
exampleFile = open(r'D:\Profiledskin\shoes\train.csv')
exampleReader = csv.reader(exampleFile)
exampleData = list(exampleReader)
X= np.array(exampleData)
X = np.float32(X)

exampleFile1 = open(r'D:\Profiledskin\shoes\train_label.csv')
exampleReader1 = csv.reader(exampleFile1)
exampleData1 = list(exampleReader1)
Y = np.array(exampleData1)
Y = np.int32(Y)
y = np.ravel(Y)

exampleFile2 = open(r'D:\Profiledskin\shoes\test.csv')
exampleReader2 = csv.reader(exampleFile2)
exampleData2 = list(exampleReader2)
X1= np.array(exampleData2)
X1= np.float32(X1)

exampleFile3 = open(r'D:\Profiledskin\shoes\test_label.csv')
exampleReader3 = csv.reader(exampleFile3)
exampleData3 = list(exampleReader3)
Y1 = np.asarray(exampleData3)
Y1 = np.int16(Y1)
y1 = np.ravel(Y1)

#clf1 = RandomForestClassifier(n_estimators=2000, n_jobs=-1)
clf2 = ExtraTreesClassifier(n_estimators=2000, max_features=10,n_jobs=-1)

#clf1.fit(X, y)
clf2.fit(X, y)

#pred1 = clf1.predict(X1)
#acc = accuracy_score(y1,pred1)

pred2 = clf2.predict(X1)
acc2 = accuracy_score(y1,pred2)


filename = r'D:\Profiledskin\shoes\finalized_model.sav'
pickle.dump(clf2, open(filename, 'wb'))
 