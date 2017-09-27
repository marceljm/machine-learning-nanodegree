#!/usr/bin/python

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

# Decision Tree

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = classify(features_train, labels_train)

prettyPicture(clf, features_test, labels_test)

# Decision Tree Accurracy

def submitAccuracies():
  return {"acc":round(acc,3)}

from sklearn.metrics import accuracy_score

labels_pred = clf.predict(features_test)

acc = accuracy_score(labels_test, labels_pred)

print submitAccuracies()
