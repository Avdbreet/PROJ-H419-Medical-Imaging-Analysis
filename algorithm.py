#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:41:56 2020

@author: Arthur
"""

""" Deep learning algorithm"""
""" 
Classification algorithm using SkLearn

Won't work, we're not into image analysis! But nice exercice, have to split the train_labels dataset

Algorithm: KNeigborsClassifier

"""
import matplotlib.pyplot as plt
import glob, pylab, pandas as pd
import pydicom, numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


#################################################################################
################# KNeighbors classifier ########################################
#################################################################################

"""
train_labels=pd.read_csv('/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
#test_labels=pd.read_csv('/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_test_labels.csv')

print(train_labels.info())
print('\n')

### Split into a train and a test datasets ####

y=train_labels.Target.values # same than train_labels['Target'].value  
X=train_labels.drop('Target', axis=1).values
print()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
"""
""" Problem with the ID which is not a float... Other kinf of problem, no sens to do it here"""

import keras
from keras.layers import Dense
from keras.models import Sequential

""" PRoblem:  no module named keras
tried: conda installl keras in terminall
"""
#################################################################################
################# Keras type model construction #################################
#################################################################################

""" Typical case of a keras model prevision :
    
    We have a df: that we separate into predictors(n columns) and target (1 column)
    
    model=Sequential()  # Set up the model
    
    n_cols= predictors.shape[1]
    
    model.add(Dense(50, activation='relu', input_shape=(784,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X, y, validation_split=0.3)
    
    predictions = model.predict(pred_data)

"""