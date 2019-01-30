#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:53:36 2019

@author: amit
"""
#%%
##                              SPEECH TRAIN
#                              train_models.py

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
import python_speech_features as pySpeech
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    features = pySpeech.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features

with open("maleaudios") as rf:
    filesM = [i.strip() for i in rf.readlines()]

with open("femaleaudios") as rf:
    filesF = [i.strip() for i in rf.readlines()]

featuresM = np.asarray(())
featuresF = np.asarray(())

# Featurization of Male audio files

for m in filesM[:100000]:
    try:
        sr,audio  = read(m)
        vectorM   = get_MFCC(sr,audio)
        if featuresM.size == 0:
            featuresM = vectorM
        else:
            featuresM = np.vstack((featuresM, vectorM))
    except Exception as e:
        print("Exception in file {}".format(m))
        
# Featurization of Female audio files

for f in filesF[:100000]:
    try:
        sr,audio  = read(m)
        vectorF   = get_MFCC(sr,audio)
        if featuresF.size == 0:
            featuresF = vectorF
        else:
            featuresF = np.vstack((featuresF, vectorF))
    except Exception as e:
        print("Exception in file {}".format(f))
        

# Creation of GMMs for Male and Female 

male_model = GaussianMixture(n_components = 8, max_iter = 500, covariance_type='diag', n_init = 3)
female_model = GaussianMixture(n_components = 8, max_iter = 500, covariance_type='diag', n_init = 3)
male_model.fit(featuresM)
female_model.fit(featuresF)



# Model saved as .gmm

cPickle.dump(male_model,open("male.gmm",'wb'))
cPickle.dump(female_model,open("female.gmm",'wb'))



#%%
###                                SPEECH TEST
#                                test_gender.py

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as pySpeech
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    features = pySpeech.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat;
    features = preprocessing.scale(features)
    return features

with open("maleaudios") as rf:
    filesM = [i.strip() for i in rf.readlines()]

with open("femaleaudios") as rf:
    filesF = [i.strip() for i in rf.readlines()]
    
modelpath  = "."

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]


predCorrect = []
for f in filesM[100000:]:
    try:
        
        sr, audio  = read(f)
        features   = get_MFCC(sr,audio)
        scores     = None
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(features))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        if winner == 0:
            predCorrect.append(True)
        else:
            predCorrect.append(False)
        print(f.split("/")[-1],"->", winner)
    except Exception as e:
        print("Exception in file{}".format(f))
        
for f in filesF[100000:]:
    
    sr, audio  = read(f)
    try:
        
        features   = get_MFCC(sr,audio)
        scores     = None
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(features))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        if winner == 1:
            predCorrect.append(True)
        else:
            predCorrect.append(False)
    except Exception as e:
        print("Exception in file {}".format(f))
        

correctCount = len([i for i in predCorrect if i is True])

print("accuracy = {}".format(correctCount/len(predCorrect)*100.0))
    
