import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial.distance import pdist, squareform #scipy spatial distance
import sklearn as sk
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU
from keras import metrics
from keras import backend as K
import time
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
import pylab
import joblib
from xgboost import XGBClassifier
filename = 'XGB_model.sav'
loaded_model = joblib.load(filename)

cols = ["attitude.roll","attitude.pitch","attitude.yaw","gravity.x","gravity.y","gravity.z","rotationRate.x","rotationRate.y","rotationRate.z","userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]

# convert folders to class labels
# downstairs/upstairs = 0,walking/jogging  = 1, standing/sitting = 2
class_translate = {"dws_1" : 0, "dws_2" : 0, "dws_11" : 0,  \
                   "ups_3" : 1, "ups_4" : 1, "ups_12" : 1, \
                   "wlk_7" : 2, "wlk_8" : 2, "wlk_15" : 2, \
                   "jog_9" : 3, "jog_16" : 3, \
                   "std_6" : 4, "std_14" : 4, \
                   "sit_5" : 5, "sit_13": 5}

#pre allocate arrays

print("motion prediction using model: ")
for p in range(1, 25):
	c = 0
	for i in class_translate.keys():    
		temp = pd.read_csv("A_DeviceMotion_data/A_DeviceMotion_data/" + i +"/" + "sub_" + str(p) + ".csv")
		temp['lable'] = class_translate[i]
		if c==0:
			test2 = temp
		else:
			test2 = test2.append(temp)
		c+=1
	test2.dropna()
	test2_x = test2.iloc[:, 1:-1].values
	test2_y = test2.iloc[:, -1].values

	pred = loaded_model.predict(test2_x)
	print('participant' + str(p) + '-> test result' + str(loaded_model.score(test2_x, test2_y)))

