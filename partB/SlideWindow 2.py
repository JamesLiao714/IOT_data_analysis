import csv
import pandas as pd
import numpy as np

train = pd.read_csv("train_acc.csv")

test = pd.read_csv("test_acc.csv")

n = 3

k = 3

windowSize = n 


#data smooth
train['X'] = train['X'].rolling(n).mean().fillna(train['X']/n)
train['Y'] = train['Y'].rolling(n).mean().fillna(train['Y']/n)
train['Z'] = train['Z'].rolling(n).mean().fillna(train['Z']/n)

test['X'] = test['X'].rolling(n).mean().fillna(test['X']/n)
test['Y'] = test['Y'].rolling(n).mean().fillna(test['Y']/n)
test['Z'] = test['Z'].rolling(n).mean().fillna(test['Z']/n)

test['predict'] = None



#knn
for i in range(len(test)):
	#test test = i ~ i + windowsize
	windowTest = test[i:i+windowSize]
	min = []
	
	testX = windowTest['X'].to_numpy()
	testY = windowTest['Y'].to_numpy()
	testZ = windowTest['Z'].to_numpy()
	size = testX.size
	for j in range(windowSize - size):
		testX = np.append(testX,0)
		testZ = np.append(testZ,0)
		testY = np.append(testY,0)
	
	X, W, B = 0, 0, 0
	for j in range(len(train)):
		windowTrain = train[j:j+windowSize]
		
		trainX = windowTrain['X'].to_numpy()
		trainY = windowTrain['Y'].to_numpy()
		trainZ = windowTrain['Z'].to_numpy()
		
		
		size = trainX.size
		for j in range(windowSize - size):
			trainX = np.append(trainX,0)
			trainZ = np.append(trainZ,0)
			trainY = np.append(trainY,0)
			
		disX = np.abs(testX - trainX).sum()
		disY = np.abs(testY - trainY).sum()
		disZ = np.abs(testZ - trainZ).sum()
		dis = disX + disY + disZ
		min.append(dis)
		
	min = np.array(min)
	minIndex = min.argsort()[0:k]
	vote = train["act"][minIndex]
	for j in minIndex:
		if vote[j] == 'X':
			X += 1 / min[j]
		elif vote[j] == 'W':
			W += 1 / min[j]
		elif vote[j] == 'B':
			B += 1 / min[j]
	if X > B and X > W:
		test['predict'][i] = 'X'
	elif B > X and B > W:
		test['predict'][i] = 'B'
	elif W > B and W > X:
		test['predict'][i] = 'W'
	


#train.to_csv("10-27-AccelerometerLinear_{}.csv".format(n),index = False)
test.to_csv("10-28-AccelerometerLinear_{}_pre.csv".format(windowSize),index = False)