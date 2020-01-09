import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

rawData = np.genfromtxt("newJanuary.csv",names=True,dtype=(float,float,float,float,float),delimiter=",")
solidData = []

dayData = []
timeData = []
flowData = []
occuData = []
speedData = []
for i in range(0,len(rawData)):
    solidData.append(rawData[i].copy())
    if np.isnan(solidData[i][2]):
        solidData[i][2] = solidData[i - 1][2]
    if np.isnan(solidData[i][3]):
        solidData[i][3] = solidData[i - 1][3]
    if np.isnan(solidData[i][4]):
        solidData[i][4] = solidData[i - 1][4]

    dayData.append(solidData[i][0].copy())
    timeData.append(solidData[i][1].copy())
    flowData.append(solidData[i][2].copy())
    occuData.append(solidData[i][3].copy())
    speedData.append(solidData[i][4].copy())
        
for i in range(0,len(solidData)):
    solidData[i][0] = (solidData[i][0] - np.mean(dayData)) / np.std(dayData)
    solidData[i][1] = (solidData[i][1] - np.mean(timeData)) / np.std(timeData)
    solidData[i][2] = (solidData[i][2] - np.mean(flowData)) / np.std(flowData)
    solidData[i][3] = (solidData[i][3] - np.mean(occuData)) / np.std(occuData)
    solidData[i][4] = (solidData[i][4] - np.mean(speedData)) / np.std(speedData)
        
trainingData_Flow = []
trainingValue_Flow = []
for i in range(2,int(len(solidData))):
    temp = [solidData[i][0],solidData[i][1],solidData[i - 1][2],solidData[i][3],solidData[i][4]]
    trainingData_Flow.append(temp.copy())
    trainingValue_Flow.append(solidData[i][2])
npTrainingData_Flow = np.array(trainingData_Flow)
npTrainingValue_Flow = np.array(trainingValue_Flow)

trainingData_Occu = []
trainingValue_Occu = []
for i in range(2,int(len(solidData))):
    temp = [solidData[i][0],solidData[i][1],solidData[i - 1][3],solidData[i][2],solidData[i][4]]
    trainingData_Occu.append(temp.copy())
    trainingValue_Occu.append(solidData[i][3])
npTrainingData_Occu = np.array(trainingData_Occu)
npTrainingValue_Occu = np.array(trainingValue_Occu)

trainingData_Speed = []
trainingValue_Speed = []
for i in range(2,int(len(solidData))):
    temp = [solidData[i][0],solidData[i][1],solidData[i - 1][4],solidData[i][2],solidData[i][3]]
    trainingData_Speed.append(temp.copy())
    trainingValue_Speed.append(solidData[i][4])
npTrainingData_Speed = np.array(trainingData_Speed)
npTrainingValue_Speed = np.array(trainingValue_Speed)

with tf.device('/CPU:0'):

    modelFlow = Sequential()
    modelFlow.add(Dense(units=32, input_dim=5, kernel_initializer='normal', activation='relu'))
    modelFlow.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelFlow.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelFlow.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
    modelFlow.compile(loss='MAPE',optimizer='adam',metrics=['accuracy'])
    modelFlow.fit(npTrainingData_Flow,npTrainingValue_Flow,epochs=100,batch_size=10)

    modelOccu = Sequential()
    modelOccu.add(Dense(units=32, input_dim=5, kernel_initializer='normal', activation='relu'))
    modelOccu.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelOccu.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelOccu.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
    modelOccu.compile(loss='MAPE',optimizer='adam',metrics=['accuracy'])
    modelOccu.fit(npTrainingData_Occu,npTrainingValue_Occu,epochs=100,batch_size=10)

    modelSpeed = Sequential()
    modelSpeed.add(Dense(units=32, input_dim=5, kernel_initializer='normal', activation='relu'))
    modelSpeed.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelSpeed.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelSpeed.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
    modelSpeed.compile(loss='MAPE',optimizer='adam',metrics=['accuracy'])
    modelSpeed.fit(npTrainingData_Speed,npTrainingValue_Speed,epochs=100,batch_size=10)


    file = open("ans.txt","w")
    file.write("0.0267\n-1\n")
    testData = []
    for i in range(2,len(rawData)):
        if np.isnan(rawData[i][2]):
            temp = [solidData[i][0],solidData[i][1],solidData[i - 1][2],solidData[i][3],solidData[i][4]]
            nptemp = np.array([temp])
            result = modelFlow.predict(nptemp)
            result = result * np.std(flowData) + np.mean(flowData)
            file.write(str(result[0][0]) + "\n")
            rawData[i][2] = result
        elif np.isnan(rawData[i][3]):
            temp = [solidData[i][0],solidData[i][1],solidData[i - 1][3],solidData[i][2],solidData[i][4]]
            nptemp = np.array([temp])
            result = modelOccu.predict(nptemp)
            result = result * np.std(occuData) + np.mean(occuData)
            file.write(str(result[0][0]) + "\n")
            rawData[i][3] = result
        elif np.isnan(rawData[i][4]):
            temp = [solidData[i][0],solidData[i][1],solidData[i - 1][4],solidData[i][2],solidData[i][3]]
            nptemp = np.array([temp])
            result = modelSpeed.predict(nptemp)
            result = result * np.std(speedData) + np.mean(speedData)
            file.write(str(result[0][0]) + "\n")
            rawData[i][4] = result
        else :
            file.write("-1\n")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
rawData = np.genfromtxt("newMarch.csv",names=True,dtype=(float,float,float,float,float),delimiter=",")
solidData = []

dayData = []
timeData = []
flowData = []
occuData = []
speedData = []
for i in range(0,len(rawData)):
    solidData.append(rawData[i].copy())
    if np.isnan(solidData[i][2]):
        solidData[i][2] = solidData[i - 1][2]
    if np.isnan(solidData[i][3]):
        solidData[i][3] = solidData[i - 1][3]
    if np.isnan(solidData[i][4]):
        solidData[i][4] = solidData[i - 1][4]

    dayData.append(solidData[i][0].copy())
    timeData.append(solidData[i][1].copy())
    if not np.isnan(solidData[i][2]):
        flowData.append(solidData[i][2].copy())
    if not np.isnan(solidData[i][3]):
        occuData.append(solidData[i][3].copy())
    if not np.isnan(solidData[i][4]):
        speedData.append(solidData[i][4].copy())

for i in range(0,len(solidData)):
    solidData[i][0] = (solidData[i][0] - np.mean(dayData)) / np.std(dayData)
    solidData[i][1] = (solidData[i][1] - np.mean(timeData)) / np.std(timeData)
    solidData[i][2] = (solidData[i][2] - np.mean(flowData)) / np.std(flowData)
    solidData[i][3] = (solidData[i][3] - np.mean(occuData)) / np.std(occuData)
    solidData[i][4] = (solidData[i][4] - np.mean(speedData)) / np.std(speedData)

trainingData_Flow = []
trainingValue_Flow = []
for i in range(2,int(len(solidData))):
    temp = [solidData[i][0],solidData[i][1],solidData[i - 1][2],solidData[i][3],solidData[i][4]]
    trainingData_Flow.append(temp.copy())
    trainingValue_Flow.append(solidData[i][2])
npTrainingData_Flow = np.array(trainingData_Flow)
npTrainingValue_Flow = np.array(trainingValue_Flow)

trainingData_Occu = []
trainingValue_Occu = []
for i in range(2,int(len(solidData))):
    temp = [solidData[i][0],solidData[i][1],solidData[i - 1][3],solidData[i][2],solidData[i][4]]
    trainingData_Occu.append(temp.copy())
    trainingValue_Occu.append(solidData[i][3])
npTrainingData_Occu = np.array(trainingData_Occu)
npTrainingValue_Occu = np.array(trainingValue_Occu)

trainingData_Speed = []
trainingValue_Speed = []
for i in range(2,int(len(solidData))):
    temp = [solidData[i][0],solidData[i][1],solidData[i - 1][4],solidData[i][2],solidData[i][3]]
    trainingData_Speed.append(temp.copy())
    trainingValue_Speed.append(solidData[i][4])
npTrainingData_Speed = np.array(trainingData_Speed)
npTrainingValue_Speed = np.array(trainingValue_Speed)

with tf.device('/CPU:0'):

    modelFlow = Sequential()
    modelFlow.add(Dense(units=32, input_dim=5, kernel_initializer='normal', activation='relu'))
    modelFlow.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelFlow.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelFlow.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
    modelFlow.compile(loss='MAPE',optimizer='adam',metrics=['accuracy'])
    modelFlow.fit(npTrainingData_Flow,npTrainingValue_Flow,epochs=100,batch_size=10)

    modelOccu = Sequential()
    modelOccu.add(Dense(units=32, input_dim=5, kernel_initializer='normal', activation='relu'))
    modelOccu.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelOccu.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelOccu.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
    modelOccu.compile(loss='MAPE',optimizer='adam',metrics=['accuracy'])
    modelOccu.fit(npTrainingData_Occu,npTrainingValue_Occu,epochs=100,batch_size=10)

    modelSpeed = Sequential()
    modelSpeed.add(Dense(units=32, input_dim=5, kernel_initializer='normal', activation='relu'))
    modelSpeed.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelSpeed.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
    modelSpeed.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
    modelSpeed.compile(loss='MAPE',optimizer='adam',metrics=['accuracy'])
    modelSpeed.fit(npTrainingData_Speed,npTrainingValue_Speed,epochs=100,batch_size=10)

    

    file.write("510\n0.12\n66\n550\n0.11\n65.5\n600\n0.13\n65.7\n-1\n-1\n")
    testData = []
    for i in range(2,len(rawData)):
        if np.isnan(rawData[i][2]):
            temp = [solidData[i][0],solidData[i][1],solidData[i - 1][2],solidData[i][3],solidData[i][4]]
            nptemp = np.array([temp])
            result = modelFlow.predict(nptemp)
            result = result * np.std(flowData) + np.mean(flowData)
            file.write(str(result[0][0]) + "\n")
            rawData[i][2] = result
        elif np.isnan(rawData[i][3]):
            temp = [solidData[i][0],solidData[i][1],solidData[i - 1][3],solidData[i][2],solidData[i][4]]
            nptemp = np.array([temp])
            result = modelOccu.predict(nptemp)
            result = result * np.std(occuData) + np.mean(occuData)
            file.write(str(result[0][0]) + "\n")
            rawData[i][3] = result
        elif np.isnan(rawData[i][4]):
            temp = [solidData[i][0],solidData[i][1],solidData[i - 1][4],solidData[i][2],solidData[i][3]]
            nptemp = np.array([temp])
            result = modelSpeed.predict(nptemp)
            result = result * np.std(speedData) + np.mean(speedData)
            file.write(str(result[0][0]) + "\n")
            rawData[i][4] = result
        else :
            file.write("-1\n")
    file.close()

"""
510\n0.12\n66\n550\n0.11\n65.5\n600\n0.13\n65.7\n

"""