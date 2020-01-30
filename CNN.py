#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
import HDFunctions
import time
import sys
from sklearn.model_selection import train_test_split
from scipy import signal
import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from scipy import signal
from keras.utils import to_categorical


# In[2]:


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# In[5]:


K = 24 #24, 32, 46, 64
seg_size = 256

with open('CNN_' + str(K) + '.pickle', 'rb') as f:
    temp = pickle.load(f)
    chans_yes = temp[0]
    chans_no = temp[1]

#Comment if necessary
for i in range(0, len(chans_yes)):
    chans_yes[i,:,0] = butter_bandpass_filter(chans_yes[i,:,0], 0.5, 45, fs=256, order=3)
    chans_yes[i,:,1] = butter_bandpass_filter(chans_yes[i,:,1], 0.5, 45, fs=256, order=3)
for i in range(0, len(chans_no)):
    chans_no[i,:,0] = butter_bandpass_filter(chans_no[i,:,0], 0.5, 45, fs=256, order=3)
    chans_no[i,:,1] = butter_bandpass_filter(chans_no[i,:,1], 0.5, 45, fs=256, order=3)

X = np.concatenate((chans_no, chans_yes)).reshape(-1, K, K, 1)
X[np.isnan(X)] = 0
y = np.ones((len(X,)))
y[:len(chans_no)] = 0

testWindowCount = int(0.25*len(X))
testIntervalCount = 10
intervalLen = int(testWindowCount/testIntervalCount)
allIntervals = np.arange(0, int(len(X)/intervalLen))
testIntervals = np.random.choice(allIntervals, testIntervalCount, replace=False)
trainIntervals = [x for x in allIntervals if x not in testIntervals]
X_train, X_test, y_train, y_test = [], [], [], []
for index in trainIntervals:
    X_train.extend(X[index*intervalLen: index*intervalLen + intervalLen])
    y_train.extend(y[index*intervalLen: index*intervalLen + intervalLen])
for index in testIntervals:
    X_test.extend(X[index*intervalLen: index*intervalLen + intervalLen])
    y_test.extend(y[index*intervalLen: index*intervalLen + intervalLen])
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_validation = to_categorical(y_validation)

X_train = np.array(X_train)
X_validation = np.array(X_validation)
X_test = np.array(X_test)


# In[35]:


def DNNprediction(X_train, X_validation, X_test, y_train, y_validation, y_test, count=5, epoch=50, K=24):
    sensitivity = []
    specificity = []
    for i in range(0, count):
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(K,K,1)))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=20, epochs=epoch, verbose=0, validation_data=(X_validation, y_validation))
        #score = model.evaluate(X_test, y_test, verbose=0)

        #predicted = model.predict(X_test)
        #print(predicted[0])
        #print(type(predicted))

        print("See Power")
        predicted = []
        for elem in X_test:
            temp = (model.predict(np.array([elem])))[0]
            predicted.append(temp)
            #time.sleep(float(K*K)/512.)
        predicted = np.array(predicted)

        keras.backend.clear_session()
        predicted[predicted > 0.5] = 1
        predicted[predicted < 0.5] = 0
        y_test_ = y_test[:, 0]
        predicted = predicted[:, 0]
        tp = sum(y_test_*predicted)
        fn = sum(y_test_*(1-predicted))
        tn = sum((1-y_test_)*(1-predicted))
        fp = sum((1-y_test_)*predicted)
        sensitivity.append(tp/(tp+fn))
        specificity.append(tn/(tn+fp))
    return sensitivity, specificity


# In[36]:


result_arr = []
result_arr.append(DNNprediction(X_train, X_validation, X_test, y_train, y_validation, y_test, count=1, epoch=5, K=K))


# In[ ]:
