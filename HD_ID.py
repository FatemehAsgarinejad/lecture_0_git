#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pickle
import HDFunctions
import time
import sys
from sklearn.model_selection import train_test_split
from scipy import signal


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


# In[29]:


def HDpediction(X_train, y_train, X_test, y_test, count=20, retrain_epoch=10, D=5000, level=400, seg_size=256):
    #reload(HDFunctions)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
    sensitivity = []
    specificity = []
    for i in range(0, count):
        if i == 0:
            model = HDFunctions.buildHDModel(X_train, y_train, X_validation, y_validation, X_test, y_test, D, level, 'data')
            #print('i == 0')
            #start = time.time()
            acc_test, acc_validation = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.validationHVs, model.validationLabels, model.testHVs, model.testLabels, retrain_epoch, SLEEP=0)
            #print(time.time() - start)
        else:
            #print('i == 1')
            start = time.time()
            print("See Power")
            acc_test = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.validationHVs, model.validationLabels, model.testHVs, model.testLabels, 0, float(0)/256.)
            print(time.time() - start)
        maxscore = 0

        if i == 0:
            for j in range(0, len( acc_validation)):
                if  acc_validation[j][0] > maxscore:
                    maxscore =  acc_validation[j][0]
                    predicted = np.array(acc_test[j][1])
        else:
            predicted = np.array(acc_test[0][1])
        tp = sum(y_test*predicted)
        fn = sum(y_test*(1-predicted))
        tn = sum((1-y_test)*(1-predicted))
        fp = sum((1-y_test)*predicted)
        sensitivity.append(tp/(tp+fn))
        specificity.append(tn/(tn+fp))
    return sensitivity, specificity


# In[6]:


#seg_size=int(sys.argv[1])
seg_size = 256
with open(str(seg_size) + '.pickle', 'rb') as f:
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

X = np.concatenate((chans_no, chans_yes)).reshape(-1,2*seg_size)
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


# In[30]:


result_arr = []
result_arr.append(HDpediction(X_train, y_train, X_test, y_test, count=3, retrain_epoch=10, D=5000, level=400, seg_size=seg_size))


# In[16]:





# In[ ]:
