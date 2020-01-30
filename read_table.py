#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pickle
import tables


# In[5]:


h5_file = tables.open_file('../HD/eeg_data_temples2.h5')
K = 24
seg_size = int(K*K/2)
seg_size = 256

X_yes = []
num_nodes = 0
total_nodes = 0

for array in h5_file.walk_nodes("/", "CArray"):
    total_nodes += 1
    if len(array.attrs["seizures"]) > 0:
        num_nodes += 1
        data = array.read()
        y = data[:,-1]
        X = data[y > 0,:2]
        for ix in np.arange(seg_size, len(X), seg_size):
            X_yes.append(X[ix-seg_size:ix])

X_no = []
for array in h5_file.walk_nodes("/", "CArray"):
    if len(array.attrs["seizures"]) == 0:
        data = array.read()
        X = data[:,:2]
        ixs = np.random.randint(seg_size, len(data), size=int(len(X_yes) / (total_nodes - num_nodes)))
        for ix in ixs:
            X_no.append(X[ix-seg_size:ix,:])
h5_file.close()
X_yes = np.array(X_yes)
X_no = np.array(X_no)
print(len(X_yes))
print(len(X_no))
#with open('CNN_' + str(K) + '.pickle', 'wb') as f:
#    pickle.dump([X_yes, X_no], f)

with open(str(seg_size) + '.pickle', 'wb') as f:
    pickle.dump([X_yes, X_no], f)

# In[ ]:
