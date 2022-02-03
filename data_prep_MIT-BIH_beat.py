import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import math
import pickle
from scipy import signal
import sklearn.preprocessing

def normalize_me(data):
    for k in range(len(data)):
        data[k, :] = sklearn.preprocessing.normalize(data[k, :].reshape(1, -1), norm='max', axis=1, copy=True, return_norm=False)
    return data

def split_data_to_chunks(k,data):
    return_data = []
    split_constant = 5 * 360
    lenght_of_data = k[1][1] - k[1][0]
    if lenght_of_data < split_constant:
        return
    a = math.floor(lenght_of_data / split_constant) * split_constant
    arrays = np.array_split(data[k[1][0]:k[1][0]+a], math.floor(a / split_constant))
    if k[0] == '(N':
        label_array = np.zeros((len(arrays),),dtype=int)
    else:
        label_array = np.ones((len(arrays),),dtype=int)
    return arrays, label_array

def give_annots(current_annot, data, limits):
    symbol = current_annot.symbol
    samples  = current_annot.sample
    # 120 points before 180 points after
    return_array = np.zeros((1,258))
    for i in range(0,len(symbol)):
        if samples[i] < 110 or samples[i] + 146 > len(data) or samples[i] > limits[-1] or samples[i] < limits[0]:
            continue
        else:
            if symbol[i] == 'N' or symbol[i] == 'L' or symbol[i] == 'R' or symbol[i] == 'e' or symbol[i] == 'j':
                new_beat = data[samples[i]-110:samples[i]+146]
                new_beat = np.append(new_beat,samples[i])
                new_beat = np.append(new_beat, 0)
                new_beat_reshape = np.reshape(new_beat, (1, 258))
                return_array = np.concatenate((return_array,new_beat_reshape))
            elif symbol[i] == 'A' or symbol[i] == 'a' or symbol[i] == 'J' or symbol[i] == 'S':
                new_beat = data[samples[i]-110:samples[i]+146]
                new_beat = np.append(new_beat,samples[i])
                new_beat = np.append(new_beat, 1)
                new_beat_reshape = np.reshape(new_beat, (1, 258))
                return_array = np.concatenate((return_array,new_beat_reshape))
            elif symbol[i] == 'V' or symbol[i] == 'E':
                new_beat = data[samples[i]-110:samples[i]+146]
                new_beat = np.append(new_beat,samples[i])
                new_beat = np.append(new_beat, 2)
                new_beat_reshape = np.reshape(new_beat, (1, 258))
                return_array = np.concatenate((return_array,new_beat_reshape))
            else:
                continue
    return_array = np.delete(return_array, 0, 0)
    return return_array

file1 = open('RECORDS.txt', 'r')
count = 0

# Using for loop
records = []
print("Using for loop")
for line in file1:
    count += 1
    records.append(line.strip())

file1.close()
training_dataset = ['100','101','103','105','106','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124']
patients_train_cnn = []
patients_valid_cnn = []
patients_test_cnn = []
limit = 360 * 60 * 5
positive_infinity = float('inf')
patients = []
patients_eval = []
cnn_train = []
for i in records:
    current_data = wfdb.rdsamp(i)
    current_annot = wfdb.rdann(i,extension='atr')
    if i in training_dataset:
        data = give_annots(current_annot, current_data[0][:, 0], [0, positive_infinity])
        data_train = normalize_me(data[len(data)//5:,0:256])
        data_train_label = data[len(data)//5:,-1].astype(int)
        data_valid = normalize_me(data[:len(data)//5,0:256])
        data_valid_label = data[:len(data)//5,-1].astype(int)
        patients_train_cnn.append([data_train, data_train_label])
        patients_valid_cnn.append([data_valid, data_valid_label])
    else:
        data_test = give_annots(current_annot, current_data[0][:, 0], [limit, positive_infinity])
        data_train = give_annots(current_annot, current_data[0][:, 0], [0,limit])

        data_train_2 = normalize_me(data_train[len(data_train)//5:,0:256])
        data_train_label = data_train[len(data_train)//5:,-1].astype(int)
        data_valid = normalize_me(data_train[:len(data_train)//5,0:256])
        data_valid_label = data_train[:len(data_train) // 5, -1].astype(int)
        patients_train_cnn.append([data_train_2, data_train_label])
        patients_valid_cnn.append([data_valid, data_valid_label])
        data_test_2 = normalize_me(data_test[:,0:256])
        data_test_label = data_test[:,-1].astype(int)
        patients_test_cnn.append([data_test_2, data_test_label])

X_train = np.delete(X_train,0,0)
y_train = y_train[1:]
X_test = np.delete(X_test,0,0)
y_test = y_test[1:]
# Preprocessing
sos = signal.butter(10,[0.8/180, 45/180],btype='bandpass', output='sos')
for i in range(len(X_train)):
    # filtered = signal.sosfiltfilt(sos, X_train[i,:])
    X_train[i,:] = sklearn.preprocessing.normalize(X_train[i,:].reshape(1,-1), norm='max', axis=1, copy=True, return_norm=False)
#
for i in range(len(X_test)):
    # filtered = signal.sosfiltfilt(sos, X_test[i,:])
    X_test[i,:] = sklearn.preprocessing.normalize(X_test[i,:].reshape(1,-1), norm='max', axis=1, copy=True, return_norm=False)

X_train = np.expand_dims(X_train,2)
X_test = np.expand_dims(X_test,2)

print('exit')