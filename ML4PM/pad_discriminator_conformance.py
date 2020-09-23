# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:44:26 2020

@author: u0132580
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:12:54 2020

@author: Jari Peeperkorn



FUNCTIONS NEED TO BE CHECKED
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SimpleRNN, LSTM, Dropout, Bidirectional
import keras
from keras.preprocessing import sequence

import copy
import random


def get_voc(log1, log2):
    log = log1 + log2
    outputset1 = set([])
    #outputset2 = set([]) #for voc with only log2 for creating antilog
    for trace in log:
        for act in trace:
            outputset1.add(act)
    voc = list(outputset1)
    return voc
    

'''
def add_dimensionact(dumlog): #adds an extra dimension, each action becomes list with one element
    newlog = []
    for trace in dumlog:
        newtrace = [[el] for el in trace]
        newlog.append(newtrace)
    return(newlog)
    
'''
def add_noise_name(logdummy, voc):
    changelog = []
    for i in range(0, len(logdummy)):
        trace = copy.deepcopy(logdummy[i])
        random_act = random.randint(0, len(logdummy[i]) - 1)
        new = random.choice(voc)
        #print(new)
        trace[random_act] = new
        changelog.append(trace)
    return changelog

def add_noise_order(logdummy):
    changelog = []
    for i in range(0, len(logdummy)):
        trace = copy.deepcopy(logdummy[i])
        random_act = random.randint(0, len(logdummy[i]) - 1)
        random_act2 = random.randint(0, len(logdummy[i]) - 1)
        first = copy.copy(logdummy[i][random_act])
        second = copy.copy(logdummy[i][random_act2])
        #print(first, second)
        trace[random_act] = second
        trace[random_act2] = first
        changelog.append(trace)
    return changelog
    

#SHOULD CHECK IF WE WANT TO DO THIS!!!!!!
def delete_not_new(log, antilog):
    unique = []
    for trace in log:
        if trace not in unique:
            unique.append(trace)
        else:
            continue
    #print(unique)
    new_antilog = []
    for trace in antilog:
        if trace not in unique:
            new_antilog.append(trace)
        else:
            continue
    return(new_antilog)


def get_antilog_noise(logdummy, voc):
    l = copy.deepcopy(logdummy)
    l1 = add_noise_name(l, voc)
    l2 = add_noise_order(l1)
    antilog = delete_not_new(logdummy, l2)
    return(antilog)
    
def get_antilog_random(log, voc):
    min_length = len(min(log, key=len))
    max_length = len(max(log, key=len)) #now we just take random between min and max, should probably be changed to normal distribution
    length_antilog = len(log) #for size antilog = size log, but can be altered
    antilog = []
    for i in range(0, length_antilog):
        size = random.randint(min_length,max_length)
        antitrace = []
        for j in range(0, size):
            index = random.randint(0,len(voc)-1)
            antitrace.append(voc[index])
        antilog.append(antitrace)
    return(antilog)
'''
def padding(dumlog, max_length, pad):
    log = copy.deepcopy(dumlog)
    newlog =  []
    for trace in log:
        if len(trace)>max_length:
            while len(trace)>max_length:
                trace.pop()
        elif len(trace)<max_length:   
            while len(trace)<max_length:
                trace.append(pad)
        newlog.append(trace)
    return newlog
'''
def get_batches(log): #get same sized batches in list of numpy multidim arrays
    #log is log WITH label!
    min_length = len(min([item[0] for item in log], key=len))
    max_length = len(max([item[0] for item in log], key=len))
    print(min_length, max_length)
    Xbatches = []
    Ybatches = []
    for size in range(min_length, max_length+1):
        print(size)
        listX = []
        listY = []
        for i in range(0, len(log)):
            if len(log[i][0]) == size:
                listX.append(log[i][0])
                listY.append(log[i][1])
        if len(listY) == 0:
            print(size, "No traces")
            continue
        arrayX = np.array(listX)
        #arrayX = np.expand_dims(arrayX, axis=2)
        #print(arrayX)
        arrayY = np.array(listY)
        Xbatches.append(arrayX)
        Ybatches.append(arrayY)   
    return(Xbatches, Ybatches)
        
    
    
def get_values(dumlog1, dumlog2, dumantilog, voc):
    
    maximumlength = len(max(dumlog2, key=len))
    
    log1 = copy.deepcopy(dumlog1)
    log2 = copy.deepcopy(dumlog2)
    antilog = copy.deepcopy(dumantilog)
    
    #log1 = padding(dumlog1, maximumlength, "Z")
    #log2 = padding(dumlog2, maximumlength, "Z")
    #antilog = padding(dumantilog, maximumlength, "Z")
    

    #voc.append("Z")
    label_encoder = LabelEncoder() #label encoder 
    label_encoder.fit(voc)
    
    #list of arrays (traces) with encoded activities as numpy array
    trainlog = []
    for i in range(len(log2)):
        dummytrace = label_encoder.transform(log2[i])
        trace = dummytrace.tolist()
        trainlog.append([trace,1]) #1 = label true trace
    for i in range(len(antilog)):
        dummytrace = label_encoder.transform(antilog[i])
        trace = dummytrace.tolist()
        trainlog.append([trace,0]) #0 = label antitrace
    testlog = []
    for trace in log1:
        dummytrace = label_encoder.transform(trace)
        testlog.append(dummytrace)
    random.shuffle(trainlog) #shuffle true and antitraces
    #X, Y = get_batches(trainlog)
    
    X = []
    Y = []
    
    for trace in trainlog:
        X.append(trace[0])
        Y.append(trace[1])
        
    X = sequence.pad_sequences(X, maxlen=maximumlength)
    X = np.array(X)
    Y = np.array(Y)
    testlog = sequence.pad_sequences(testlog, maxlen=maximumlength)
    testlog = np.array(testlog)
    print("Fixed preprocessing")
    
    model = Sequential()
    model.add(Embedding(input_dim=len(voc),output_dim=4, input_length=maximumlength))
    #model.add(LSTM(units=32))
    model.add(Bidirectional(SimpleRNN(units=8)))
    #model.add(Bidirectional(LSTM(units=8)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
    hist = model.fit(X, Y, batch_size=64, epochs=40, verbose = 0, validation_split = 0.2)
    print("training done")
    '''
    squeezed = []
    #where = []
    for i in range(0, len(testlog)):
        #print(testlog[i])
        squeezed.append(model.predict(testlog[i])[-1][0])
        #print(model.predict(testlog[i]))
        #where.append(model.predict(testlog[i]))
    '''
    squeezed = model.predict(testlog)
    fit1 = np.average(squeezed)
    count_0 = 0
    count_1 = 0
    for pred in squeezed:
        if pred < 0.5:
            count_0 = count_0 + 1
        if pred >= 0.5:
            count_1 = count_1 + 1
    print("Amount predicted antitrace: ", count_0, ". Amount predicted true trace: ", count_1)
    fitness = count_1/(count_1 + count_0)
    return(fit1, fitness)


def get_distance(log1dum, log2dum, method): #we will take antilog on log2
    log1 = copy.deepcopy(log1dum)
    log2 = copy.deepcopy(log2dum)
    voc = get_voc(log1, log2)
    
    if method == 'random':
        print("Random:")
        antilog = get_antilog_random(log2, voc)
        print("Antilog obtained")
        fitness1, fitness2 = get_values(log1, log2, antilog, voc)
        print("fitness obtained")  
    elif method == 'noise':
        print("Noise on all traces, random replacement and random switching, each once, delete traces that are also in log")
        antilog = get_antilog_noise(log2, voc)
        print("Antilog obtained")
        fitness1, fitness2 = get_values(log1, log2, antilog, voc)
        print("fitness obtained") 
    else:
        print("Invalid input method, try random or noise")  
    return(fitness1, fitness2)

