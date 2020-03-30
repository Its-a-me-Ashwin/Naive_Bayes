# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:34:24 2019

@author: 91948
"""

#KNN
import math
import numpy as np
import matplotlib.pyplot as plt
import random

def manhatten(a,b):
    if a.shape != b.shape:
        return -1
    dist = 0
    for i in range(a.shape[0]):
        dist += abs(int(a[i])-int(b[i]))
    return dist

def euclidian(a,b):
    if a.shape != b.shape:
        return -1
    dist = 0
    for i in range(a.shape[0]):
        dist += math.sqrt(abs(int(a[i])**2 - int(b[i])**2))
    return dist


def Bagging (num_train,k = 0):
    bagged_num_train = list()
    i = 0
    while (i < int(k)):
        try:
            bagged_num_train.append(num_train[random.randint(0,num_train.shape[0])])
            i +=1
        except Exception as e:
            continue
            print(e)
    bagged_num_train = np.array(bagged_num_train)
    return bagged_num_train

path = 'House-votes-data.txt'
def get_data(path):
    file = open(path,'r')
    lines = file.read().split()
    input_data = list()
    output_data = list()
    num_train = list()
    
    for line in lines:
        attrs = line.split(',')
        num_train.append(attrs)
        output_data.append(attrs[len(attrs)-1])
        input_data.append(attrs[:-1])
    output_data = list(map(lambda x: 1 if x == 'republican' else 0, output_data ))
    
    
    num_train = np.array(num_train)
    for column in range(num_train.shape[1]):
        if column == 16:
            for i in range(num_train.shape[0]):
                num_train[i][column] = 1 if num_train[i][column] == 'republican' else 0
        n = list(np.char.count(num_train[:,column],'n')).count(1)
        y = list(np.char.count(num_train[:,column],'y')).count(1)
        for i in range(num_train.shape[0]):
            if num_train[i][column] == '?':
                num_train[i][column] = 'n' if n > y else 'y'
    #print(num_train)
    for column in range(num_train.shape[1]):
        if column == 16: continue
        for i in range(num_train.shape[0]):
            num_train[i][column] = 1 if num_train[i][column] == 'y' else 0
    #print(num_train)
    return num_train

def KNN(test_num,num_train,k,function):
    distance = list()
    for i in range(num_train.shape[0]):
        distance.append((function(test_num,num_train[i]),i))
    distance.sort(key = lambda x:x[0])
    top = list(map(lambda x:x[1],distance))[:k]
    out = list()
    for i in top:
        out.append(num_train[i][-1:][0])
    if out.count('0') < out.count('1'):
        return 1
    else:
        return 0
    
def plot_scatter(x,y,number_of_categories):
    from sklearn.decomposition import PCA
    pcs = PCA(n_components = 2)
    t_new = pcs.fit_transform(x)
    
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, number_of_categories))
    y_new = np.asarray(y,dtype = np.int16)
    colors = cmap[y_new]

    # Extract the x- and y-values.
    x_p = t_new[:, 0]
    y_p = t_new[:, 1]
    plt.scatter(x_p, y_p, color=colors)
    plt.show()
    
    
    
def get(path):
    file = open(path,'r')
    lines = file.read().split()
    input_data = list()
    output_data = list()
    num_train = list()
    
    for line in lines:
        attrs = line.split(',')
        num_train.append(attrs)
        output_data.append(attrs[len(attrs)-1])
        input_data.append(attrs[:-1])
    output_data = list(map(lambda x: 1 if x == 'republican' else 0, output_data ))
    
    
    num_train = np.array(num_train)
    for column in range(num_train.shape[1]):
        if column == 16:
            for i in range(num_train.shape[0]):
                num_train[i][column] = 1 if num_train[i][column] == 'republican' else 0
        n = list(np.char.count(num_train[:,column],'n')).count(1)
        y = list(np.char.count(num_train[:,column],'y')).count(1)
        for i in range(num_train.shape[0]):
            if num_train[i][column] == '?':
                num_train[i][column] = 'n' if n > y else 'y'
    return num_train
            
num = get(path)
#from tqdm import tqdm
def test ():    
    num_train = get_data(path)
    result = list()
    for k in range(20):
        acc = 0
        t_p = 0
        f_n = 0
        f_p = 0
        t_n = 0
        total = 0
        size_test = num_train.shape[0]*1/5
        size_train = num_train.shape[0]*4/5
        train = Bagging(num_train,size_train)
        test = Bagging(num_train,size_test)
        for i in range(num_train.shape[0]):    
            a = KNN(num_train[i],num_train,k,euclidian)
            if a == 0 and num_train[i][-1:][0] == '0':
                acc += 1
                t_n += 1
            if a == 1 and num_train[i][-1:][0] == '1':
                acc += 1
                t_p += 1
            if a == 0 and num_train[i][-1:][0] == '1':
                f_n += 1
            if a == 1 and num_train[i][-1:][0] == '0':
                f_p += 1
            total += 1
            #print("Final acc:",round(acc/total*100.0,3))
        result.append([acc/total*100.0,t_p*100/total,t_n*100/total,f_p*100/total,f_n*100/total])
        acc = list(map(lambda x:x[0],result))
        acc = sum(acc)/len(acc)
        #print("Avg acc :" ,acc)
    plotx = list(map(lambda x:x[0],result))
    print(plotx)
    plt.plot(list(range(20)),plotx)
    plt.xlabel("K value")
    plt.ylabel("Acc")
    plt.show()

test()



num_train = get_data(path)
numeric_train = list()
for i in range(num_train.shape[0]):
    numeric_train.append([])
    for j in range(num_train.shape[1]):
        x = None
        if num_train[i][j] == '1':
            x = 1
        elif num_train[i][j] == '0':
            x = 0
        else:
            x = 1
        numeric_train[i].append(x)
numeric_train = np.array(numeric_train)
plot_scatter(numeric_train,num[:,-1],2)
