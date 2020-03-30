# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:45:27 2019

@author: 91948
"""


# import modules
import numpy as np
import random
import time

# path to data set
path = 'House-votes-data.txt'


# clean and fill missing data
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
    return num_train


# get prior probability using the data set
def prior(num_train):
    total = 0
    republic = 0
    non_republic = 0
    for i in range(num_train.shape[0]):
        if num_train[i][num_train.shape[1]-1] == '1':
            republic += 1
        else:
            non_republic+=1
        total+=1
    return republic/total,non_republic/total


# use to predict using Bayes model
def BayesProb(num_train,testdata,col,Result = None) :
    totalcount = 0
    resultcount = 0
    for data in range(num_train.shape[0]) :
        if num_train[data][col] == testdata :
            if num_train[data][16] == Result :
                resultcount+=1
        if num_train[data][16] == Result :
            totalcount+=1
    #print(resultcount,totalcount)
    return [resultcount,totalcount]


# using 5 flod bagging
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

# function to fit the data set and caluculate results
def fit(num_train,num_test = None,show = True):
    acc = 0
    t_p = 0
    f_n = 0
    f_p = 0
    t_n = 0
    total = 0
    testing_num = None
    if type(num_test) == None:
        testing_num = num_train.copy()
    else: 
        testing_num = num_test
    start_time = time.time()
    for j in range(testing_num.shape[0]):
        num1,num2 = prior(num_train)
        for i in range(num_train.shape[1]-1) :
            listing = BayesProb(num_train,testing_num[j][i],i,'1')
            num1 = num1 *listing[0]
            listing = BayesProb(num_train,testing_num[j][i],i,'0')
            num2 = num2 *listing[0]
        #print(num2,num1)
        if num1 > num2 and num_train[j][-1:][0] == '1' and type(num_test) == None:
            acc +=1
            t_p += 1
        if num2 > num1  and num_train[j][-1:][0] == '0' and type(num_test) == None:
            acc += 1
            t_n += 1
        if num1 > num2 and num_train[j][-1:][0] == '0' and type(num_test) == None:
            f_p +=1
        if num2 > num1  and num_train[j][-1:][0] == '1' and type(num_test) == None:
            f_n +=1
            
            
            
        if num1 > num2 and testing_num[j][-1:][0] == '1' and not type(num_test) == None:
            acc +=1
            t_p += 1
        if num2 > num1  and testing_num[j][-1:][0] == '0' and not type(num_test) == None:
            acc += 1
            t_n += 1
        if num1 > num2 and testing_num[j][-1:][0] == '0' and not type(num_test) == None:
            f_p +=1
        if num2 > num1  and testing_num[j][-1:][0] == '1' and not type(num_test) == None:
            f_n +=1
            
        
        total += 1
    if show:
        print("Time for",testing_num.shape[0] if not type(num_test) == None else num_train.shape[0],"test:",round(time.time()-start_time,4))
        print("Final acc:",round(acc/total*100.0,3))
        return [acc/total*100.0,t_p*100/total,t_n*100/total,f_p*100/total,f_n*100/total]
    

# give the output and results of the model
def printMat(result):      
    acc = list(map(lambda x:x[0],result))
    t_p = list(map(lambda x:x[1],result))
    t_n = list(map(lambda x:x[2],result))
    f_p = list(map(lambda x:x[3],result))
    f_n = list(map(lambda x:x[4],result))
    
    acc = sum(acc)/len(acc)
    t_p = sum(t_p)/len(t_p)
    t_n = sum(t_n)/len(t_n)
    f_p = sum(f_p)/len(f_p)
    f_n = sum(f_n)/len(f_n)
    
    
    print("Avg acc :" ,acc)
    print("Avg true positive :" ,round(t_p,4),"%")
    print("Avg true positive :" ,round(t_n,4),"%")
    print("Avg false positive :" ,round(f_p,4),"%")
    print("Avg false negative :" ,round(f_n,4),"%")
    
    
    print("Precission:",round(t_p/(t_p+f_p),4))
    print("Recall",round(t_p/(t_p+f_n),4))
    print("F1 Score",round(2*((t_p/(t_p+f_p)*(t_p/(t_p+f_n)))/((t_p/(t_p+f_p))+(t_p/(t_p+f_n))))
    ,4))
    
    
    


# set up the data     
num_train = get_data(path)
result = list()
size_test = num_train.shape[0]*1/5
size_train = num_train.shape[0]*4/5
for i in range(5):
    result.append(fit(Bagging(num_train,size_train),Bagging(num_train,size_test)))
printMat(result)

