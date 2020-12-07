# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:13:48 2020

@author: cyy
"""

import matplotlib.pyplot as plt
import numpy as np
import csv

def loadData( fname  = "./PA1_train.csv" ):
    # load data from file
    data = []
    with open(fname) as csvfile:
        trainData_reader  = csv.reader(csvfile, delimiter=" ")
        line_count = 0
        for row in trainData_reader:
            if line_count != 0:
                line = ",".join(row).split(",")
                # spliting the time to day month year
                time = line[2].split("/")
                # remove ID
                del line[1]
                #remove time 
                del line[1]
                data.append(list(map(float,line + time)))
            line_count += 1
    return data

def dataNomalizaton(dataList):
    dataMatrix =  np.array(dataList)
    y = np.zeros(dataMatrix.shape[0]) + dataMatrix[:,19]
    data_min = dataMatrix.min(0)
    data_max = dataMatrix.max(0)
    
    dataMatrix[:,1:] = (dataMatrix[:,1:] - data_min[1:]) / (data_max[1:] - data_min[1:])
    
    return np.delete(np.delete(np.delete(dataMatrix, 19, 1), 16, 1), 15, 1), y

def training(nl_train_data, y, rate, convergence_norm = 0.5):
     weight = np.zeros(nl_train_data.shape[1])
     norm_record = []
     #while(1):
     for i in range(100000):
         gradient = gradient_helper(weight, nl_train_data, y)
         weight = weight - (rate * gradient)
         norm_gradient = np.linalg.norm(gradient)
         norm_record.append(norm_gradient)
         #print(gradient)
         #print(weight)
         #print(np.linalg.norm(gradient))
         #print(norm_gradient)
         if norm_gradient <= convergence_norm or np.isinf(norm_gradient):
             return weight, norm_record
     return weight, norm_record

def gradient_helper(w, x, y):   
    gradient = 0
    N = x.shape[0]
    for i in range(N):
        gradient += 2 * (np.dot(w, x[i]) - y[i]) * x[i]
    
    return gradient/N

if __name__ == "__main__":
    #learning rate and convergence
    learning_rate = pow(10,-1)
    convergence_norm = 0.5
    
    # Load data from csv file
    train_data = loadData("./PA1_train.csv")
    # Load data from csv file
    test_data = loadData("./PA1_dev.csv")
    # Nomalization training Data
    nl_train_data, y = dataNomalizaton(train_data)
    # Nomalization testing Data
    nl_test_data, yt = dataNomalizaton(test_data)
    
    wr,norm_record = training(nl_train_data, y, learning_rate, convergence_norm)
    plt.plot(norm_record)
    plt.show()
    print("the weight:", wr )
    
   """ 
    x1 = [1,2,3,4]
    y1 = [418.49, 406.93, 402.36, 402.21]
    plt.plot(x1, y1, label = "line 1")
    # line 2 points
    x2 = [1,2,3,4]
    y2 = [461.08, 450.29, 439.69, 434.77]
    # plotting the line 2 points 
    plt.plot(x2, y2, label = "line 2")
    plt.show()
  """  

    