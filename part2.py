# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:26:56 2020

@author: cyy
"""

import load as ld
import nonNomalization as nnl
import training as tr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #learning rate and convergence
    learning_rate = pow(10,-4)
    convergence_norm = 0.5
    
    # Load data from csv file
    train_data = ld.loadData("./PA1_train.csv")
    # Load data from csv file
    test_data = ld.loadData("./PA1_dev.csv")
    # Nomalization training Data
    nl_train_data, y = nnl.dataNonNomalizaton(train_data)
    # Nomalization testing Data
    nl_test_data, yt = nnl.dataNonNomalizaton(test_data)
    
    wr,norm_record = tr.training(nl_train_data, y, learning_rate, convergence_norm)
    plt.plot(norm_record)
    plt.show()
    
    print("the weight:", wr )