# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 01:09:38 2020

@author: cyy
"""
import nomalization as nl
import dataStatistics as st
import load as ld
import csv

if __name__ == "__main__":
    # Load data from csv file
    train_data = ld.loadData("./PA1_train.csv")
    # Output the statistics result of train data
    st.statisticsData(train_data, "./part0_numerical_statistic.csv", "./part0_categorical _percentage.csv")
    # Nomalization training Data
    nl_train_data, y = nl.dataNomalizaton(train_data)
    with open("./part0_nomalizaed_data.csv",'w', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter='\n')
        fileWriter.writerow(nl_train_data)