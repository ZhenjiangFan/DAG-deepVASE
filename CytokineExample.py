#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import shutil
from DL.DNN.DNN import DNN;
from DL.FDR.FDR_control import FDR_control;
from causal.DegenerateGaussianScore import DegenerateGaussianScore
import networkx as nx
import scipy.stats



dataFolderPath = "data/sepsis";


newDataFileName="Down-sampled_pediatric_sepsis_data.txt";
#Load the X data and Y data
dataset = pd.read_csv(dataFolderPath+"/"+newDataFileName,sep='\t');


# import the MGM package
from MGM.MGM import MGM
# Initialize a MGM object
mgm = MGM();

mgmOutputFile = mgm.runMGM(dataFolderPath, newDataFileName,lambda_continuous_continuous = 0.3, lamda_continuous_discrete = 0.3, lamda_discrete_discrete = 0.3);

mgmOutputFilePath = mgmOutputFile[0];
print("MGM's output was saved as the following file:");
print(mgmOutputFilePath);
print("The likelihood values were saved as the following file:");
likelihoodFilePath = mgmOutputFile[1];
print(likelihoodFilePath);


colList = ["SIRS"];

#For each of the columns, run the deep learning feature selection procedure
#using it as the target variable and other variables as independent variables
for colName in colList:
    print(colName);
    #Clear the old DNN inportance results
    resultDir = dataFolderPath+"/DNN_result";
    if os.path.exists(resultDir):
        shutil.rmtree(resultDir);
    
    #Remove the old input data 
    XDataName = "XData.txt";
    XDataPath = dataFolderPath+"/"+XDataName;
    if os.path.exists(XDataPath):
        os.remove(XDataPath);
    #Create the input data using independent variables
    XDF = dataset[[col for col in dataset.columns if colName != col]];
    XDF.to_csv(XDataPath,index=None,sep="\t");
    #Create the output data using the current column as the target
    YDF = dataset[[colName]];
    #Remove the old knockoff data
    knockoffDataName = "XDataKnockoff.txt";
    knockoffDataPath = dataFolderPath+"/"+knockoffDataName;
    if os.path.exists(knockoffDataPath):
        os.remove(knockoffDataPath);
        
    #Create knockoff generator
    from DL.knockoff.KnockoffGenerator import KnockoffGenerator;
    generator = KnockoffGenerator();
    knockoffFilePath = generator.CholLuKnockoff(dataFolderPath, XDataName,sep="\t");
    
    print("The newly generated knockoff file is named as:");
    print(knockoffFilePath);
    
    # After generating the knockoff data, run DNN
    XKnockoffData = pd.read_csv(knockoffFilePath,sep="\t");
    print(XKnockoffData.shape);
    print(XDF.shape);
    print(YDF.shape);
    
    #Run deep neural network
    XValues = XKnockoffData.values;
    YValues = YDF.values;
    
    pNum = int(XValues.shape[1] / 2);
    nNum = XValues.shape[0];
    print(XValues.shape);
    print(YValues.shape);
    print(pNum);
    
    XOrigin = XValues[:, 0:pNum];
    XKnockoff = XValues[:, pNum:];

    X3DTrain = np.zeros((nNum, pNum, 2));
    X3DTrain[:, :, 0] = XOrigin;
    X3DTrain[:, :, 1] = XKnockoff;
    labelTrain = YValues;
    
    coeff = 0.05 * np.sqrt(2.0 * np.log(pNum) / nNum);
    numOutputs = YDF.shape[1];

    #Save the DNN output to the following directory.
    if not os.path.exists(resultDir):
        os.makedirs(resultDir);
    
    dnn = DNN();
    model = dnn.build_DNN(pNum, numOutputs, coeff);
    callback = DNN.Job_finish_Callback(resultDir,pNum);
    dnn.train_DNN(model, X3DTrain, labelTrain,callback);
    
    #Apply FDR control to DNN result
    control = FDR_control();
    selectedFeatures = control.controlFilter(dataFolderPath+"/"+XDataName, resultDir, offset=1, q=0.05);
    #Save the selected associations
    selectedAssociations = [];
    for ele in selectedFeatures:
        selectedAssociations.append({"Feature1":ele[0],"Feature2":colName});
        
    if not os.path.exists(dataFolderPath+"/DNNSelection"):
        os.makedirs(dataFolderPath+"/DNNSelection");
    pd.DataFrame(selectedAssociations).to_csv(dataFolderPath+"/DNNSelection/DNNSelectedAssociations_"+colName+".csv")
   





