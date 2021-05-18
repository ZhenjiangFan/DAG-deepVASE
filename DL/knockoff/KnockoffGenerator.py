import numpy as np
import pandas as pd
import scipy
import os;
os.environ['R_HOME'] = '/ihome/hpark/zhf16/.conda/envs/env36/lib/R';
import rpy2.robjects as robjects;

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, BatchNormalization, merge#keras.layers 
from keras import backend as K
from keras import regularizers

import deeplift
import deeplift.conversion.keras_conversion as kc
import deeplift.blobs as blobs
from deeplift.blobs import NonlinearMxtsMode
from deeplift.backend import function as compile_func
from collections import OrderedDict

class KnockoffGenerator:
    
    def __init__(self):
        print("__init__");
        
        
    def ISEE_knockoff(self, folder_path, file_name):
        #print("===");
        dataset = pd.read_csv(folder_path+os.path.sep+file_name);
        feature_list = dataset.columns.tolist();
        dataset.to_csv(file_name,index=False,header=False);

        # print(dataset.shape);
        knockoff_feature_list = [];
        for idx in range(0,dataset.shape[1]):
            knockoff_feature_list.append("K"+str(idx+1));
        print(feature_list);
        print(knockoff_feature_list);
        
        # load the R instance
        r = robjects.r;
        #Load the R script
        r['source']('/ihome/hpark/zhf16/test/DeepPINK/knockoff/RANK/genKnock.R');
        #Load the function.
        generateKnockoff = robjects.globalenv['generateKnockoff'];
        #Call the function
        knockoff_file_name = generateKnockoff(folder_path, file_name, "log", 5, 0, 0);
        
        #Override the data file with column names
        dataset.columns = feature_list;
        dataset.to_csv(folder_path+os.path.sep+file_name,index=False);
        
        #Save the (original data + knockoff data)
        knockoff_file_name = knockoff_file_name[0];
        orginal_knockoff_data = pd.read_csv(knockoff_file_name);
        orginal_knockoff_data.columns = feature_list+knockoff_feature_list;
        orginal_knockoff_data.to_csv(knockoff_file_name,index=False);
        
        return knockoff_file_name;
    
    def Chol_Lu_knockoff(self, folder_path, file_name):
        dataset = pd.read_csv(folder_path+os.path.sep+file_name);
        
        num_samples = dataset.shape[0];
        num_variables = dataset.shape[1];
        feature_names = dataset.columns.tolist();
        data_values = dataset.values;
        
        cov = np.corrcoef(data_values.T);#corrcoef cov
        #If the covariance matrix is positvely defined, use cholesky; else, use lu.
        if np.all(np.linalg.eigvals(cov) > 0):
            L = np.linalg.cholesky(cov);
        else:
            L = scipy.linalg.lu(cov)[0];
        print(L.shape);
        
        uncorrelated = np.random.standard_normal((num_samples,num_variables));

        print(uncorrelated.shape);
        mean = [0]*num_variables;
        
        correlated = np.dot(L, uncorrelated.T);# + np.array(mean).reshape(num_variables, 1)
        
        knockoff_feature_list = [];
        for idx in range(0,dataset.shape[1]):
            knockoff_feature_list.append("K"+str(idx+1));
        print(knockoff_feature_list);
        
        knockoffs = pd.DataFrame(correlated.T, columns=knockoff_feature_list);
        
        data_knockoffs = pd.concat([dataset, knockoffs], axis=1, join='inner');
        
        file_name = file_name.replace(".csv","_chol_lu_knockoff.csv");
        data_knockoff_path = folder_path+os.path.sep+file_name;
        data_knockoffs.to_csv(data_knockoff_path,index=False);
        return data_knockoff_path;
    def DNN_knockoff(self, folder_path, file_name, response_file_name):
        X_data = pd.read_csv(folder_path+os.path.sep+file_name);
        print(X_data.shape);
        
        Y_data = pd.read_csv(folder_path+os.path.sep+response_file_name);
        print(Y_data.shape);
        n_samples = X_data.shape[0];
        n_features = X_data.shape[1];
        feature_names = X_data.columns.tolist();
        
        n_targets =  Y_data.shape[1];
        target_names = Y_data.columns.tolist();
        features = X_data.values;
        output = Y_data.values;
        
        #Build a model
        model = Sequential();
        #model.add(Dense(1, activation='linear', bias=False, init='glorot_normal', W_regularizer=regularizers.l1(100.0), input_dim=X_dim))
        model.add(Dense(128, activation='relu', bias=True, init='glorot_normal', input_dim=n_features));
        model.add(Dense(64, activation='relu', bias=True, init='glorot_normal'));
        model.add(Dense(32, activation='relu', bias=True, init='glorot_normal'));
        model.add(Dense(n_targets, init='glorot_normal'));
        model.compile(loss='mse', optimizer = 'adam');
        model.summary()
        # Train the model, iterating on the data in batches of 32 samples
        model.fit(features, output, nb_epoch=50, batch_size=32);
        deeplift_model = kc.convert_sequential_model(model, nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault);
        deeplift_contribs_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-1);
        scores = np.array(deeplift_contribs_func(task_idx=0, input_data_list=[features], batch_size=100, progress_update=None));
        
        knockoff_feature_list = [];
        for idx in range(0,X_data.shape[1]):
            knockoff_feature_list.append("K"+str(idx+1));
        print(knockoff_feature_list);
        knockoff_data = pd.DataFrame(data=scores, columns=knockoff_feature_list);
        
        data_knockoffs = pd.concat([X_data, knockoff_data], axis=1, join='inner');
        
        file_name = file_name.replace(".csv","_DNN_knockoff.csv");
        data_knockoff_path = folder_path+os.path.sep+file_name;
        data_knockoffs.to_csv(data_knockoff_path,index=False);
        return data_knockoff_path;