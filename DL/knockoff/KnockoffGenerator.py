import numpy as np
import pandas as pd
import scipy
import os;
#os.environ['R_HOME'] = '/ihome/hpark/zhf16/.conda/envs/env36/lib/R';
#import rpy2.robjects as robjects;



from collections import OrderedDict

class KnockoffGenerator:
    
    ISEE_path = "";
    
    def __init__(self):
        print("KnockoffGenerator__init__");
        
    
    def set_ISEE_path(self,ISEE_path):
        self.ISEE_path = ISEE_path;
    def set_R_home(self,R_home):
        os.environ['R_HOME'] = R_home;
        import rpy2.robjects as robjects;
        # load the R instance
        #r = robjects.r;
        # load the R instance
        r = robjects.r;
        #Load the R script
        r['source'](self.ISEE_path+os.path.sep+'DL/knockoff/RANK/genKnock.R');
        #Load the function.
        self.generateKnockoff = robjects.globalenv['generateKnockoff'];
        
    def ISEEKnockoff(self, folder_path, file_name,sep="\t"):
        #print("===");
        dataset = pd.read_csv(folder_path+os.path.sep+file_name,sep=sep);
        feature_list = dataset.columns.tolist();
        dataset.to_csv(folder_path+os.path.sep+file_name,index=False,header=False,sep=sep);

        # print(dataset.shape);
        knockoff_feature_list = [];
        for idx in range(0,dataset.shape[1]):
            knockoff_feature_list.append("K"+str(idx+1));
        #print(feature_list);
        #print(knockoff_feature_list);
        
        ## load the R instance
        #r = robjects.r;
        ##Load the R script
        #r['source'](self.ISEE_path+os.path.sep+'DL/knockoff/RANK/genKnock.R');
        ##Load the function.
        #generateKnockoff = robjects.globalenv['generateKnockoff'];
        #Call the function
        knockoff_file_name = self.generateKnockoff(folder_path, file_name, "log", 5, 0, 0,sep=sep);
        
        #Override the data file with column names
        dataset.columns = feature_list;
        dataset.to_csv(folder_path+os.path.sep+file_name,index=False,sep=sep);
        
        #Save the (original data + knockoff data)
        knockoff_file_name = knockoff_file_name[0];
        print(knockoff_file_name);
        
        orginal_knockoff_data = pd.read_csv(knockoff_file_name,header=None,sep=sep);
        orginal_knockoff_data.columns = feature_list+knockoff_feature_list;
        
        if knockoff_file_name.endswith("_knockoff.csv"):
            knockoff_file_name = knockoff_file_name.replace("_knockoff.csv","_Omega_knockoff.csv");
        else:
            knockoff_file_name = knockoff_file_name.replace("_knockoff.txt","_Omega_knockoff.txt");
        #knockoff_file_path = folder_path+os.path.sep+knockoff_file_name;
        orginal_knockoff_data.to_csv(knockoff_file_name,index=False,sep=sep);
        
        return knockoff_file_name;
    
    def CholLuKnockoff(self, folder_path, file_name,sep="\t"):
        dataset = pd.read_csv(folder_path+os.path.sep+file_name,sep=sep);
        
        num_samples = dataset.shape[0];
        num_variables = dataset.shape[1];
        feature_names = dataset.columns.tolist();
        data_values = dataset.values;
        print(data_values.shape);
        cov = np.corrcoef(data_values.T);
        #If the covariance matrix is positvely defined, use cholesky; else, use lu.
        if np.all(np.linalg.eigvals(cov) > 0):
            try:
                L = np.linalg.cholesky(cov);
            except np.linalg.LinAlgError:
                L = scipy.linalg.lu(cov)[0];
        else:
            L = scipy.linalg.lu(cov)[0];
        #print(L.shape);
        
        uncorrelated = np.random.standard_normal((num_samples,num_variables));

        #print(uncorrelated.shape);
        mean = [0]*num_variables;
        
        correlated = np.dot(L, uncorrelated.T);# + np.array(mean).reshape(num_variables, 1)
        
        knockoff_feature_list = [];
        for idx in range(0,dataset.shape[1]):
            knockoff_feature_list.append("K"+str(idx+1));
        print(knockoff_feature_list);
        
        knockoffs = pd.DataFrame(correlated.T, columns=knockoff_feature_list);
        
        data_knockoffs = pd.concat([dataset, knockoffs], axis=1, join='inner');
        
        if file_name.endswith(".csv"):
            file_name = file_name.replace(".csv","_chol_lu_knockoff.csv");
        else:
            file_name = file_name.replace(".txt","_chol_lu_knockoff.txt");
        
        data_knockoff_path = folder_path+os.path.sep+file_name;
        data_knockoffs.to_csv(data_knockoff_path,index=False,sep='\t');
        return data_knockoff_path;
    def DNN_knockoff(self, folder_path, file_name, response_file_name,sep="\t"):
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

        X_data = pd.read_csv(folder_path+os.path.sep+file_name,sep=sep);
        print(X_data.shape);
        
        Y_data = pd.read_csv(folder_path+os.path.sep+response_file_name,sep=sep);
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
        data_knockoffs.to_csv(data_knockoff_path,index=False,sep='\t');
        return data_knockoff_path;