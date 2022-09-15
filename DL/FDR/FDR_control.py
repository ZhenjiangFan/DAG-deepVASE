import numpy as np
import pandas as pd
import scipy.stats as stats
import os


class FDR_control:
    
    def __init__(self):
        print("__init__");
    
    def kfilter(self, W, offset=1.0, q=0.05):
        """
        Adaptive significance threshold with the knockoff filter
        :param W: vector of knockoff statistics
        :param offset: equal to one for strict false discovery rate control
        :param q: nominal false discovery rate
        :return a threshold value for which the estimated FDP is less or equal q
        """
        t = np.insert(np.abs(W[W!=0]),0,0)
        t = np.sort(t)
        ratio = np.zeros(len(t));
        for i in range(len(t)):
            ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        #print(ratio)    
        index = np.where(ratio <= q)[0]
        #print(index) 
        if len(index)==0:
            thresh = float('inf')
        else:
            thresh = t[index[0]]
       
        return thresh
    
    
    def controlFilter(self, X_data_path, W_result_path, offset=1, q=0.05):
        
        X_data = pd.read_csv(X_data_path,sep="\t");
        feature_list = X_data.columns.tolist();
        number_of_features = X_data.shape[1];
        
        result_path = W_result_path+'/result_epoch20_featImport.csv';
        result_file = pd.read_csv(result_path,header=None);
        result_list = result_file.iloc[0].values.tolist();
        print(len(result_list));
    
        original_result_list = result_list[:number_of_features];
        knockoff_result_list = result_list[number_of_features:];
    
        stats = [];
        for index in range(0,number_of_features):
            stat = abs(original_result_list[index])-abs(knockoff_result_list[index]);
            stats.append(stat);
        
        W = [[i] for i in stats];
        threshold = self.kfilter(W,offset=offset, q=q);
        if threshold==np.inf:
            threshold=0;
        #print("---");
        #print(threshold);
    
        selected_features = [];
        for index in range(0,number_of_features):
            #print("Original: {}, Knockoff: {}".format(original_result_list[index],knockoff_result_list[index]))
            stat = abs(original_result_list[index])-abs(knockoff_result_list[index]);
            if stat>threshold:
                selected_features.append((feature_list[index],stat));   
            
        #print(len(selected_features));
        return selected_features;
        
        

