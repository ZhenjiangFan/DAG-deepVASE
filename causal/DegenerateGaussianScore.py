import numpy as np
import pandas as pd
import math as math
from scipy.linalg import lu,det
import networkx as nx


class DegenerateGaussianScore:
    dataframe = pd.DataFrame();
    #The mixed variables of the original dataset.
    variables = [];
    #The continuous variables of the post-embedding dataset.
    continuousVariables = [];
    #The penalty discount.
    penaltyDiscount = 1.0;
    #The structure prior.
    structurePrior = 0.0;
    #The number of instances.
    N = 0;
    #The embedding map.
    #private Map<Integer, List<Integer>> embedding;
    embedding = {};
    #The covariance matrix.
    cov = pd.DataFrame();
    #A constant.
    L2PE = np.log(2.0*math.pi*math.e);
    
    is_var_discrete_map = {};
    
    continuous_list = [];
    
    discrete_threshold = 0.02;
    
    def __init__(self, data, continuous_list=[], discrete_threshold=0.2):
        dataframe = data;
        self.variables = dataframe.columns.tolist();
        self.N = dataframe.shape[0];
        self.embedding = {};
        self.continuous_list = continuous_list;
        self.discrete_threshold=discrete_threshold;
        
        is_var_discrete_map = {};
        for var in dataframe.columns:
            if var in self.continuous_list:
                is_var_discrete_map[var] = False;
                print(var);
            else:
                is_var_discrete_map[var] = 1.*dataframe[var].nunique()/dataframe[var].count() < self.discrete_threshold;
        #print(is_var_discrete_map);
        
        A = [];
        B = [];
        i = 0;
        i_ = 0;
        while i_ < len(self.variables):
            v = self.variables[i_];
            if is_var_discrete_map[v]:
                keys = {};
                for j in range(0,self.N):
                    key = v+"_";
                    key = key+str(dataframe.iloc[j][i_]);
                    #print(key);
                    if key not in keys:
                        keys[key] = i;
                        A.append(key);
                        t_a = [0] * self.N;
                        B.append(t_a);
                        i=i+1;
                    B[keys[key]][j] = 1;
                
                #Remove a degenerate dimension.
                i=i-1;
                #print(keys)
                keys.pop(A[i]);
                A.pop(i);
                B.pop(i);

                self.embedding[i_] = list(keys.values());
                i_=i_+1;
            else:
                A.append(v);
                t_b = [0] * self.N;
                for j in range(0,self.N):
                    t_b[j] = dataframe.iloc[j][i_];
                
                B.append(t_b);
                index = [];
                index.append(i);
                self.embedding[i_] = index;
                i=i+1;
                i_=i_+1;
        #print(self.embedding);
        
        #print(self.N);
        #print(len(B));
        B_ = np.zeros((self.N, len(B)))
        for j in range(0,len(B)):
            for k in range(0,self.N):
                B_[k][j] = B[j][k];
        #print(B_)
        self.continuousVariables = A;
        #print(self.continuousVariables)
        self.cov = pd.DataFrame(data=B_,columns=A).cov();
        #print(self.cov)
    def calculateStructurePrior(self,k):
        if self.structurePrior <= 0:
            return 0;
        else:
            n = len(self.variables) - 1;
            p = self.structurePrior / n;
            return k*np.log(p) + (n - k)*np.log(1.0 - p);
    
    def localScore(self,i,parents):
        t_A = [];
        t_B = [];
        
        t_A.extend(self.embedding.get(i));
        for i_ in parents: 
            t_B.extend(self.embedding.get(i_));
        
        A_ = [0]*(len(t_A) + len(t_B));
        B_ = [0]*len(t_B);
        
        for i_ in range(0,len(t_A)):
            A_[i_] = t_A[i_];
        for i_ in range(0,len(t_B)):
            A_[len(t_A)+i_] = t_B[i_];
            B_[i_] = t_B[i_];
        #print(A_)
        #print(B_)
        dof = (len(A_)*(len(A_)+1) - len(B_)*(len(B_)+1))/2.0;
        #print(dof)
        
        ldetA = np.log(det(self.cov.iloc[A_, A_].values));
        #print(det(self.cov.iloc[A_, A_].values))
        #print(ldetA)
        
        ldetB = np.log(det(self.cov.iloc[B_, B_].values));
        #print(det(self.cov.iloc[B_, B_].values))
        #print(ldetB)
        
        lik = self.N *(ldetB - ldetA + self.L2PE*(len(B_) - len(A_)));
        #print(self.N)
        #print(lik)
        struct_prior = self.calculateStructurePrior(len(parents));
        #print(struct_prior)
        result = lik + 2*struct_prior - dof*self.penaltyDiscount*np.log(self.N);
        #print(result)
        return result;
    
    def removeCycles(causalGraph):
        #print(causalGraph.edges());
        cycyles = list(nx.simple_cycles(causalGraph));
        for cycle in cycyles:
            source_node = cycle[0];
            target_node_index = 1;

            marked_source_node = "";
            marked_target_node = "";
            marked_weight = 0;

            while (target_node_index<len(cycle)):
                target_node = cycle[target_node_index];
                weight = causalGraph.get_edge_data(source_node,target_node)['weight'];
                #print(weight);
                if (marked_source_node =="" and marked_target_node=="") or marked_weight<weight:
                    marked_weight = weight;
                    marked_source_node = source_node;
                    marked_target_node = target_node;

                source_node = target_node;
                target_node_index=target_node_index+1;
            target_node = cycle[0];
            weight = causalGraph.get_edge_data(source_node,target_node)['weight'];
            #print(weight);
            if marked_weight<weight:
                marked_weight = weight;
                marked_source_node = source_node;
                marked_target_node = target_node;
            #Delete the node with smallest weight
            causalGraph.remove_edge(source_node,target_node);
        #print(causalGraph.edges());
        return causalGraph;
