
# Introduction

#### 

# Special Package Requirements

:stars: rpy2\
:stars: keras (:bell: please use the version of 2.3.1, which can be installed using "pip install Keras==2.3.1".)\
:stars: Deeplift (:bell: please use the version of 0.5.1-theano, which should be downloaded online.)\
:stars: Tensorflow (:bell: please use the version of 2.2.0, which can be installed using "pip install tensorflow==2.2.0".)\
:stars: jpype\
:stars: R environment\
:stars: Java environment

# Installation

#### Please download the source code to your local workplace.

# Source Code 

📁The sub folder [*/MGM*](./MGM/) contains the Python version of MGM implementation.\
📁 The sub folder [*/DL*](./DL/) contains the Python implemenation of DeepPINK procedure (knockoff data generation, DNN, and FDR).\
📁 The sub folder [*/causal*](./causal/) contains the Python implementation of DG procedure.

# Example

#### The source code [*/Example.ipynb*](./Example.ipynb) demonstrates how to use causalDeepVASE.


#### Import required packages:


```python
import pandas as pd
import numpy as np
import os
```

#### Run MGM to get direct or linear associations


```python
XY_file_name = "200p_1000samples_XY.txt";
data_folder_path = "/ihome/hpark/zhf16/causalDeepVASE/data";
'''
Run MGM
Note: MGM was implemented in Java and the following Python APIs call the Java implementation.
Please restart the Python program after encountering a JVM problem.
The input data file should be ".txt" format and should also include the response variables.
Here is what the input data should look like:
X1 X2 ... Xp Y1 ... Yq
1  1  ... 1  1  ... 1
'''
# import the MGM package
from MGM.MGM import MGM
# Initialize a MGM object
mgm = MGM();
'''
Run MGM
Parameters:
    data_folder_path: the directory at where the input data is located.
    XY_file_name: the input data.
    lambda_continuous_continuous: the panalty value 'lamda' set for the associations whose two variables are continuous.
    lamda_continuous_discrete: the panalty value 'lamda' set for the associations whose one variable is continuous and the other is discrete.
    lamda_discrete_discrete: the panalty value 'lamda' set for the associations whose two variables are discrete.
    
Return:
    mgm_output_file: a file that contains all the selected associations.
'''
mgm_output_file = mgm.runMGM(data_folder_path, XY_file_name,lambda_continuous_continuous = 0.3, lamda_continuous_discrete = 0.3, lamda_discrete_discrete = 0.3);
print("Please find MGM's output file as:");
mgm_output_file_path = data_folder_path+os.path.sep+mgm_output_file;
print(mgm_output_file_path);
```

#### Run DNN to get indirect or nonlinear associations
##### Generate knockoff data


```python
X_file_name = "200p_1000samples_X.csv";
'''
#Generate knockoff data using one of three methods: ISEE Omega, DNN, and Cholesky_LU.
#Recommended: ISEE Omega or Cholesky_LU.
The code for generating ISEE Omega knockoff is implemented using R. Please make sure your computer has R installed.
'''
#Import the package
from DL.knockoff.KnockoffGenerator import KnockoffGenerator;
#Initialize the knockoff generator object
generator = KnockoffGenerator();


# knockoff_file_path = generator.Chol_Lu_knockoff(data_folder_path, X_file_name);

#If want to generate ISEE Omega knockoff, please set the ISEE code path and R home environment.

generator.set_ISEE_path("/ihome/hpark/zhf16/causalDeepVASE/");

generator.set_R_home('/ihome/hpark/zhf16/.conda/envs/env36/lib/R');

knockoff_file_path = generator.ISEE_knockoff(data_folder_path, X_file_name);

# Y_file_name = '200p_1000samples_Y.csv';

# knockoff_file_path = generator.DNN_knockoff(data_folder_path, X_file_name,Y_file_name);

print("The newly generated knockoff file is named as:")
print(knockoff_file_path);
```

##### Run DNN


```python
''''''
# After generating the knockoff data, run DNN
Y_file_name = '200p_1000samples_Y.csv';
X_knockoff_data = pd.read_csv(knockoff_file_path);
print(X_knockoff_data.shape)
# X_knockoff_data

#nutrient_data
original_data_Y = pd.read_csv(data_folder_path+os.path.sep+Y_file_name);
# original_data_Y

X_values = X_knockoff_data.values;
Y_values = original_data_Y.values;
    
pVal = int(X_values.shape[1] / 2);
n = X_values.shape[0];
print(X_values.shape);
print(Y_values.shape);
print(pVal);
    
X_origin = X_values[:, 0:pVal];
X_knockoff = X_values[:, pVal:];

x3D_train = np.zeros((n, pVal, 2));
x3D_train[:, :, 0] = X_origin;
x3D_train[:, :, 1] = X_knockoff;
label_train = Y_values;
    
coeff = 0.05 * np.sqrt(2.0 * np.log(pVal) / n);

n_outputs = original_data_Y.shape[1];

#Save the DNN output to the following directory.
result_dir = 'data/DNN_result/';
if not os.path.exists(result_dir):
    os.makedirs(result_dir);
    
from DL.DNN.DNN import DNN;
dnn = DNN();
model = dnn.build_DNN(pVal, n_outputs, coeff);
callback = DNN.Job_finish_Callback(result_dir,pVal);
dnn.train_DNN(model, x3D_train, label_train,callback);
```

##### Apply FDR control


```python
#Apply FDR control to DNN result
from DL.FDR.FDR_control import FDR_control;
control = FDR_control();
selected_features = control.controlFilter(data_folder_path +os.path.sep+ X_file_name, "/ihome/hpark/zhf16/causalDeepVASE/data/DNN_result", offset=1, q=0.05);
#Save the selected associations
selected_associations = [];
for ele in selected_features:
    selected_associations.append({"Feature1":ele,"Feature2":"Y"});
pd.DataFrame(selected_associations).to_csv("data/DNN_selected_associations.csv")
```


```python
#Run DG
#Load data
X_data = pd.read_csv("X_n1000_p50_rep20.csv");
# X_data
Y_data = pd.read_csv('y_si_n1000_p50_rep20.csv');
#Merge X and Y
dataset = pd.concat([X_data, Y_data], axis=1, join='inner');
print(dataset.shape);

#Calculate the covariance matrix
cov_mat = dataset.cov();
corr_inv = np.linalg.inv(cov_mat)
corr_inv = pd.DataFrame(data=corr_inv, index=cov_mat.index,columns=cov_mat.columns)
# corr_inv.head(2)

#Convert the columns to their numerical representations
col_map = {};
col_map_rev = {};
col_list = dataset.columns.to_list();
for index,ele in enumerate(col_list):
    col_map[ele] = index;
    col_map_rev[index] = ele;
print(dataset.shape);

# t = dataset.shape[0]**(1/2)

#The data may need to be normalized if neccessary.
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler();
# scaled_values = scaler.fit_transform(dataset);
# dataset.loc[:,:] = scaled_values;

#Initialize DG object
from causal.DegenerateGaussianScore import DegenerateGaussianScore
dg = DegenerateGaussianScore(dataset,discrete_threshold=0.2);
```


```python
selected_associations_sum = [];
#Load both MGM-identified and DNN associations
MGM_associations = pd.read_csv("X_n1000_p50_rep20_MGM_associations.csv");
for index,row in MGM_associations.iterrows():
    if row["Feature1"]=="Y" or row["Feature2"]=="Y":
        print("Found.");
        selected_associations_sum.append({"Feature1":row["Feature1"],"Feature2":row["Feature2"]});
        
DNN_associations = pd.read_csv("DNN_selected_associations.csv");
for index,row in DNN_associations.iterrows():
    selected_associations_sum.append({"Feature1":row["Feature1"],"Feature2":row["Feature2"]});
```


```python
for ele in selected_associations_sum:
    f1 = ele["Feature1"];
    f2 = ele["Feature2"];
    
    inv_val = abs(corr_inv[f1][f2]);
    if inv_val<0.0:
        continue;
    
    n1_idx = col_map[f1];
    n2_idx = col_map[f2];
    
    s1 = dg.localScore(n1_idx,{n2_idx});
    s2 = dg.localScore(n2_idx,{n1_idx});
    
    if s1<s2:
        print("Cause: "+f2+", Effect: "+f1);
    elif s1>s2:
        print("Cause: "+f1+", Effect: "+f2);
    else:
        print("Same score.");
```


```python

```





# Acknowledgement and References

#### :trophy: Some components of this project come from the follwing projects:
:star: The MGM Java implemention is from [causalMGM](https://github.com/benoslab/causalMGM) and [Tetrad](https://www.ccd.pitt.edu).\
:star: The DeepPINK implementation is from [DeepPINK](https://github.com/younglululu/DeepPINK).\
:star: The FDR filter function is from [DeepKnockoffs](https://github.com/msesia/deepknockoffs).\
:star: The Python implementation of DG algorithm is based on its Java version from [Tetrad](https://www.ccd.pitt.edu).\
:star: The implementation of the PC algorithm used in this project is from [pcalg](https://github.com/keiichishima/pcalg).

# Contact
:email: Please let us know if you have any questions, bug reports, or feedback via the following email:
<p align="center">
    :e-mail: hyp15@pitt.edu
</p>
    

