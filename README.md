
# Introduction
Identifying nonlinear causal relations and estimating their effect size help understand the complex disease pathobiology. Directed Acyclic Graphs using deep-learning VAriable SElection (DAG-deepVASE) is the first computational method that learns both linear and nonlinear causal relations and estimates the effect size using a deep-neural network approach coupled with the knockoff framework.
####

# Test DAG-DeepVASE using a Docker container
1. Download Docker [here](https://docs.docker.com/get-docker/) and install it.
2. Download DAG-DeepVASE docker image via the Docker command below:
```python
zhenjiangfan/dagdeepvase:latest
```
3. Run the Docker image in a Docker container:
```python
docker run -it --name dagdeepvase dagdeepvase
```
4. Test DAG-DeepVASE by running the Python code below in the Docker container:
```python
python3 ExampleForDataWithTarget.py
```
4. Test DAG-DeepVASE using the TCGA BRCA RNA-seq data:
```python
python3 ExampleUsingBRCAData.py
```

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

#### The source code [*/Example.ipynb*](./Example.ipynb) demonstrates how to use DAG-DeepVASE.
#### The source code [*/ExampleForDataWithNoTarget.ipynb*](./ExampleForDataWithNoTarget.ipynb) demonstrates how to use DAG-DeepVASE when the dataset does not have a target variable.


#### Import required packages:


```python
import pandas as pd
import numpy as np
import os
```

#### Run MGM to get linear associations


```python
XYDataFileName = "XY_n1000_p100_rep20.txt";
dataFolderPath = "data/simData";
'''
Run MGM
Note: MGM was implemented in Java and the following Python APIs call the Java implementation.
Please restart the Python program after encountering a JVM problem.
The format of the input data file must be ".txt" in which columns are separated by "\t" and it should also include the response variables.
Here is what the input data should look like:
X1 X2 ... Xp
1  1  ... 1
'''
# import the MGM package
from MGM.MGM import MGM
# Initialize a MGM object
mgm = MGM();
'''
Run MGM
Parameters:
    dataFolderPath: the directory that stores the input data.
    DataFileName: the name of the input data.
    lambda_continuous_continuous: the panalty value 'lambda' set for the associations whose two end variables are continuous.
    lamda_continuous_discrete: the panalty value 'lambda' set for the associations whose one end variable is continuous and the other is discrete.
    lamda_discrete_discrete: the panalty value 'lambda' set for the associations whose two end variables are discrete.
    
Return:
    mgmOutputFile: a tuple, where the first file contains all the selected associations and the second file contains the corresponding likelihoods.
'''
mgmOutputFile = mgm.runMGM(dataFolderPath, XYDataFileName,lambda_continuous_continuous = 0.3, lamda_continuous_discrete = 0.3, lamda_discrete_discrete = 0.3);
"""
MGM uses the Python package Jpype to call MGM's Java implementation.
According to Jpype documents, it says "Due to limitations in the JPype, 
it is not possible to restart the JVM after being terminated."
Therefore, please restart the Python kernel if you encounter an OSError (i.e., "OSError: JVM cannot be restarted").
"""
print("MGM's output was saved as the following file:");
print(mgmOutputFile[0]);
print("The likelihood values were saved as the following file:");
print(mgmOutputFile[1]);
```

#### Run DNN to get nonlinear associations
##### Generate knockoff data


```python
'''
#Generate knockoff data using one of two methods: ISEE Omega and Cholesky_LU.
The code for generating ISEE Omega knockoff is implemented using R. Please make sure your computer has R installed.
'''
from DL.knockoff.KnockoffGenerator import KnockoffGenerator;
generator = KnockoffGenerator();

DataFileName = "X_n1000_p100_rep20.txt";
# knockoffFilePath = generator.CholLuKnockoff(dataFolderPath, DataFileName,sep="\t");

#If want to generate ISEE Omega knockoff, please set the ISEE code path and R home environment.
generator.set_ISEE_path("/absolute_path_of_DAG_DeepVASE");#/home/user/DAG_DeepVASE/
generator.set_R_home('absolute_path_to_directory_where_r_is_installed');#e.g.,/home/user/lib/R
knockoffFilePath = generator.ISEEKnockoff(dataFolderPath, DataFileName,sep="\t");

print("The newly generated knockoff file is named as:");
print(knockoffFilePath);
```

##### Run DNN


```python
''''''
# After generating the knockoff data, run DNN
XKnockoffData = pd.read_csv(knockoffFilePath,sep="\t");

YDataFileName = 'y_si_n1000_p100_rep20.txt';
Ydata = pd.read_csv(dataFolderPath+os.path.sep+YDataFileName,sep="\t");

XKValues = XKnockoffData.values;
YValues = Ydata.values;
    
pNum = int(XKValues.shape[1] / 2);
n = XKValues.shape[0];
    
XOrigin = XKValues[:, 0:pNum];
knockoff = XKValues[:, pNum:];

X3DTrain = np.zeros((n, pNum, 2));
X3DTrain[:, :, 0] = XOrigin;
X3DTrain[:, :, 1] = knockoff;
labelTrain = YValues;
coeff = 0.05 * np.sqrt(2.0 * np.log(pNum) / n);
nOutputs = Ydata.shape[1];

#Save the DNN output to the following directory.
resultDir = dataFolderPath+os.path.sep+'DNN_result/';
if not os.path.exists(resultDir):
    os.makedirs(resultDir);
    
from DL.DNN.DNN import DNN;
dnn = DNN();
model = dnn.build_DNN(pNum, nOutputs, coeff);
callback = DNN.Job_finish_Callback(resultDir,pNum);
dnn.train_DNN(model, X3DTrain, labelTrain,callback);
```

##### Apply FDR control


```python
#Apply FDR control to DNN result
from DL.FDR.FDR_control import FDR_control;
control = FDR_control();
XDataFileName = "X_n1000_p100_rep20.txt";
selected_features = control.controlFilter(dataFolderPath +os.path.sep+ XDataFileName, resultDir, offset=1, q=0.05);
#Save the selected associations
selected_associations = [];
for ele in selected_features:
    selected_associations.append({"Feature1":ele[0],"Feature2":"Y"});
pd.DataFrame(selected_associations).to_csv(dataFolderPath +os.path.sep+"DNN_selected_associations.csv");
```


```python
# Run DG
# Load data
dataset = pd.read_csv(dataFolderPath+os.path.sep+XYDataFileName,sep="\t");

#Calculate the covariance matrix
cov_mat = dataset.cov();
corr_inv = np.linalg.inv(cov_mat)
corr_inv = pd.DataFrame(data=corr_inv, index=cov_mat.index,columns=cov_mat.columns)

#Convert the columns to their numerical representations
col_map = {};
col_map_rev = {};
col_list = dataset.columns.to_list();
for index,ele in enumerate(col_list):
    col_map[ele] = index;
    col_map_rev[index] = ele;
print(dataset.shape);

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
selectedAssociationsSum = [];
#Load both MGM-identified and DNN associations
MGMAssociations = pd.read_csv(mgmOutputFile[0]);
for index,row in MGMAssociations.iterrows():
    f1 = row["Feature1"];
    f2 = row["Feature2"];
    if f1=="Y" or f2=="Y":
        tempList = [f1,f2];
        tempList.sort();
        tempStr = tempList[0]+"___"+tempList[1];
        if tempStr not in selectedAssociationsSum:
            selectedAssociationsSum.append(tempStr);
        
DNNAssociations = pd.read_csv(dataFolderPath +os.path.sep+"DNN_selected_associations.csv");
for index,row in DNNAssociations.iterrows():
    f1 = row["Feature1"];
    f2 = row["Feature2"];
    tempList = [f1,f2];
    tempList.sort();
    tempStr = tempList[0]+"___"+tempList[1];
    if tempStr not in selectedAssociationsSum:
        selectedAssociationsSum.append(tempStr);
```


```python
import networkx as nx
#Calculate causal directions
causalGraph = nx.DiGraph();
for ele in selectedAssociationsSum:
    strs = ele.split("___");
    f1 = strs[0];
    f2 = strs[1];
    
    inv_val = abs(corr_inv[f1][f2]);
    if inv_val<=0.0:
        continue;

    n1_idx = col_map[f1];
    n2_idx = col_map[f2];

    s1 = dg.localScore(n1_idx,{n2_idx});
    s2 = dg.localScore(n2_idx,{n1_idx});

    if s1<s2:
        print("Cause: "+f2+", Effect: "+f1);
        dif = s2-s1;
        causalGraph.add_edge(f2, f1, weight=dif);
    elif s1>s2:
        print("Cause: "+f1+", Effect: "+f2);
        dif = s1-s2;
        causalGraph.add_edge(f1, f2, weight=dif);
    else:
        print("Same score.");
        
#Remove cycles
causalGraph = dg.removeCycles(causalGraph);

import scipy.stats
#Identify if a causal relationship is positive or negative.
edgeList = [];
for edge in causalGraph.edges():
    cause = edge[0];
    effect = edge[1];
    effectSize = np.log(causalGraph.get_edge_data(cause,effect)['weight']);
    corr = scipy.stats.pearsonr(dataset[cause].values,dataset[effect].values)[0];
    if corr>0:
        edgeList.append({"Cause":cause,"Effect":effect,"EffectSize":effectSize,"CauseDirection":"Positive"});
    elif corr == 0:
        edgeList.append({"Cause":cause,"Effect":effect,"EffectSize":effectSize,"CauseDirection":"Undefined"});
    else:
        edgeList.append({"Cause":cause,"Effect":effect,"EffectSize":effectSize,"CauseDirection":"Negative"});
```


```python

```





# Acknowledgement and References

#### :trophy: Some components of this project come from the follwing projects:
:star: The MGM Java implemention is from [causalMGM](https://github.com/benoslab/causalMGM) and [TetradLite](https://github.com/benoslab/tetradLite).\
:star: The DeepPINK implementation is from [DeepPINK](https://github.com/younglululu/DeepPINK).\
:star: The FDR filter function is from [DeepKnockoffs](https://github.com/msesia/deepknockoffs).\
:star: The Python implementation of DG algorithm is based on its Java version from [Tetrad](https://www.ccd.pitt.edu).\
:star: The implementation of the PC algorithm used in this project is from [pcalg](https://github.com/keiichishima/pcalg).

# Contact
:email: Please let us know if you have any questions, bug reports, or feedback via the following email:
<p align="center">
    :e-mail: hyp15@pitt.edu
</p>
    

