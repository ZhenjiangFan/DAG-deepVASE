import pandas as pd
import numpy as np
import os
import shutil
from DL.DNN.DNN import DNN;
from DL.FDR.FDR_control import FDR_control;
from causal.DegenerateGaussianScore import DegenerateGaussianScore
import networkx as nx
import scipy.stats



dataFileName = "MircobiomeBacteriaBMI.txt";
dataFolderPath = "data/BMI";


# In[3]:


'''
Run MGM
Note: MGM was implemented in Java and the following Python APIs call the Java implementation.
Please restart the Python program after encountering a JVM problem.
The input data file should be ".txt" format and should also include the response variables.
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
    dataFolderPath: the directory at where the input data is stored.
    DataFileName: the input data.
    lambda_continuous_continuous: the panalty value 'lambda' set for the associations whose two end variables are continuous.
    lamda_continuous_discrete: the panalty value 'lambda' set for the associations whose one end variable is continuous and the other is discrete.
    lamda_discrete_discrete: the panalty value 'lambda' set for the associations whose two end variables are discrete.
    
Return:
    mgmOutputFile: a tuple, where the first file contains all the selected associations and the second file contains the corresponding likelihoods.
'''
mgmOutputFile = mgm.runMGM(dataFolderPath, dataFileName,lambda_continuous_continuous = 0.3, lamda_continuous_discrete = 0.3, lamda_discrete_discrete = 0.3);
"""
MGM uses the Python package Jpype to call MGM's Java implementation.
According to Jpype documents, it says "Due to limitations in the JPype, 
it is not possible to restart the JVM after being terminated."
Therefore, please restart the Python kernel if you encounter an OSError (i.e., "OSError: JVM cannot be restarted").
"""
mgmOutputFilePath = mgmOutputFile[0];
print("MGM's output was saved as the following file:");
print(mgmOutputFilePath);
print("The likelihood values were saved as the following file:");
likelihoodFilePath = mgmOutputFile[1];
print(likelihoodFilePath);


# In[13]:


dataset = pd.read_csv(dataFolderPath+"/"+dataFileName,sep="\t");
#colList = dataset.columns.tolist();
colList = ["BMI"];

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
    


# In[14]:


#Collect all the associations
DNNSelectionList = [];
fileList = os.listdir(dataFolderPath+"/DNNSelection/");
for fileName in fileList:
    if "DNNSelectedAssociations" in fileName:
        assoDF = pd.read_csv(dataFolderPath+"/DNNSelection/"+fileName);
        for index,row in assoDF.iterrows():
            f1 = row["Feature1"];
            f2 = row["Feature2"];
            #print("F1:"+f1+"      F2:"+f2)
            tempList = [f1,f2];
            tempList.sort();
            tempStr = tempList[0]+"___"+tempList[1];
            if tempStr not in DNNSelectionList:
                DNNSelectionList.append(tempStr);
print(len(DNNSelectionList));


# In[15]:


#Collect MGM associations
MGMSelectionList = [];
assoDF = pd.read_csv(mgmOutputFilePath,index_col=0);
for index,row in assoDF.iterrows():
    f1 = row["Feature1"];
    f2 = row["Feature2"];
    if f1 == "BMI" or f2 == "BMI":
        tempList = [f1,f2];
        tempList.sort();
        tempStr = tempList[0]+"___"+tempList[1];
        if tempStr not in MGMSelectionList:
            MGMSelectionList.append(tempStr);
print(len(MGMSelectionList));


# In[16]:


#Merge two selection lists
finalSelectionList = set(DNNSelectionList).union(MGMSelectionList);
print(len(finalSelectionList));



# In[17]:


#Calculate the covariance matrix
cov_mat = dataset.cov();
corr_inv = np.linalg.inv(cov_mat)
corr_inv = pd.DataFrame(data=corr_inv, index=cov_mat.index,columns=cov_mat.columns)

#Convert the columns to their numerical representations
col_map = {};
col_map_rev = {};
colList = dataset.columns.tolist();
for index,ele in enumerate(colList):
    col_map[ele] = index;
    col_map_rev[index] = ele;

#Normalize the dataset if neccesary.
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler();
scaled_values = scaler.fit_transform(dataset);
dataset.loc[:,:] = scaled_values;
'''

#Initialize DG object
#Please set the ordinal discrete variables or the variables that should be handled as continuous variables as continuous variables.
continuous_list = [];#"Ferritin_cat","MAS","Death"
dg = DegenerateGaussianScore(dataset,continuous_list=continuous_list,discrete_threshold=0.05);


# In[18]:

'''
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


# In[19]:
'''

#Calculate causal directions
causalGraph = nx.DiGraph();
for ele in finalSelectionList:
    strs = ele.split("___");
    f1 = strs[0];
    f2 = strs[1];
    #print("F1:"+f1+"      F2:"+f2)
    #if "Ferritin_cat" in ele or "MAS" in ele or "Death" in ele:
    inv_val = abs(corr_inv[f1][f2]);
    if inv_val<=0.0:
        continue;

    n1_idx = col_map[f1];
    n2_idx = col_map[f2];

    s1 = dg.localScore(n1_idx,{n2_idx});
    s2 = dg.localScore(n2_idx,{n1_idx});

    if s1<s2:
        #print("Cause: "+f2+", Effect: "+f1);
        dif = s2-s1;
        causalGraph.add_edge(f2, f1, weight=dif);
    elif s1>s2:
        #print("Cause: "+f1+", Effect: "+f2);
        dif = s1-s2;
        causalGraph.add_edge(f1, f2, weight=dif);
    else:
        print("Same score.");
#Remove cycles
causalGraph = dg.removeCycles(causalGraph);


# In[20]:


#Identify if a causal relationship is positive or negative and then save them.
edgeList = [];
for edge in causalGraph.edges():
    cause = edge[0];
    effect = edge[1];
    print("Cause: "+cause+", Effect: "+effect);
    
    effectSize = np.log(causalGraph.get_edge_data(cause,effect)['weight']);
    corr = scipy.stats.pearsonr(dataset[cause].values,dataset[effect].values)[0];
    if corr>0:
        edgeList.append({"Cause":cause,"Effect":effect,"EffectSize":effectSize,"CauseDirection":"Positive"});
    elif corr == 0:
        edgeList.append({"Cause":cause,"Effect":effect,"EffectSize":effectSize,"CauseDirection":"Undefined"});
    else:
        edgeList.append({"Cause":cause,"Effect":effect,"EffectSize":effectSize,"CauseDirection":"Negative"});
    #print(corr);

# In[21]:



pd.DataFrame(edgeList).to_csv(dataFolderPath+"/CausalResult.csv");




