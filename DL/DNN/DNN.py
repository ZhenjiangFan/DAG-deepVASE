import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, BatchNormalization, LocallyConnected1D, Flatten, Conv1D
from keras import backend as K
from keras import regularizers
#from keras.objectives import mse
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant


import numpy as np
import pandas as pd

import time
import math
import sys
import os

class DNN:
    
    num_epochs = 20;
    batch_size = 10;
    filterNum = 1;

    bias = True;
    activation='relu';
    output_layer_activation = 'linear'
    
    def __init__(self):
        print("__init__");
    def __init__(self,num_epochs = 20, batch_size=30, filterNum=1, bias=True, activation='relu', output_layer_activation='linear'):
        print("__init__parameters");
        self.num_epochs=num_epochs;
        self.batch_size=batch_size;
        self.filterNum = filterNum;
        self.bias = bias;
        self.activation = activation;
        self.output_layer_activation=output_layer_activation;
        
    def show_layer_info(self, layer_name, layer_out):
        print('[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())))
        pass
    
    def build_DNN(self, pVal, n_outputs ,coeff=0):

        input = Input(name='input', shape=(pVal,2))
        self.show_layer_info('Input', input);

        local1 = LocallyConnected1D(self.filterNum,1, use_bias=self.bias, kernel_initializer=Constant(value=0.1))(input);
        self.show_layer_info('LocallyConnected1D', local1);

        local2 = LocallyConnected1D(1,1, use_bias=self.bias, kernel_initializer='glorot_normal')(local1);
        self.show_layer_info('LocallyConnected1D', local2);

        flat = Flatten()(local2);
        self.show_layer_info('Flatten', flat);

        dense1 = Dense(pVal, activation=self.activation,use_bias=self.bias, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1(coeff))(flat);
        self.show_layer_info('Dense', dense1);

        dense2 = Dense(pVal, activation=self.activation, use_bias=self.bias, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1(coeff))(dense1);
        self.show_layer_info('Dense', dense2);

        outputs = Dense(n_outputs, activation=self.output_layer_activation, kernel_initializer='glorot_normal')(dense2)#
        self.show_layer_info('Dense', outputs)

        model = Model(inputs=input, outputs=outputs)
        model.compile(loss='mse', optimizer='adam')
        return model;
    def train_DNN(self, model, X, y, job_callback):
        #num_sequences = len(y);
        #num_positives = np.sum(y);
        ###num_negatives = num_sequences - num_positives;
        #class_weight={True: num_sequences / num_positives, False: num_sequences / num_negatives}
        model.fit(X, y, epochs=self.num_epochs, batch_size=self.batch_size, callbacks=[job_callback],verbose=1);
        return model;
    
    class Job_finish_Callback(keras.callbacks.Callback):
        #Constructor
        def __init__(self, outputDir, pVal):
            self.pVal = pVal
            self.outputDir = outputDir
        def on_epoch_end(self, epoch, logs={}):
            print('on_epoch_end')
            h_local1_weight = np.array(self.model.layers[1].get_weights()[0]);
            h_local2_weight = np.array(self.model.layers[2].get_weights()[0]);

            print('h_local1_weight = ' + str(h_local1_weight.shape))
            print('h_local2_weight = ' + str(h_local2_weight.shape))
            h0 = np.zeros((self.pVal, 2));
            h0_abs = np.zeros((self.pVal, 2));

            for pIdx in range(self.pVal):
                h0[pIdx, :] = np.matmul(h_local1_weight[pIdx, :, :], h_local2_weight[pIdx, :, :]).flatten();
                h0_abs[pIdx, :] = np.matmul(np.fabs(h_local1_weight[pIdx, :, :]), np.fabs(h_local2_weight[pIdx, :, :])).flatten();
            
            print('h0 = ' + str(h0.shape))
            print('h0_abs = ' + str(h0_abs.shape))

            h1 = np.array(self.model.layers[4].get_weights()[0]);
            h2 = np.array(self.model.layers[5].get_weights()[0]);
            h3 = np.array(self.model.layers[6].get_weights()[0]);

            print('h1 = ' + str(h1.shape))
            print('h2 = ' + str(h2.shape))
            print('h3 = ' + str(h3.shape))

            W1 = h1;
            W_curr = h1;
            W2 = np.matmul(W_curr, h2);
            W_curr = np.matmul(W_curr, h2);
            W3 = np.matmul(W_curr, h3);

            print('W1 = ' + str(W1.shape))
            print('W2 = ' + str(W2.shape))
            print('W3 = ' + str(W3.shape))
            v0_h0 = h0[:, 0].reshape((self.pVal, 1));
            v1_h0 = h0[:, 1].reshape((self.pVal, 1));
            v0_h0_abs = h0_abs[:, 0].reshape((self.pVal, 1));
            v1_h0_abs = h0_abs[:, 1].reshape((self.pVal, 1));

            #v1 = np.vstack((v0_h0_abs, v1_h0_abs)).T;
            #v2 = np.vstack((np.sum(np.square(np.multiply(v0_h0_abs, np.fabs(W2))), axis=1).reshape((self.pVal, 1)), np.sum(np.square(np.multiply(v1_h0_abs, np.fabs(W2))), axis=1).reshape((self.pVal, 1)))).T;
            v3 = np.vstack((np.sum(np.square(np.multiply(v0_h0_abs, np.fabs(W3))), axis=1).reshape((self.pVal, 1)),                                                 np.sum(np.square(np.multiply(v1_h0_abs, np.fabs(W3))), axis=1).reshape((self.pVal, 1)))).T;

            v5 = np.vstack((np.sum(np.multiply(v0_h0, W3), axis=1).reshape((self.pVal, 1)),
                        np.sum(np.multiply(v1_h0, W3), axis=1).reshape((self.pVal, 1)))).T;

            with open(os.path.join(self.outputDir, 'result_epoch'+ str(epoch+1) +'_featImport.csv'), "a+") as myfile:
                myfile.write(','.join([str(x) for x in v3.flatten()]) + '\n');
            with open(os.path.join(self.outputDir, 'result_epoch'+ str(epoch+1) +'_featWeight.csv'), "a+") as myfile:
                myfile.write(','.join([str(x) for x in v5.flatten()]) + '\n');
