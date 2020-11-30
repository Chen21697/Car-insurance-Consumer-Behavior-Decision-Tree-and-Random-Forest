#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:32:15 2020

@author: yuwenchen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
trainSet_x = pd.DataFrame(pd.read_csv('pa4_train_X.csv'))
trainSet_y = pd.DataFrame(pd.read_csv('pa4_train_y.csv', header = None))

valSet_x = pd.DataFrame(pd.read_csv('pa4_dev_X.csv'))
valSet_y = pd.DataFrame(pd.read_csv('pa4_dev_y.csv', header = None))

lenFeature = len(trainSet_x.columns)
sampleNum = len(trainSet_x)
validNum = len(valSet_x)

df = pd.concat((trainSet_x, trainSet_y), axis = 1)

df2 = pd.concat((valSet_x, valSet_y), axis = 1)

cols_name = trainSet_x.columns
#%%
def entropy(y_true):
    length = len(y_true)
    
    if y_true.sum() == length or y_true.sum() == 0: # pure case
        return 0
    else:
        bin_list = np.bincount(y_true)
        p1 = bin_list[1]/length
        p0 = bin_list[0]/length
        return -p1*np.log2(p1)-p0*np.log2(p0)
 

def feature_condtional_entropy(x, y, root_entropy):
    '''
    input:
        x : one feature
        y : ground turth
        root_entropy: H(S)
    output:
        the condtional entropy
    '''
                
    left_x = np.array([], dtype = int)
    left_y = np.array([], dtype = int)
    right_x = np.array([], dtype = int)
    right_y = np.array([], dtype = int)
    
    for (pred, ture) in zip(x, y):
        if pred == 0:
            left_x = np.append(left_x, pred)
            left_y = np.append(left_y, ture)
        elif pred == 1:
            right_x = np.append(right_x, pred)
            right_y = np.append(right_y, ture)

    
    # after splitting, if one node is the same as the rootnode, then the entropy is then 0
    if(left_x.size == 0 or right_x.size == 0):
        cond_entropy = 0
    else:
        l_entropy = entropy(left_y)
        r_entropy = entropy(right_y)
        
        l_p = len(left_x)/len(y)
        l_r = len(right_x)/len(y)
        cond_entropy = root_entropy - l_p*l_entropy - l_r*r_entropy
    
    return cond_entropy 

def find_best_feature(cols_name_list, x, y, root_e):
    

    cmp_list = []
    for i in cols_name_list:
        #print(i)
        feature_x = x[i].to_numpy()
        cmp_list.append(feature_condtional_entropy(feature_x, y, root_e))
    
    featureName = cols_name_list[cmp_list.index(max(cmp_list))] # name of feature with largest information gain
    return featureName, max(cmp_list)

def split(f_name, data):
    
    #left node
    left_data = data.where(data[f_name] == 0).dropna().astype(int)
    
    #right node
    right_data = data.where(data[f_name] == 1).dropna().astype(int)
            
    return left_data, right_data 
        
#%%
class Node:
    def __init__(self, data, testData, leaf = None):
        self.left = None
        self.right = None
        
        # training set
        self.data = data # whole data
        self.X = self.data.iloc[:, 0:lenFeature] # only training
        self.y = self.data.iloc[:, -1] # only labeled
        
        #validation set
        self.data2 = testData
        self.X2 = self.data2.iloc[:, 0:lenFeature] # only training
        self.y2 = self.data2.iloc[:, -1] # only labeled
        
        self.leaf = False # pure or not
        self.entropy = entropy(self.y) # entropy of this node
        self.feature = None
        self.infor_gain = None
        self.pred_value = 0

        
class decisionTree():
    def __init__(self, maxDepth, data, testData):
        self.maxDepth = maxDepth
        self.data = data
        self.testData = testData
        
        
    def buildTree(self):
        # build the root node
        self.rootNode = Node(self.data, self.testData)
        self.t_acc = []
        self.v_acc = []
        
        nodelist = []
        nodelist.append([self.rootNode])
        
        #nodeNum = 1
        depth = 1
        
        while depth <= self.maxDepth:
            print("Depth of tree is now:", depth)
            layer_list = [] # for each layer
            t_major = 0
            v_major = 0
            
            for i in nodelist[depth-1]:
                
                # caculate the majority of y_ture for training set
                temp_table = i.y.value_counts()
                max_num = temp_table.max()
                t_major = t_major + max_num
                
                # get predicted value
                pred_v = temp_table.index[temp_table == max_num].tolist()[0]
                
                if not i.y2.empty:
                    v_major = v_major + ((i.y2 == pred_v).sum())

                
                best_f_name, largest_info_gain = find_best_feature(cols_name, i.X, i.y, i.entropy)
                
                i.infor_gain = largest_info_gain
                i.feature = best_f_name
                
                left_data, right_data = split(best_f_name, i.data)
                left_data2, right_data2 = split(best_f_name, i.data2)
                
                # test if empty or not here
                # create left node
                if not left_data.empty:
                    #print("left!!")
                    i.left = Node(left_data, left_data2)
                    layer_list.append(i.left)
                    
                
                # create right node
                if not right_data.empty:
                    #print("right!!")
                    i.right = Node(right_data, right_data2)
                    layer_list.append(i.right)
                
            
            acc1 = t_major/sampleNum
            print("training acc:", acc1)
            acc2 = v_major/validNum
            print("validation acc:", acc2)

            self.t_acc.append(acc1)
            self.v_acc.append(acc2)
            
            nodelist.append(layer_list)
            depth = depth + 1
            
#%%
if __name__ == '__main__':
    d_Tree = decisionTree(5, df, df2)
    d_Tree.buildTree()
    