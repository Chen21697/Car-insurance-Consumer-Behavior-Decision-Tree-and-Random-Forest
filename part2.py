#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:51:17 2020

@author: yuwenchen
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)

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

def predict(node, row, depth):
    d_count = 0
    pred_value = node.pred_value
    cur_node = node
    
    while d_count < depth-1:
        #print(d_count)
        if row[cur_node.feature] == 0:
            if (cur_node.left is None):
                break
            else:
                cur_node = cur_node.left
        
        elif row[cur_node.feature] == 1:
            if(cur_node.right is None):
                break
            else:
                cur_node = cur_node.right

        
        pred_value = cur_node.pred_value
        d_count += 1
        
    return pred_value

#%%
class Node:
    def __init__(self, data, feature_num):
        self.left = None
        self.right = None
        self.feature_num = feature_num
        
        # training set
        self.data = data # whole data
        self.X = self.data.iloc[:, 0:-1] # only training
        self.y = self.data.iloc[:, -1] # only labeled
        
        self.entropy = entropy(self.y) # entropy of this node
        self.feature = None
        self.infor_gain = None
        self.pred_value = None

        
class decisionTree():
    def __init__(self, maxDepth, feature_num, data):
        self.maxDepth = maxDepth
        self.feature_num = feature_num
        self.data = data
        
    def buildTree(self):
        # build the root node
        self.rootNode = Node(self.data, self.feature_num)
        self.t_acc = []
        
        nodelist = []
        nodelist.append([self.rootNode])
        
        depth = 1
        
        while depth <= self.maxDepth:
            layer_list = [] # for each layer

            for i in nodelist[depth-1]:
                
                # caculate the majority of y_ture for training set
                temp_table = i.y.value_counts()
                max_num = temp_table.max()
                
                # get predicted value
                pred_v = temp_table.index[temp_table == max_num].tolist()[0]
                i.pred_value = pred_v

                # select m random features here
                m_features = np.random.choice(cols_name, self.feature_num, replace = False)
                best_f_name, largest_info_gain = find_best_feature(m_features, i.X, i.y, i.entropy)
                
                i.infor_gain = largest_info_gain
                i.feature = best_f_name
                
                left_data, right_data = split(best_f_name, i.data)
                
                # test if empty or not here 
                # create left node
                if not left_data.empty:
                    i.left = Node(left_data, i.feature_num)
                    layer_list.append(i.left)
                    
                
                # create right node
                if not right_data.empty:
                    i.right = Node(right_data, i.feature_num)
                    layer_list.append(i.right)
            
                del i.data
                del i.X
                del i.y
            
            nodelist.append(layer_list)
            depth = depth + 1
            
            
class randomForest():
    def __init__(self, ori_data, ori_test, T, m, dmax):
        self.ori_data = ori_data
        self.ori_test = ori_test
        self.T = T
        self.m = m
        self.dmax = dmax
        
        self.data_list = []
        self.tree_list = []
    
    def generateData(self):
        for i in range(self.T):
            #randomly sample another tree
            t_d = self.ori_data.sample(sampleNum, random_state=i, replace=True)
            
            t_d_x = t_d.iloc[:, 0:lenFeature]
            t_d_y = t_d.iloc[:, -1] 
           
            #randomly select feature based on for this dataset
            self.data_list.append(pd.concat((t_d_x, t_d_y), axis = 1))
    
    def generateTrees(self):
        for i in range(self.T):
            print("Building tree:", i+1)
            myTree = decisionTree(self.dmax, self.m ,self.data_list[i])
            myTree.buildTree()
            self.tree_list.append(myTree)
            
    def predict_valid(self, t, m):
        
        #for validaitonset
        ground_turth = self.ori_test.iloc[:, -1]
        gt = ground_turth.tolist()
        
        predict_list = []
        for i in range(t): # for every tree
            tempList = []
            for j in range(validNum): # for every row
                temp = predict(self.tree_list[i].rootNode, self.ori_test.iloc[j], self.dmax)
                tempList.append(temp)

            predict_list.append(np.array(tempList))
        
        
        vote_pred = np.zeros((validNum,), dtype=int)
        for i in predict_list:
            vote_pred = vote_pred + i
        
        
        result = []
        for i in vote_pred:
            if i >= t/2:
                result.append(1)
            else:
                result.append(0)
                
        # compare the result
        count = 0
        for i in range(len(gt)):
            if gt[i] == result[i]:
                count+=1

        return (count/validNum)
    
    def predict_train(self, t, m):
        
        #for validaitonset
        ground_turth = self.ori_data.iloc[:, -1]
        gt = ground_turth.tolist()
        
        predict_list = []
        for i in range(t): # for every tree
            tempList = []
            for j in range(sampleNum): # for every row
                temp = predict(self.tree_list[i].rootNode, self.ori_data.iloc[j], self.dmax)
                tempList.append(temp)

            predict_list.append(np.array(tempList))
        
        vote_pred = np.zeros((sampleNum,), dtype=int)
        for i in predict_list:
            vote_pred = vote_pred + i
        
        
        result = []
        for i in vote_pred:
            if i >= t/2:
                result.append(1)
            else:
                result.append(0)
                
        # compare the result
        count = 0
        for i in range(len(gt)):
            if gt[i] == result[i]:
                count+=1

        return (count/sampleNum)
#%%   
if __name__ == '__main__':
    depth = 2
    T = [10,20,30,40,50,60,70,80,90,100]
    m = [5,25,50,100]
    forest = []
    res_acc = []
    
    # build a forest with 100 tree
    for j in m:
        print("m:", j)
        myRF = randomForest(df, df2, 100, j, depth)
        myRF.generateData()
        myRF.generateTrees()
        forest.append(myRF)
    print("The forests have been buildt.")
    
    print("Predicting on the training set now.....")
    for i in range(len(m)):
        myRF = forest[i]
        
        temp = []
        for t in T:
            print("T:", t)
            result = myRF.predict_train(t,depth)
            print(result)
            temp.append(result)
        res_acc.append(temp)
       
    print("Predicting on the validation set now.....")
    v_res_acc = []
    for i in range(len(m)):
        myRF = forest[i]
        
        temp = []
        for t in T:
            print("T:",t)
            result = myRF.predict_valid(t,depth)
            print(result)
            temp.append(result)
        v_res_acc.append(temp)