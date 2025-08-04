#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:10:06 2020

@author: hanli
"""
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Datas
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from numpy import interp
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import torch
import math
#from myeemd import myeemd
#import data_prepare as dp
#import filters


def dataset_loader(data, frac=0.7):
    train_data = DataFrame.sample(data,n=None, frac=frac, replace=False, weights=None, random_state=None, axis=None)
    indexs = list(train_data.index)
    train_label = train_data['label'].as_matrix()
    train_data = train_data.drop(['label'],axis = 1)

    test_data = data.drop(index=indexs, axis=1, inplace=False)
    test_label = test_data['label'].as_matrix()
    test_data = test_data.drop(['label'],axis = 1)

    train_label = torch.from_numpy(train_label.reshape(len(train_label))).type(torch.LongTensor)
    test_label = torch.from_numpy(test_label.reshape(len(test_label))).type(torch.LongTensor)
    train_data = torch.from_numpy(train_data.values.astype(float)).type(torch.FloatTensor).unsqueeze(1)
    test_data = torch.from_numpy(test_data.values.astype(float)).type(torch.FloatTensor).unsqueeze(1)

    Train_data = Datas.TensorDataset(train_data,train_label)
    Test_data = Datas.TensorDataset(test_data,test_label)
    train_loader = Datas.DataLoader(dataset=Train_data, batch_size=batch_size, shuffle=True)
    test_loader = Datas.DataLoader(dataset=Test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
#%%
def kappa(yls, yss, pres, yl_cnn, ys_cnn, pre):
    yls = yls
    yss = yss
    pres = pres.cpu()
    for i in range(len(yl_cnn)-1):
        yls = np.concatenate((yls, yl_cnn[i+1]), axis=0)
        yss = np.concatenate((yss, ys_cnn[i+1]), axis=0)
        pres = np.concatenate((pres, pre[i+1].cpu()), axis=0)
    kappa = cohen_kappa_score(yls, pres)
    print('kappa: ', kappa)

    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    classification = classification_report(yls, pres, target_names=target_names)
    print(classification)
#%%    
def ROC_curve_multi(yls, yss, yl_cnn, ys_cnn):     #多分類用
    yls = yls
    yss = yss
    for i in range(len(yl_cnn)-1):
        yls = np.concatenate((yls, yl_cnn[i+1]), axis=0)
        yss = np.concatenate((yss, ys_cnn[i+1]), axis=0)
        yls = label_binarize(yls, classes=[0, 1])
        n_classes = yls.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(yls[:, i], yss[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(yls.ravel(), yss.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
#%% 混淆矩陣繪製
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.YlGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    
#%%       
def ROC_curve_two(yls, yss, yl_cnn, ys_cnn):   #二分類用
    yls = yls
    yss = yss
    for i in range(len(yl_cnn)-1):
        yls = np.concatenate((yls, yl_cnn[i+1]), axis=0)
        yss = np.concatenate((yss, ys_cnn[i+1]), axis=0)
    fpr, tpr, _ = metrics.roc_curve(yls.ravel(), yss[:, 1].ravel())   #分類取分類正確且做扁平化
    roc_auc = metrics.auc(fpr, tpr)
     
    #曲線繪製
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Area under the curve of receiver operating curve')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, roc_auc
    
#%% open raw(實驗室讀檔程式)

def open_raw(path):
    Raw_Data = open(path,'rb').read()

    RAW = []
    for s in Raw_Data:
        RAW.append(s)

    header = RAW[0:512]             #RAW files header
#RAW = RAW[512:len(RAW)]         #RAW data

    header2 = []                    #RAW files header to String
    for s in header:
        header2.append(chr(s))

#%% Acquisition sampling rate ratio
    
    splr = header2[39:54]   
    splr2 = ''.join(splr)
    splr2 = float(splr2)        #Sampling Rate

    start = 55
    SRn = []
    for i in range(header[36]):     #Acquisition sampling rate ratio SRn
        SRtemp = header2[start+i*15+i:start+(i+1)*15+i]
        splrtemp = ''.join(SRtemp)
        splr_float = float(splrtemp)
        SRn.append(splr2/splr_float)

    maxi = int(max(SRn))
    
#%%  Find the Channel 


#matrix = np.zeros([maxi,header[36]])
    channel=[]
    for i in range(maxi) :
        for j in range(header[36]):
            #matrix[i][j] = i
            if i % SRn[j] == 0:
                channel.append(j)
                
#%% Data segmentation
#Data = np.zeros([header[36],])
    Data = []
    cont = []
    for i in range(header[36]):
        Data.append([])
        cont.append(1)
           
    Raw_Data = RAW[512:math.floor((len(RAW)-512)/len(channel))*len(channel)+512]
    for i in range(0, len(Raw_Data), len(channel)*2):
        for j in range(len(channel)):
            Data[channel[j]].append(Raw_Data[i + 2*j+1]*256+Raw_Data[i + 2*j])
    return Data


#with open (fname,'r', errors='ignore') as f:
#    content = f.readline(512)
# you may also want to remove whitespace characters like `\n` at the end of each line
#content = [x.strip() for x in content]    

    