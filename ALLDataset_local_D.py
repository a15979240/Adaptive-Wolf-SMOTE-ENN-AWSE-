import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.decomposition import PCA
import random
pca=PCA(n_components=2)


from sklearn import feature_selection as fs
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from collections import OrderedDict
from tabulate import tabulate
from sklearn.feature_selection import SelectFromModel



import os

drP=r"D:/KEEL/DAT"
# print(os.listdir(drP))
ddrp=os.listdir(drP)
print(ddrp)

def ALL_ATA(c=0):
    
    pb="D:/KEEL/DAT/"+ddrp[c]
    dpd=pd.read_table(pb,sep= ',')
    l=len(dpd.columns)
    # print(dpd)
    X=dpd.iloc[:,0:(l-1)]
    for i in range(l-1):
        if type(X.iloc[1,i])==str:
            X.iloc[:,i]=pd.Categorical(X.iloc[:,i],X.iloc[:,i].unique())
            X.iloc[:,i]=X.iloc[:,i].cat.rename_categories(range(len(X.iloc[:,i].value_counts())))
        else:
            continue

    # print(len(dpd.iloc[:,-1]))
    
    y=dpd.iloc[:,-1].str.replace(' ','')

    
    XDF=X
    # print(y.iloc[:])
    M=len(y[y.iloc[:]=='negative'])
    R=len(y[y.iloc[:]=='positive'])
    # print(M)
    # print(dpd.iloc[:,-1]=='positive')
    y=y.values
    if M<R:
        M=len(y[y[:]=='negative'])
        R=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=1
            else:
                y[i]=0
        
    else:
        R=len(y[y[:]=='negative'])
        M=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=0
            else:
                y[i]=1
    
    y=y.astype('int')
    
    X=X.values
    # print(type(X[0,2]))
    return X,y,M,R,dpd,XDF
# 特徵選取後
def ALL_ATA_fs(c=0):
    
    pb="Y:/DATA/python/MLB/KEEL/DAT/"+ddrp[c]
    dpd=pd.read_table(pb,sep= ',')
    l=len(dpd.columns)
    # print(dpd)
    X=dpd.iloc[:,0:(l-1)]
    for i in range(l-1):
        if type(X.iloc[1,i])==str:
            X.iloc[:,i]=pd.Categorical(X.iloc[:,i],X.iloc[:,i].unique())
            X.iloc[:,i]=X.iloc[:,i].cat.rename_categories(range(len(X.iloc[:,i].value_counts())))
        else:
            continue

    # print(len(dpd.iloc[:,-1]))
    
    y=dpd.iloc[:,-1].str.replace(' ','')

    
    XDF=X
    
    # print(y.iloc[:])
    M=len(y[y.iloc[:]=='negative'])
    R=len(y[y.iloc[:]=='positive'])
    # print(M)
    # print(dpd.iloc[:,-1]=='positive')
    y=y.values
    if M<R:
        M=len(y[y[:]=='negative'])
        R=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=1
            else:
                y[i]=0
        
    else:
        R=len(y[y[:]=='negative'])
        M=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=0
            else:
                y[i]=1
    
    y=y.astype('int')
    XDF.columns=range(len(XDF.T))
    # Random Forest
    clf = RFR(n_estimators=100, random_state=0)
    clf.fit(XDF, y)
    # Random Forest feature importances()使用隨機森林特徵選取
    featureImportance = list(clf.feature_importances_)
    rankingDic = dict((XDF.columns[i], featureImportance[i]) for i in range(len(XDF.columns)))
    rankingDic = OrderedDict(sorted(rankingDic.items(), key=lambda t: t[1], reverse=True ))
    # 顯示相關性比率順序
    # tb=tabulate(rankingDic.items() , headers=['columnName', 'importance'], tablefmt='orgtbl')

    # 取出特徵
    model = SelectFromModel(clf, prefit=True)
    selectedFeatureIndices =model.get_support()
    # selectedFeatureColNames = list(XDF.columns[selectedFeatureIndices])
    # 特徵定位
    # print(np.where(selectedFeatureIndices==True)[0])
    AA=np.where(selectedFeatureIndices==True)[0]
    # print('fs特徵',AA,'FS特徵長: ',len(AA))
    # nv=list(rankingDic.keys())[1]
    # unql=AA
    
    # if len(AA)==1:
    #     AA=np.append(AA,[nv])
    # print(AA)
    # HDBSCAN
    X=X.iloc[:,AA]
    
    X=X.values
    # X=X.values
    # print(type(X[0,2]))
    return X,y,M,R,dpd,XDF

# X,y,M,R,dpd,XDF=ALL_ATA(c=0)
# print(R/M)M為少數類
# print(R,M)
# print(ddrp[0])


# 資料集過多，拆分三組
drP9_1=r"D:/KEEL/DAT9_1"
ddrp9_1=os.listdir(drP9_1)

drP9_2=r"D:/KEEL/DAT9_2"
ddrp9_2=os.listdir(drP9_2)

drP9_3=r"D:/KEEL/DAT9_3"
ddrp9_3=os.listdir(drP9_3)


def ALL_ATA9_1(c=0):
    # 取CSV檔案
    pb="D:/KEEL/DAT9_1/"+ddrp9_1[c]
    dpd=pd.read_table(pb,sep= ',')
    l=len(dpd.columns)
    # 取資料內容
    X=dpd.iloc[:,0:(l-1)]
    # 數值化
    for i in range(l-1):
        if type(X.iloc[1,i])==str:
            X.iloc[:,i]=pd.Categorical(X.iloc[:,i],X.iloc[:,i].unique())
            X.iloc[:,i]=X.iloc[:,i].cat.rename_categories(range(len(X.iloc[:,i].value_counts())))
        else:
            continue


    # 取特徵類別
    y=dpd.iloc[:,-1].str.replace(' ','')
    XDF=X
    


    M=len(y[y.iloc[:]=='negative'])
    R=len(y[y.iloc[:]=='positive'])

    y=y.values

    # 多少數類判斷
    if M<R:
        M=len(y[y[:]=='negative'])
        R=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=1
            else:
                y[i]=0
        
    else:
        R=len(y[y[:]=='negative'])
        M=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=0
            else:
                y[i]=1
       
    
    y=y.astype('int')
    
    X=X.values
    # print(type(X[0,2]))
    
    return X,y,M,R,dpd,XDF

def ALL_ATA9_2(c=0):
    # print(ddrp9[c])
    pb="D:/KEEL/DAT9_2/"+ddrp9_2[c]
    dpd=pd.read_table(pb,sep= ',')
    l=len(dpd.columns)
    # print(dpd)
    X=dpd.iloc[:,0:(l-1)]
    for i in range(l-1):
        if type(X.iloc[1,i])==str:
            X.iloc[:,i]=pd.Categorical(X.iloc[:,i],X.iloc[:,i].unique())
            X.iloc[:,i]=X.iloc[:,i].cat.rename_categories(range(len(X.iloc[:,i].value_counts())))
        else:
            continue

    # print(len(dpd.iloc[:,-1]))
    
    y=dpd.iloc[:,-1].str.replace(' ','')
    XDF=X
    

    # print(y.iloc[:])
    M=len(y[y.iloc[:]=='negative'])
    R=len(y[y.iloc[:]=='positive'])
    # print(M)
    # print(dpd.iloc[:,-1]=='positive')
    y=y.values
    if M<R:
        M=len(y[y[:]=='negative'])
        R=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=1
            else:
                y[i]=0
        
    else:
        R=len(y[y[:]=='negative'])
        M=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=0
            else:
                y[i]=1
       
    
    y=y.astype('int')
    
    X=X.values
    # print(type(X[0,2]))
    
    return X,y,M,R,dpd,XDF
def ALL_ATA9_3(c=0):
    # print(ddrp9[c])
    pb="D:/KEEL/DAT9_3/"+ddrp9_3[c]
    dpd=pd.read_table(pb,sep= ',')
    l=len(dpd.columns)
    # print(dpd)
    X=dpd.iloc[:,0:(l-1)]
    for i in range(l-1):
        if type(X.iloc[1,i])==str:
            X.iloc[:,i]=pd.Categorical(X.iloc[:,i],X.iloc[:,i].unique())
            X.iloc[:,i]=X.iloc[:,i].cat.rename_categories(range(len(X.iloc[:,i].value_counts())))
        else:
            continue

    # print(len(dpd.iloc[:,-1]))
    
    y=dpd.iloc[:,-1].str.replace(' ','')
    XDF=X
    

    # print(y.iloc[:])
    M=len(y[y.iloc[:]=='negative'])
    R=len(y[y.iloc[:]=='positive'])
    # print(M)
    # print(dpd.iloc[:,-1]=='positive')
    y=y.values
    if M<R:
        M=len(y[y[:]=='negative'])
        R=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=1
            else:
                y[i]=0
        
    else:
        R=len(y[y[:]=='negative'])
        M=len(y[y[:]=='positive'])
        for i in range(len(dpd)):
            if y[i]=='negative':
                y[i]=0
            else:
                y[i]=1
       
    
    y=y.astype('int')
    
    X=X.values
    # print(type(X[0,2]))
    
    return X,y,M,R,dpd,XDF

X,y,M,R,dpd,XDF=ALL_ATA9_3(c=0)

print(dpd)

