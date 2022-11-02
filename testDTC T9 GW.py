import imp
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE ##imblearn採樣套件SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC #SVM中的算法SVC
from imblearn.ensemble import EasyEnsembleClassifier # EasyEnsemble
import matplotlib.pyplot as plt #繪圖用
from sklearn.metrics import accuracy_score #準確度
# from collections import Counter
from sklearn.tree import DecisionTreeClassifier #決策樹分類器
from sklearn.naive_bayes import GaussianNB #高斯貝是分類器
from sklearn.model_selection import train_test_split #傳統切分法
from sklearn.metrics import recall_score  #召回率
from sklearn.metrics import confusion_matrix   #混淆矩阵
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# from sklearn import metrics
from sklearn.decomposition import PCA #降維用，現在用不到
from imblearn.combine import SMOTETomek #imblearn採樣套件SMOTETomek

import random
from sklearn.neighbors import KNeighborsClassifier #KNN
# from sklearn.preprocessing import StandardScaler 
pca=PCA(n_components=2)
import _smote_variants2 as sv # 修改過的smote_variants套件
# import begin as bg

import ALLDataset_local_D as ADt #資料來源
from sklearn.model_selection import (RepeatedStratifiedKFold, KFold,
                                     cross_val_score, StratifiedKFold)
import sklearn.metrics as m
# 評估分數
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score #ROC
from sklearn.metrics import roc_curve #AUC
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
# from c45 import C45
import csv
from sklearn.neural_network import MLPClassifier #神經網路分類器
import os
# hdbscan聚類
# import hdbscan as hdb
# from hdbscan.flat import (HDBSCAN_flat,
#                           approximate_predict_flat,
#                           membership_vector_flat,
#                           all_points_membership_vectors_flat)
from sklearn import feature_selection as fs #特徵選取
from sklearn.ensemble import RandomForestRegressor as RFR #隨機森林回歸
from sklearn.ensemble import RandomForestClassifier as RFC #隨機森林分類器
from collections import OrderedDict
from tabulate import tabulate
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import f1_score
import statistics
import time
import warnings
warnings.filterwarnings("ignore")
#drPA=r"Y:/DATA/python/MLB/KEEL/DAT"
drPA=r"D:/KEEL/DAT9_1"#資料集來源
ddrpA=os.listdir(drPA)
lenda=len(ddrpA)
clN=['GWES_F_p5i7']
# clN=['GWES_F_p5i7','GWES_F_p5i3','GWES_F_p5i1']

gmea=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)
# ACC
Acc=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)

# F1
F1R=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)

# ROC
ROCR=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)


# KAPPA
KAPPAR=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)

#matthews
matR=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)


#recall
recallR=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)

#precision
precisionR=pd.DataFrame(np.random.randn(len(ddrpA),len(clN)),columns=clN,index=ddrpA)

p=1.0


for ds in range(lenda):
    X,y,M,R,dpd,XDF=ADt.ALL_ATA9_1(c=ds)
    # X,y,M,R,dpd,XDF=ADt.ALL_ATA(c=ds)
    
    # clf=[sv.GWOSMOTE_F_ENN(pack_size=5,iterations=7,n_jobs=2)
    # ,sv.GWOSMOTE_F_ENN(pack_size=5,iterations=3,n_jobs=2),
    # sv.GWOSMOTE_F_ENN(pack_size=5,iterations=1,n_jobs=2)]
    clf=[sv.GWOSMOTE_F_ENN(pack_size=5,iterations=7,n_jobs=2)]
    print(ddrpA[ds])
    GMMAX,ACCMAX,F1MAX,ROCMAX,KAPPAMAX,MATMAX,RECALLMAX,PREMAX=[],[],[],[],[],[],[],[]
    for cl in range(len(clf)):#分類器
        # t5=time.time()
        kfold = StratifiedKFold(min([M, 10]),shuffle=True,random_state=1)
        # print(X[:,np.where(selectedFeatureIndices==True)])
        for i in range(5):#循環5次，取多一點結果
            GMD,ACCD,F1D,ROCD,KAPPAD,MATD,RECALLD,PRED=[],[],[],[],[],[],[],[]
            for train_index, test_index in kfold.split(X,y):

                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]
                nX,ny=clf[cl].sample(X_train, y_train)
                # dt = DecisionTreeClassifier(random_state=2)
                dt=RFC()
                # dt=GaussianNB()
                dt.fit(nX, ny)
                y_pred = dt.predict(X_test)
                

                GMD.append(geometric_mean_score(y_test, y_pred))
                ACCD.append(accuracy_score(y_test, y_pred))
                F1D.append(f1_score(y_test, y_pred))
                ROCD.append(roc_auc_score(y_test, y_pred))
                KAPPAD.append(cohen_kappa_score(y_test, y_pred))
                MATD.append(matthews_corrcoef(y_test, y_pred))
                RECALLD.append(recall_score(y_test, y_pred))
                PRED.append(precision_score(y_test, y_pred))

            # 取平均    
            GMm=statistics.mean(GMD)
            ACCm=statistics.mean(ACCD)
            F1m=statistics.mean(F1D)
            ROCm=statistics.mean(ROCD)
            KAPPAm=statistics.mean(KAPPAD)
            MATm=statistics.mean(MATD)
            RECALLm=statistics.mean(RECALLD)
            PREm=statistics.mean(PRED)
            # 引入list
            GMMAX.append(GMm)
            ACCMAX.append(ACCm)
            F1MAX.append(F1m)
            ROCMAX.append(ROCm)
            KAPPAMAX.append(KAPPAm)
            MATMAX.append(MATm)
            RECALLMAX.append(RECALLm)
            PREMAX.append(PREm)

        #引入表，並取平均
        gmea.iloc[ds,cl]=statistics.mean(GMMAX)
        Acc.iloc[ds,cl]=statistics.mean(ACCMAX)
        F1R.iloc[ds,cl]=statistics.mean(F1MAX)
        ROCR.iloc[ds,cl]=statistics.mean(ROCMAX)
        KAPPAR.iloc[ds,cl]=statistics.mean(KAPPAMAX)
        matR.iloc[ds,cl]=statistics.mean(MATMAX)
        recallR.iloc[ds,cl]=statistics.mean(RECALLMAX)
        precisionR.iloc[ds,cl]=statistics.mean(PREMAX)
    # 製表
    gmea.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_GMEAN00X.csv')
    Acc.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_ACC00X.csv')
    F1R.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_F100X.csv')
    ROCR.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_ROC00X.csv')
    KAPPAR.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_KAPPA00X.csv')
    matR.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_MAT00X.csv')
    recallR.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_recall00X.csv')
    precisionR.to_csv('D:/0602RFC_T9GWSENN_1_p5i1_3_57_V1_PREC00X.csv')
gmea.to_csv('Y:/DATA/碩論/GW/GMEAN/0602RFC_T9GWSENN_1_p5i1_3_57_V1_GMEAN00X.csv')
Acc.to_csv('Y:/DATA/碩論/GW/ACC/0602RFC_T9GWSENN_1_p5i1_3_57_V1_ACC00X.csv')
F1R.to_csv('Y:/DATA/碩論/GW/F1/0602RFC_T9GWSENN_1_p5i1_3_57_V1_F100X.csv')
ROCR.to_csv('Y:/DATA/碩論/GW/ROC/0602RFC_T9GWSENN_1_p5i1_3_57_V1_ROC00X.csv')
KAPPAR.to_csv('Y:/DATA/碩論/GW/KAPPA/0602RFC_T9GWSENN_1_p5i1_3_57_V1_KAPPA00X.csv')
matR.to_csv('Y:/DATA/碩論/GW/matthews/0602RFC_T9GWSENN_1_p5i1_3_57_V1_MAT00X.csv')
recallR.to_csv('Y:/DATA/碩論/GW/recall/0602RFC_T9GWSENN_1_p5i1_3_57_V1_recall00X.csv')
precisionR.to_csv('Y:/DATA/碩論/GW/precision/0602RFC_T9GWSENN_1_p5i1_3_57_V1_PREC00X.csv')
