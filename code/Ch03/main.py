"""
Author: Erutaner
Date: 2024.04.01
"""
from trees import createTree
import pandas as pd
# df = pd.read_csv(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\gender_submission.csv",index_col=0)
# df.info()
# survived_list = df['Survived'].tolist()
# print(survived_list)

def createDataSet():
    dataSet = [[0,1,'no'],
               [1,1,'no'],
               [0,0,'no'],
               [1,0,'yes']]
    labels = ['feature1','feature2']
    #change to discrete values
    return dataSet, labels
dataSet, labels = createDataSet()
print(createTree(dataSet,labels))
