"""
Author: Erutaner
Date: 2024.03.29
"""
from trees import createDataSet, splitDataSet, createTree,classify,calcShannonEnt
import pandas as pd
import numpy as np
from utils import read_dataset, df_train2list, df_test2list, split_data_into_chunks
from treePlotter import createPlot


import random
################################################################################ 随机森林
# labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# df = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\train.csv")
# df.info()
# train_data = df_train2list(df)
# tree_list = []
# # 41和17得到93的精度
# for i in range(42):
#     # 计算列表长度的70%
#     num_to_select = int(len(train_data) * 0.8)  # 需要选取的元素数量
#     # 随机选取70%的元素
#     selected_items = random.sample(train_data, num_to_select)
#     labels_train = labels[:]
#     tree_test = createTree(selected_items, labels_train, threshold=30)
#     tree_list.append(tree_test)
#
#
# df_test = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\test.csv")
# test_data = df_test2list(df_test)
# df_gt = pd.read_csv(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\gender_submission.csv",index_col=0)
# list_gt = df_gt['Survived'].tolist()
#
# right_num = 0
# labels_test = labels[:]
# for sample, gt in zip(test_data, list_gt):
#     pred_list = []
#     for tree in tree_list:
#         pred = classify(tree, labels_test, sample)
#         pred_list.append(pred)
#     if sum(pred_list) >= 14:
#         pred = 1
#     else:
#         pred = 0
#
#     if pred == gt:
#         right_num += 1
#     else:
#         continue
# print(f"The precision of a forest with 21 trees is: {right_num/len(test_data)}.")
######################################################################################### 随机森林

########################################################## 单个ID3决策树
# labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# df = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\train.csv")
# # df.info()
# train_data = df_train2list(df)
# labels_train = labels[:]
# tree_test = createTree(train_data,labels_train,threshold=30)
#
df_test = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\test.csv")
test_data = df_test2list(df_test)
df_gt = pd.read_csv(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\gender_submission.csv",index_col=0)
list_gt = df_gt['Survived'].tolist()
# labels_test = labels[:]
#
# right_num = 0
# for sample,gt in zip(test_data,list_gt):
#     pred = classify(tree_test,labels_test,sample)
#     if pred == gt:
#         right_num += 1
#     else:
#         continue
# print(f"The precision of a single decision tree is: {right_num/len(test_data)}.")
#################################################################################### 单个ID3决策树

#################################################################################### 单个ID3决策树：K折交叉验证
labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\train.csv")
labels_train = labels[:]
labels_test = labels[:]
# df.info()
train_data = df_train2list(df)
tree_test = createTree(train_data,labels_train,threshold=30)
sub_train_data = split_data_into_chunks(train_data,8)
precision_list = []
for i in range(8):
    labels_train = labels[:]
    test_data = sub_train_data[i]
    train_data = sub_train_data[:]
    train_data.remove(test_data)
    train_data = [sample for sublist in train_data for sample in sublist]
    tree_test = createTree(train_data, labels_train, threshold=30)
    right_num = 0
    for sample in test_data:
        pred = classify(tree_test, labels_test, sample)
        if pred == sample[-1]:
            right_num += 1
        else:
            continue
    precision_list.append(right_num/len(test_data))
print(f"The precision of 8-fold cross validation is: {sum(precision_list)/len(precision_list)}")


#################################################################################### 单个ID3决策树：K折交叉验证

# print("The training set: ",train_data)
# labels_train = labels[:]
# tree_test = createTree(train_data,labels_train,threshold=30)
# print("The tree: ",tree_test)

# df_test = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\test.csv")
# test_data = df_test2list(df_test)
# df_gt = pd.read_csv(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\gender_submission.csv",index_col=0)
# list_gt = df['Survived'].tolist()
# labels_test = labels[:]
# right_num = 0
#
# for sample,gt in zip(test_data,list_gt):
#     pred = classify(tree_test,labels_test,sample)
#     if pred == gt:
#         right_num += 1
#     else:
#         continue
# print(f"The precision of this decision tree is: {right_num/len(test_data)}.")

right_num = 0
for gt in list_gt:
    if gt == random.randint(0, 1):
        right_num += 1
    else:
        continue
print(f"The precision of random guessing is: {right_num/len(list_gt)}")

# right_num = 0
# for gt in list_gt:
#     if gt == 1:
#         right_num += 1
#     else:
#         continue
# print(f"The precision of 瞎几把蒙 1 is: {right_num/len(list_gt)}")

# createPlot(tree_test)