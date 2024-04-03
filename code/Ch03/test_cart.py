"""
Author: Erutaner
Date: 2024.03.29
"""
from trees_cart import createDataSet, splitDataSet, createTree,classify,calcGini
import pandas as pd
import numpy as np
from utils import read_dataset, df_train2list, df_test2list, split_data_into_chunks
from treePlotter import createPlot
import random

# dataSet = [[1, 0.1, 1, 'no'],
#            [1, 0.3, 1, 'yes'],
#            [1, 0.4, 0, 'no'],
#            [0, 0.6, 1, 'yes'],
#            [0, 1.7, 1, 'yes'],
#            [1, 1.8, 1, 'yes'],
#            [1, 0.35, 1, 'no'],
#            [0, 1.85, 0, 'no']]
# labels = ['no surfacing', 'continue_feature', 'flippers']
###########################################################################单个cart决策树
# labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# df = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\train.csv")
# # df.info()
# train_data = df_train2list(df)
# labels_train = labels[:]
# # threshold为60是精确度95%
# tree_test = createTree(train_data,labels_train,threshold=60)
# df_test = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\test.csv")
# test_data = df_test2list(df_test)
# df_gt = pd.read_csv(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\gender_submission.csv",index_col=0)
# list_gt = df_gt['Survived'].tolist()
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
###########################################################################单个cart决策树

########################################### cart决策森林
#
# labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# df = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\train.csv")
# # df.info()
# train_data = df_train2list(df)
# tree_list = []
# # 41和17得到93的精度
# for i in range(21):
#     # 计算列表长度的70%
#     num_to_select = int(len(train_data) * 0.8)  # 需要选取的元素数量
#     # 随机选取70%的元素
#     selected_items = random.sample(train_data, num_to_select)
#     labels_train = labels[:]
#     tree_test = createTree(selected_items, labels_train, threshold=70)
#     tree_list.append(tree_test)
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
#     if sum(pred_list) >= 9:
#         pred = 1
#     else:
#         pred = 0
#
#     if pred == gt:
#         right_num += 1
#     else:
#         continue
# print(f"The precision of a forest with 42 trees is: {right_num/len(test_data)}.")

########################################### cart决策森林

########################################## K折交叉验证
labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = read_dataset(r"D:\桌面\武大本科期间文件\大三下文件\应用机器学习\实验三\实验二资料\data\titanic8120\train.csv")
labels_train = labels[:]
labels_test = labels[:]
# df.info()
train_data = df_train2list(df)
# tree_test = createTree(train_data,labels_train,threshold=60)
sub_train_data = split_data_into_chunks(train_data,5)
precision_list = []
for i in range(6):
    labels_train = labels[:]
    test_data = sub_train_data[i]
    train_data = sub_train_data[:]
    train_data.remove(test_data)
    train_data = [sample for sublist in train_data for sample in sublist]
    tree_test = createTree(train_data, labels_train, threshold=90)
    right_num = 0
    for sample in test_data:
        pred = classify(tree_test, labels_test, sample)
        if pred == sample[-1]:
            right_num += 1
        else:
            continue
    precision_list.append(right_num/len(test_data))
print(f"The precision of 6-fold cross validation is: {sum(precision_list)/len(precision_list)}")
########################################## K折交叉验证