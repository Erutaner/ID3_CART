"""
Author: Erutaner
Date: 2024.04.01
"""
import pandas as pd
import numpy as np


def split_data_into_chunks(data, n_chunks):
    """将数据列表分成n_chunks个均匀的子列表"""
    chunk_size = len(data) // n_chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    # 处理任何剩余的元素，将它们均匀地添加到前面的chunks中
    for i in range(len(data) % n_chunks):
        chunks[i].append(data[-(i+1)])
    return chunks

def read_dataset(fname):
    data = pd.read_csv(fname, index_col=0)

    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    data['Sex'] = (data['Sex'] == 'male').astype('int')

    mode_embarked = data['Embarked'].mode()[0]
    data['Embarked'].fillna(mode_embarked, inplace=True)
    labels = np.sort(data['Embarked'].unique()).tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))

    mean_age = data['Age'].mean()
    data['Age'].fillna(mean_age, inplace=True)

    mean_fare = data['Fare'].mean()
    data['Fare'].fillna(mean_fare, inplace=True)

    return data

def df_train2list(df):
    # 将'Survived'列移动到最后
    cols = list(df.columns)
    cols.append(cols.pop(cols.index('Survived')))
    df_1 = df[cols]
    # 将DataFrame转换为列表，其中每个元素是一个子列表
    data_list = df_1.values.tolist()
    columns_to_convert = [0, 1, 3, 4, 6, 7]
    # 将指定列数据转换为整型
    converted_matrix = [
    [
        int(item) if index in columns_to_convert else item
        for index, item in enumerate(row)
    ]
    for row in data_list
    ]
    return converted_matrix

# Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
def df_test2list(df):
    data_list = df.values.tolist()
    columns_to_convert = [0, 1, 3, 4, 6]
    # 将指定列数据转换为整型
    converted_matrix = [
        [
            int(item) if index in columns_to_convert else item
            for index, item in enumerate(row)
        ]
        for row in data_list
    ]
    return converted_matrix