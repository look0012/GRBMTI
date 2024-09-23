# 导入必要的库
import numpy as np
import pandas as pd
import csv
import math
import random

# 设置CSV字段大小限制，以适应读取文件时可能出现的大字段
#csv.field_size_limit(500 * 1024 * 1024)

# 从文件中读取CSV数据，并将其附加到给定的列表中

def read_csv(save_list, file_name):
    # 使用 with 语句来确保文件正确关闭
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            save_list.append(row)

# 将数据存储为CSV文件
def store_csv(data, file_name):
    # 指定文件编码为 'utf-8-sig' 来支持包含特殊字符的数据
    with open(file_name, mode="w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

# 根据给定的关系数据生成负样本
def generate_negative_sample(relationship_pd):
    relationship_matrix = pd.pivot_table(relationship_pd, index='Pair1', columns='Pair2', values='Rating', fill_value=0)
    negative_sample = []
    counter = 0
    while counter < len(relationship_pd):
        print(counter)
        temp_1 = random.randint(0, len(relationship_matrix.index) - 1)
        temp_2 = random.randint(0, len(relationship_matrix.columns) - 1)
        if relationship_matrix.iloc[temp_1, temp_2] == 0:
            relationship_matrix.iloc[temp_1, temp_2] = -1
            row = []
            row.append(np.array(relationship_matrix.index).tolist()[temp_1])
            row.append(np.array(relationship_matrix.columns).tolist()[temp_2])
            negative_sample.append(row)
            counter = counter + 1
        else:
            pass
    return negative_sample, relationship_matrix

# 以下代码块仅在直接运行此脚本时执行（而不是作为模块导入时）
if __name__ == '__main__':

    relationship = []  # 存储关系数据的列表
    read_csv(relationship, 'output1.csv')  # 从CSV文件中读取数据
    random.shuffle(relationship)  # 随机打乱关系数据
    relationship_train = relationship[0: int(0.7 * len(relationship))]  # 划分训练数据

    # 将训练数据存储为正样本的CSV文件
    store_csv(relationship_train, 'PositiveSample_Train.csv')

    # 划分验证和测试数据
    relationship_validation = relationship[int(0.7 * len(relationship)):int(0.8 * len(relationship))]
    relationship_test = relationship[int(0.8 * len(relationship)):]

    # 将验证和测试数据存储为正样本的CSV文件
    store_csv(relationship_validation, 'PositiveSample_Validation.csv')
    store_csv(relationship_test, 'PositiveSample_Test.csv')

    # 从关系数据创建Pandas DataFrame，并添加“Rating”列
    relationship_pd = pd.DataFrame(relationship, columns=['Pair1', 'Pair2'])
    relationship_pd['Rating'] = [1] * len(relationship_pd)

    # 生成负样本并得到相应的关系矩阵
    negative_sample, relationship_matrix = generate_negative_sample(relationship_pd)
    # 将关系矩阵存储为CSV文件
    relationship_matrix.to_csv('Relationship_Matrix.csv')

    # 将生成的负样本存储为CSV文件
    store_csv(negative_sample, 'NegativeSample.csv')

    # 将负样本划分为训练、验证和测试集
    negative_sample_train = negative_sample[0: int(0.7 * len(negative_sample))]
    negative_sample_validation = negative_sample[int(0.7 * len(negative_sample)):int(0.8 * len(negative_sample))]
    negative_sample_test = negative_sample[int(0.8 * len(negative_sample)):]

    # 将训练、验证和测试负样本存储为CSV文件
    store_csv(negative_sample_train, 'NegativeSample_Train.csv')
    store_csv(negative_sample_validation, 'NegativeSample_Validation.csv')
    store_csv(negative_sample_test, 'NegativeSample_Test.csv')
