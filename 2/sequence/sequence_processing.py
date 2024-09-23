

import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr) - 5):
            if 'N' in sr[i:i + 6]:
                tmp.append('null')
            else:
                tmp.append(sr[i:i + 6])
        word_seq.append(' '.join(tmp))
    return word_seq

def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq

def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq

def get_tokenizer():
    f = ['A', 'C', 'G', 'T']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0] + i[1] + i[2] + i[3] + i[4] + i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4096
    tokenizer = Tokenizer(num_words=NB_WORDS, lower=False)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer

def get_data(miRna, mRNA):
    tokenizer = get_tokenizer()
    MAX_LEN = 30
    X_mi = sentence2num(miRna, tokenizer, MAX_LEN)
    MAX_LEN = 4000
    X_inc = sentence2num(mRNA, tokenizer, MAX_LEN)
    return X_mi, X_inc


def process_file(file_name, data_dir, tokenizer):
    file_path = data_dir + file_name
    with open(file_path, 'r') as file:
        miRna_tra = []
        mRNA = []
        y_tra = []
        line_count = 0  # 行计数器
        for line in file:
            line_count += 1
            parts = line.strip().split(',')  # 使用逗号分隔
            if len(parts) < 7:
                print(f"警告: 文件 {file_name} 的第 {line_count} 行数据不完整。跳过此行。")
                continue
            try:
                label = int(parts[-1])
            except ValueError:
                print(f"错误: 文件 {file_name} 的第 {line_count} 行标签不是有效数字。标签值：{parts[-1]}，跳过此行。")
                continue
            miRna_tra.append(parts[4].replace('U', 'T'))
            mRNA.append(parts[5])
            y_tra.append(label)

    print(f'处理文件: {file_name}')
    print('正样本数量:' + str(sum(y_tra)))
    print('负样本数量:' + str(len(y_tra) - sum(y_tra)))

    X_mi_tra, X_M_tra = get_data(miRna_tra, mRNA)

    # 根据文件名选择变量名和保存路径
    if 'train_org' in file_name:
        np.savez(data_dir + 'train_org.npz', X_mi_tra=X_mi_tra, X_M_tra=X_M_tra, y_tra=y_tra)
    elif 'case_study2' in file_name:
        np.savez(data_dir + 'case_study2.npz', X_mi_test=X_mi_tra, X_M_test=X_M_tra, y_test=y_tra)
    elif 'val_org' in file_name:
        np.savez(data_dir + 'val_org.npz', X_mi_val=X_mi_tra, X_M_val=X_M_tra, y_val=y_tra)

    print(f"{file_name} 数据集处理成功")


# 主程序
Data_dir = 'data/'
# file_names = ['train_org.fa', 'test_org.fa', 'val_org.fa']
file_names=['case_study2.fa']
tokenizer = get_tokenizer()

for file_name in file_names:
    process_file(file_name, Data_dir, tokenizer)





# testnames = ['aly','mtr','stu','bdi']
# for name in testnames:
#     miRna_tes = []
#     lncRna_tes = []
#     y_tes = []
#     temp = []
#     with open(test_dir + '%s-TestSetH.fasta'%name,'r') as a:
#         for line in a:
#             line = line.strip()
#             temp.append(line.split(',')[2])
#             lncRna_tes.append(line.split(',')[3])
#             y_tes.append(int(line.split(',')[6]))
#     for c in temp:
#         miRna_tes.append(c.replace('U','T'))
#
#     print('测试集')
#     print('pos_samples:'+ str(int(sum(y_tes))))
#     print('neg_samples:'+ str(len(y_tes)-int(sum(y_tes))))
#
#     X_mi_tes,X_lnc_tes=get_data(miRna_tes,lncRna_tes)
#     np.savez(Data_dir+'%stest2021.npz'%name,X_mi_tes=X_mi_tes,X_lnc_tes=X_lnc_tes,y_tes=y_tes)
#     print("%s success"%name)

#print(len(max(miRna_tra, key=len)))
#print(len(max(incRna_tra,key=len)))
#print(len(max(miRna_tes, key=len)))
#print(len(max(incRna_tes,key=len)))



