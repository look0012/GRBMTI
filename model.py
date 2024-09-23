from keras import initializers
from keras.src.engine.input_spec import InputSpec
from keras.src.engine.base_layer import Layer
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras
import numpy as np
import tensorflow as tf

MAX_LEN_mi = 30
MAX_LEN_M = 4000
EMBEDDING_DIM = 64

kmers = 6
name='RNA2vec/'

if kmers == 6:
    NB_WORDS = 4096
    mRnaembedding_matrix = np.load(name+'6m.npy',allow_pickle=True)
    miRnaembedding_matrix = np.load(name+'6mi.npy',allow_pickle=True)




def get_model(len_behavior1, len_behavior2):
    # 输入定义
    miRna = Input(shape=(MAX_LEN_mi,))
    mRna = Input(shape=(MAX_LEN_M,))
    behavior1 = Input(shape=(len_behavior1,))
    behavior2 = Input(shape=(len_behavior2,))
    print(miRnaembedding_matrix.shape)
    # miRNA 和 mRNA 的嵌入和卷积处理
    emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[miRnaembedding_matrix], trainable=True)(miRna)
    emb_m = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[mRnaembedding_matrix], trainable=True)(mRna)
    # miRna_conv_layer = Convolution1D(filters=64, kernel_size=10, padding="same")(emb_mi)
    # mRna_conv_layer = Convolution1D(filters=64, kernel_size=40, padding="same")(emb_m)
    miRna_conv = Conv1D(filters=64, kernel_size=3, padding="same")(emb_mi)
    miRna_conv = Dropout(0.5)(miRna_conv)
    mRna_conv = Conv1D(filters=64, kernel_size=3, padding="same")(emb_m)
    mRna_conv = Dropout(0.5)(mRna_conv)
    # # 卷积层后的处理

    miRna_gru = Bidirectional(GRU(32, return_sequences=False))(miRna_conv)
    mRna_gru = Bidirectional(GRU(32, return_sequences=False))(mRna_conv)

    # GRU 和 注意力层
    # l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(miRna_out)
    # l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(mRna_out)


    # 特征合并
    # combined_features = Concatenate(axis=1)([l_gru_1, l_gru_2, behavior1, behavior2])


    # 创建 Keras 模型
    # model = Model(inputs=[miRna, mRna, behavior1, behavior2], outputs=combined_features)
    combined_features = Dense(256, activation='relu')(Concatenate()([miRna_gru, mRna_gru, behavior1, behavior2]))
    output = Dense(1, activation='sigmoid')(combined_features)

    # 创建模型，这里没有输出层，因为这是特征抽取模型
    model = Model(inputs=[miRna, mRna, behavior1, behavior2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    return model
