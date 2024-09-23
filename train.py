import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
from model import get_model
import csv
import numpy as np
import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import csv
import tensorflow as tf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from model import get_model, MAX_LEN_mi, MAX_LEN_M
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RocAucCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true_train = self.model.predict(self.training_data[0])
        y_true_val = self.model.predict(self.validation_data[0])

        roc_auc_train = roc_auc_score(self.training_data[1], y_true_train)
        roc_auc_val = roc_auc_score(self.validation_data[1], y_true_val)

        logs['roc_auc_train'] = roc_auc_train
        logs['roc_auc_val'] = roc_auc_val

        print(f' - roc_auc_train: {roc_auc_train:.4f} - roc_auc_val: {roc_auc_val:.4f}')

class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.mi = val_data[0]
        self.lnc = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.mi,self.lnc])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        val_accuracy = logs.get('val_accuracy', 0.0)  # 默认为 0.0 如果没有找到
        filename = f"./model/2021bs64/{self.name}Model{epoch}_val_acc_{val_accuracy:.4f}.h5"
        self.model.save_weights(filename)
        # self.model.save_weights(
        #     "./model/2021bs64/%sModel%d.h5" % (self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
       
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    not_found_counter = 0  # 计数器，用于记录未找到特征的配对数量

    for i in range(len(InteractionPair)):
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]

        found1 = False
        for j in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[j][0]:
                SampleFeature1.append(NodeBehavior[j][1:])
                found1 = True
                break

        if not found1:
            print("Pair not found for Pair1:", Pair1)
            not_found_counter += 1

        found2 = False
        for k in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[k][0]:
                SampleFeature2.append(NodeBehavior[k][1:])
                found2 = True
                break

        if not found2:
            print("Pair not found for Pair2:", Pair2)
            not_found_counter += 1

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    print("Number of pairs not found:", not_found_counter)

    return SampleFeature1, SampleFeature2

def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

name = 'premiM'
Data_dir='data/'
train = np.load(Data_dir+'train.npz')
val=np.load(Data_dir+'val.npz')
X_mi_tra, X_M_tra, y_tra = train['X_mi_tra'], train['X_M_tra'], train['y_tra']

X_mi_val,X_M_val, y_val=val['X_mi_val'], val['X_M_val'], val['y_val']

PositiveSample_Train = []
ReadMyCsv1(PositiveSample_Train, Data_dir+'PositiveSample_Train.csv')
PositiveSample_Validation = []
ReadMyCsv1(PositiveSample_Validation, Data_dir+'PositiveSample_Validation.csv')
PositiveSample_Test = []
ReadMyCsv1(PositiveSample_Test, Data_dir+'PositiveSample_Test.csv')

NegativeSample_Train = []
ReadMyCsv1(NegativeSample_Train, Data_dir+'NegativeSample_Train.csv')
NegativeSample_Validation = []
ReadMyCsv1(NegativeSample_Validation, Data_dir+'NegativeSample_Validation.csv')
NegativeSample_Test = []
ReadMyCsv1(NegativeSample_Test, Data_dir+'NegativeSample_Test.csv')

x_train_pair = []
x_train_pair.extend(PositiveSample_Train)
x_train_pair.extend(NegativeSample_Train)
x_validation_pair = []
x_validation_pair.extend(PositiveSample_Validation)
x_validation_pair.extend(NegativeSample_Validation)
AllNodeBehavior = []
ReadMyCsv1(AllNodeBehavior, 'AllNodeBehavior.csv')


x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)

behavior_train_1 = x_train_1_Behavior
behavior_train_2 = x_train_2_Behavior
behavior_val_1 = x_validation_1_Behavior
behavior_val_2 = x_validation_2_Behavior
len_behavior1 = behavior_train_1.shape[1]
len_behavior2 = behavior_train_2.shape[1]
model_dir = 'my_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'model_90.h5')

# model = get_model(len_behavior1, len_behavior2)
# features_train = model.predict([X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2])
# features_val = model.predict([X_mi_val, X_M_val, behavior_val_1, behavior_val_2])
model = get_model(len_behavior1, len_behavior2)
model.summary()
checkpoint_callback = ModelCheckpoint(
    model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_freq='epoch',
    verbose=1
)

roc_auc_callback = RocAucCallback()
roc_auc_callback.training_data = ([X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2], y_tra)
roc_auc_callback.validation_data = ([X_mi_val, X_M_val, behavior_val_1, behavior_val_2], y_val)
history = model.fit(
    [X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2],
    y_tra,
    validation_data=([X_mi_val, X_M_val, behavior_val_1, behavior_val_2], y_val),
    epochs=10, 
    batch_size=32,
    callbacks=[checkpoint_callback, roc_auc_callback]
)
# model = get_model(len_behavior1, len_behavior2)
# checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
# history = model.fit([X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2], y_tra,
#                     validation_data=([X_mi_val, X_M_val, behavior_val_1, behavior_val_2], y_val),
#                     epochs=50, batch_size=32, callbacks=[checkpoint])

best_model = keras.models.load_model(model_path)
train_pred = best_model.predict([X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2])
val_pred = best_model.predict([X_mi_val, X_M_val, behavior_val_1, behavior_val_2])



train_auc = roc_auc_score(y_tra, train_pred)
val_auc = roc_auc_score(y_val, val_pred)
print(f"训练集 AUC: {train_auc:.4f}, 验证集 AUC: {val_auc:.4f}")

# # 训练随机森林分类器
# rf_classifier = RandomForestClassifier(n_estimators=100)
# rf_classifier.fit(features_train, y_tra)
#
# # 保存随机森林模型
# model_save_dir = './model_save_dir'
# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir)
# joblib.dump(rf_classifier, os.path.join(model_save_dir, 'rf_model.pkl'))
#
# # 使用训练好的随机森林进行预测
# train_pred = rf_classifier.predict(features_train)
# val_pred = rf_classifier.predict(features_val)
#
# # 可选：计算一些性能指标
# train_auc = roc_auc_score(y_tra, train_pred)
# val_auc = roc_auc_score(y_val, val_pred)
# print("训练集 AUC: {:.4f}, 验证集 AUC: {:.4f}".format(train_auc, val_auc))
#
# t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# print("开始时间: "+t1+" 结束时间："+t2)