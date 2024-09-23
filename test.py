import os
import numpy as np
import keras
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import csv
from model import get_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

# 根据你的实际情况导入模型定义函数

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

# 加载数据
Data_dir='data/'
val=np.load(Data_dir+'val.npz')
X_mi_val,X_M_val, y_val=val['X_mi_val'], val['X_M_val'], val['y_val']

AllNodeBehavior = []
ReadMyCsv1(AllNodeBehavior, 'AllNodeBehavior_deepwalk.csv')
PositiveSample_Validation = []
ReadMyCsv1(PositiveSample_Validation, Data_dir+'PositiveSample_Validation.csv')
NegativeSample_Validation = []
ReadMyCsv1(NegativeSample_Validation, Data_dir+'NegativeSample_Validation.csv')
x_validation_pair = []
x_validation_pair.extend(PositiveSample_Validation)
x_validation_pair.extend(NegativeSample_Validation)

x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化评估指标列表
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
aupr_scores = []

# 开始交叉验证
for train_idx, val_idx in kfold.split(X_mi_val, y_val):
    X_mi_train, X_mi_val_fold = X_mi_val[train_idx], X_mi_val[val_idx]
    X_M_train, X_M_val_fold = X_M_val[train_idx], X_M_val[val_idx]
    y_train, y_val_fold = y_val[train_idx], y_val[val_idx]
    behavior_train_1, behavior_val_1 = x_validation_1_Behavior[train_idx], x_validation_1_Behavior[val_idx]
    behavior_train_2, behavior_val_2 = x_validation_2_Behavior[train_idx], x_validation_2_Behavior[val_idx]

    # 加载最佳模型
    model_dir = 'my_model'
    model_path = os.path.join(model_dir, 'model_deepwalk.h5')
    best_model = keras.models.load_model(model_path)

    # 预测验证集
    val_pred = best_model.predict([X_mi_val_fold, X_M_val_fold, behavior_val_1, behavior_val_2])

    # 计算评估指标
    threshold = 0.5  # 二分类阈值，根据需要调整
    val_pred_binary = (val_pred > threshold).astype(int)

    accuracy = accuracy_score(y_val_fold, val_pred_binary)
    f1 = f1_score(y_val_fold, val_pred_binary)
    roc_auc = roc_auc_score(y_val_fold, val_pred)
    precision, recall, _ = precision_recall_curve(y_val_fold, val_pred)
    aupr = auc(recall, precision)

    # 将评估指标存储到列表中
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)
    aupr_scores.append(aupr)

    # 输出当前折的评估结果
    print(f"当前折：准确率 (Accuracy): {accuracy:.4f}, F1 值 (F1 Score): {f1:.4f}, ROC AUC 值: {roc_auc:.4f}, Precision-Recall AUC 值 (AUPR): {aupr:.4f}")

# 输出所有折的平均评估结果
print(f"\n五折交叉验证平均结果：")
print(f"平均准确率 (Accuracy): {np.mean(accuracy_scores):.4f}")
print(f"平均F1 值 (F1 Score): {np.mean(f1_scores):.4f}")
print(f"平均ROC AUC 值: {np.mean(roc_auc_scores):.4f}")
print(f"平均Precision-Recall AUC 值 (AUPR): {np.mean(aupr_scores):.4f}")