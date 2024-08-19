import numpy as np
import pandas as pd
import random
random.seed (3167)

from sklearn.model_selection import train_test_split
from sklearn.model_selection  import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, average_precision_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
# from xgboost import XGBClassifier

clf = LogisticRegression()
clf = GaussianNB()
clf = RandomForestClassifier(bootstrap=True, n_estimators=198)
clf = SVC(kernel='linear', gamma='scale', C=0.01, probability=True)
clf = MLPClassifier(hidden_layer_sizes=(256,512,256))

# train_data_path = "./data/output_chrom_919back/multi/data_512bs_5gram_919feature.out.logfoldchange.csv"
train_data_path = "./data/output_6gram_0.84/multi_novivo/data_512bs_6gram_2feature.out.diff.csv"
# test_data_path = "./data/output_6gram_0.84/vivo/data_512bs_6gram_2feature.out.diff.csv"

train_Data  = pd.read_csv(train_data_path, sep='\t') 
# test_Data  = pd.read_csv(test_data_path, sep='\t') 

feature_str = list(train_Data.columns)[5:] # 删除特征 以外的列名
label_str = "Label"

feature_train = train_Data[feature_str]
# feature_test = test_Data[feature_str]
label_train = [1 if str =="driver" else 0 for  str  in train_Data[label_str]]
label_test = [1 if str =="driver" else 0 for  str  in test_Data[label_str]]

index = random.sample(range(0,len(feature_train)), int(len(feature_train)*0.9))
index_ = [i for i in range(0,len(feature_train)) if i not in index ]
feature_test= feature_train.iloc[index_] 
label_test = [label_train[i] for i in index_] 
feature_train= feature_train.iloc[index]
label_train = [label_train[i] for i in index] 


"""
功能：ROC曲线取最佳阈值
注意：如果是PR曲线，输入的变量为1-Recall_Rate， Precision_Rates, thresholds
"""
def DecThreshold(fpr, tpr, thresholds):
    distances = []
    for i in range(len(thresholds)):
        distances.append(((fpr[i]) ** 2 + (tpr[i] - 1) ** 2))
    a = distances.index(min(distances))
    threshold = thresholds[a]
    return threshold
"""
功能：按阈值四舍五入
"""   
def round_0_1(x, threshold):
    if x >= threshold:
        return np.ceil(x)
    else:
        return np.floor(x) 

# xgb Training
DM_Xy_train = xgb.DMatrix(feature_train, label_train) 
DM_Xy_test = xgb.DMatrix(feature_test, label_test)
num_round = 2000  # 迭代次数=100
l2 = 2000
l1 = 20
threads = 40
param = {# 'max_depth':8,
                    'booster': 'gbtree',  # booster': 'gblinear',
                    'alpha': l1,
                    'lambda': l2,
                    'eta': 0.1,
                    'objective':'binary:logistic',  # 'objective': 'reg:squarederror',
                    'nthread': threads,
                    'eval_metric': 'auc',
                    'verbosity': 0}

evallist = [(DM_Xy_train, 'train'), (DM_Xy_test, 'eval')]  # 评估性能过程，有前后顺序要求
raw_model = xgb.train(params=param,
                    dtrain=DM_Xy_train,
                    num_boost_round=num_round,
                    evals=evallist,
                    verbose_eval=True,        # 显示eval过程
                    # early_stopping_rounds=100  # 该参数无法更好体现出模型在test上的性能
                    )

# xgb Predicting
pred_test_raw = raw_model.predict(DM_Xy_test)
pred_test_int = np.ones(pred_test_raw.shape)  # 创建零向量

# False_Positive_Rates, True_Positive_Rates, thresholds = roc_curve(DM_Xy_test.get_label(), pred_test_raw, pos_label=1) 
# threshold = DecThreshold(False_Positive_Rates, True_Positive_Rates, thresholds)
threshold = 0.5
pred_test_int = np.array([round_0_1(i,threshold) for i in pred_test_raw ]) # 取预测为1的一列按阈值预测

print('acc:', accuracy_score(DM_Xy_test.get_label(), pred_test_int))
print('F1:', f1_score(DM_Xy_test.get_label(), pred_test_int))
print('AUROC:', roc_auc_score(DM_Xy_test.get_label(), pred_test_raw))
print("AUPRC", average_precision_score(DM_Xy_test.get_label(), pred_test_raw))

# # Saving
# out_model_path = "./data/output/CDM.model"
# raw_model.save_model(out_model_path)



# # others ML
# clf.fit(feature_train, label_train)
# pred_test_raw = clf.predict_proba(feature_test)[:,1]

# threshold = 0.5
# pred_test_int = np.array([round_0_1(i,threshold) for i in pred_test_raw ]) # 取预测为1的一列按阈值预测

# print('acc:', accuracy_score(label_test, pred_test_int))
# print('F1:', f1_score(label_test, pred_test_int))
# print('AUROC:', roc_auc_score(label_test, pred_test_raw))
# print("AUPRC", average_precision_score(label_test, pred_test_raw))


