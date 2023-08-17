import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

train_data_path = "./data/output/multi_novivo/data_512bs_6gram_2feature.out.diff.csv"
test_data_path = "./data/output/vivo/data_512bs_6gram_2feature.out.diff.csv"

train_Data  = pd.read_csv(train_data_path, sep='\t') 
test_Data  = pd.read_csv(test_data_path, sep='\t') 

feature_str = list(train_Data.columns)[5:] # 删除特征 以外的列名
label_str = "Label"

train_Data = train_Data.sample(frac=1, random_state=0)  # 打乱
test_Data = test_Data.sample(frac=1, random_state=0) # 打乱

feature_train = train_Data[feature_str]
feature_test = test_Data[feature_str]
label_train = [1 if str =="driver" else 0 for  str  in train_Data[label_str]]
label_test = [1 if str =="driver" else 0 for  str  in test_Data[label_str]]

DM_Xy_train = xgb.DMatrix(feature_train, label_train)
DM_Xy_test = xgb.DMatrix(feature_test, label_test)

# Training
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

# Predicting
pred_test_raw = raw_model.predict(DM_Xy_test)
pred_test_int = np.ones(pred_test_raw.shape)  # 创建零向量

for i in range(len(pred_test_raw)):  # 取整
    if pred_test_raw[i] > 0.5:
        pred_test_int[i] = 1
    else:
        pred_test_int[i] = 0
print('acc:', accuracy_score(DM_Xy_test.get_label(), pred_test_int))
print('AUROC:', roc_auc_score(DM_Xy_test.get_label(), pred_test_raw))

# Saving
out_model_path = "./data/output/CDM.model"
raw_model.save_model(out_model_path)


# training
# clf = XGBClassifier(n_estimators=152, max_depth=5, learning_rate=0.01,
#                               subsample=0.878947, colsample_bytree=0.5, min_child_weight=1, 
#                               use_label_encoder=False, eval_metric=['logloss','auc','error']) # 256 、01 + 未寻参

# clf.fit(feature_train, label_train)

# # Predicting
# pred_test_raw = clf.predict_proba(feature_test)[:,1]
# pred_test_int = np.ones(pred_test_raw.shape) # 创建零向量

# for i in range(len(pred_test_raw)):  # 取整
#     if pred_test_raw[i] > 0.5:
#         pred_test_int[i] = 1
#     else:
#         pred_test_int[i] = 0

# print('acc:', accuracy_score(label_test, pred_test_int))
# print('AUROC:', roc_auc_score(label_test, pred_test_raw))

# # 网格搜索
# clf = XGBClassifier(n_estimators=152, max_depth=5, learning_rate=0.01,
#                               subsample=0.878947, colsample_bytree=0.5, min_child_weight=1, 
#                               use_label_encoder=False, eval_metric=['logloss','auc','error'])

# param_dist = {'n_estimators':range(80,200,4),
#                             'max_depth':range(2,15,1),
#                             'learning_rate':np.linspace(0.01,2,20),
#                             'subsample':np.linspace(0.7,0.9,20),
#                             'colsample_bytree':np.linspace(0.5,0.98,10),
#                             'min_child_weight':range(1,9,1) }  
# search_times = 100
# grid_search = RandomizedSearchCV(clf, param_dist, cv=3, scoring='roc_auc',n_iter= search_times, n_jobs = -1, verbose=50) 
# grid_search.fit(feature_train, label_train)
# print("best_param : {0}".format(grid_search.best_params_))
