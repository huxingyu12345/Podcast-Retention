import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import gc

import json

def train_model():
    print("Loading split data...")
    train = pd.read_pickle('train_final.pkl')
    val = pd.read_pickle('val_final.pkl')
    test = pd.read_pickle('test_processed.pkl')

    # 从统一配置文件读取特征
    with open('features.json', 'r') as f:
        features = json.load(f)
    target = 'label'

    print(f"Features used for training: {features}")

    # 准备训练和验证数据
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]

    # 冷启动评估准备
    train_uids = set(train['uid'].unique())
    train_episodes = set(train['episode_id'].unique())
    is_cold_user = ~val['uid'].isin(train_uids)
    is_cold_episode = ~val['episode_id'].isin(train_episodes)

    # 释放原始内存
    del train, val
    gc.collect()

    # 指定类别特征
    cat_features = ['tab_name', 'scene_name', 'entrance_type', 'address', 'sex', 'rg_source', 'first_category', 'language', 'rg_weekday', 'rg_month', 'age_bin']
    
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_features)

    # 改进 3: 增强模型参数的正则化与鲁棒性
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02, # 降低学习率以提高泛化
        'num_leaves': 63,      # 增加叶子节点提升复杂度，配合正则化
        'max_depth': -1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,      # L1 正则
        'lambda_l2': 0.1,      # L2 正则
        'min_data_in_leaf': 50, # 防止过拟合
        'verbose': 1,          # 修改为 1 以显示基础训练信息
        'seed': 42,
        'n_jobs': -1
    }

    print("Starting training...")
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=10) # 缩短打印周期，每 10 轮显示一次进度
        ]
    )

    # 验证集多维度评估
    val_pred = model.predict(X_val)
    overall_auc = roc_auc_score(y_val, val_pred)
    print(f"\nOverall Validation AUC: {overall_auc:.4f}")

    if is_cold_user.any():
        cold_user_auc = roc_auc_score(y_val[is_cold_user], val_pred[is_cold_user])
        print(f"Cold User AUC: {cold_user_auc:.4f}")
    
    if is_cold_episode.any():
        cold_episode_auc = roc_auc_score(y_val[is_cold_episode], val_pred[is_cold_episode])
        print(f"Cold Episode AUC: {cold_episode_auc:.4f}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values(by='importance', ascending=False)
    print("\nFeature Importance (Top 15):")
    print(importance.head(15))

    # 测试集预测
    print("\nPredicting on test set...")
    test_pred = model.predict(test[features])
    
    # 生成提交文件
    result = pd.DataFrame({
        'id': test['id'],
        'label': test_pred
    })
    result.to_csv('result_lgb.csv', index=False)
    print("Results saved to result.csv")

if __name__ == '__main__':
    train_model()
