import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import os
import gc
import json

def train_model():
    print("Loading split data for XGBoost...")
    train = pd.read_pickle('train_final.pkl')
    val = pd.read_pickle('val_final.pkl')
    test = pd.read_pickle('test_processed.pkl')

    with open('features.json', 'r') as f:
        features = json.load(f)
    target = 'label'

    print(f"Features used for training: {len(features)}")

    # 准备数据格式
    dtrain = xgb.DMatrix(train[features], label=train[target])
    dval = xgb.DMatrix(val[features], label=val[target])
    dtest = xgb.DMatrix(test[features])

    # 冷启动评估准备
    train_uids = set(train['uid'].unique())
    train_episodes = set(train['episode_id'].unique())
    is_cold_user = ~val['uid'].isin(train_uids)
    is_cold_episode = ~val['episode_id'].isin(train_episodes)

    del train
    gc.collect()

    # XGBoost 参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'nthread': -1,
        'tree_method': 'hist' # 使用直方图算法加速
    }

    print("Starting XGBoost training...")
    watchlist = [(dtrain, 'train'), (dval, 'valid')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # 验证集多维度评估
    val_pred = model.predict(dval)
    overall_auc = roc_auc_score(val[target], val_pred)
    print(f"\nOverall Validation AUC: {overall_auc:.4f}")

    if is_cold_user.any():
        cold_user_auc = roc_auc_score(val[target][is_cold_user], val_pred[is_cold_user])
        print(f"Cold User AUC: {cold_user_auc:.4f}")
    
    if is_cold_episode.any():
        cold_episode_auc = roc_auc_score(val[target][is_cold_episode], val_pred[is_cold_episode])
        print(f"Cold Episode AUC: {cold_episode_auc:.4f}")

    # 特征重要性
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values(by='importance', ascending=False)
    print("\nXGBoost Feature Importance (Top 10):")
    print(importance_df.head(10))

    # 测试集预测
    print("\nPredicting on test set...")
    test_pred = model.predict(dtest)
    
    result = pd.DataFrame({
        'id': test['id'],
        'label': test_pred
    })
    result.to_csv('result_xgb.csv', index=False)
    print("Results saved to result_xgb.csv")

if __name__ == '__main__':
    train_model()
