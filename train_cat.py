import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import os
import gc
import json

def train_model():
    print("Loading split data for CatBoost...")
    train = pd.read_pickle('train_final.pkl')
    val = pd.read_pickle('val_final.pkl')
    test = pd.read_pickle('test_processed.pkl')

    with open('features.json', 'r') as f:
        features = json.load(f)
    target = 'label'

    # 定义类别特征列表
    cat_features = ['tab_name', 'scene_name', 'entrance_type', 'address', 'sex', 'rg_source', 'first_category', 'language', 'rg_weekday', 'rg_month', 'age_bin']
    # 确保类别特征是整数类型（CatBoost 要求）
    for col in cat_features:
        train[col] = train[col].astype(int)
        val[col] = val[col].astype(int)
        test[col] = test[col].astype(int)

    print(f"Features used for training: {len(features)}")

    # 冷启动评估准备
    train_uids = set(train['uid'].unique())
    train_episodes = set(train['episode_id'].unique())
    is_cold_user = ~val['uid'].isin(train_uids)
    is_cold_episode = ~val['episode_id'].isin(train_episodes)

    # 准备 CatBoost Pool
    train_pool = Pool(train[features], train[target], cat_features=cat_features)
    val_pool = Pool(val[features], val[target], cat_features=cat_features)
    test_pool = Pool(test[features], cat_features=cat_features)

    del train
    gc.collect()

    # CatBoost 参数
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type='CPU' # 如果有 GPU 可以改为 'GPU'
    )

    print("Starting CatBoost training...")
    model.fit(train_pool, eval_set=val_pool)

    # 验证集多维度评估
    val_pred = model.predict_proba(val_pool)[:, 1]
    overall_auc = roc_auc_score(val[target], val_pred)
    print(f"\nOverall Validation AUC: {overall_auc:.4f}")

    if is_cold_user.any():
        cold_user_auc = roc_auc_score(val[target][is_cold_user], val_pred[is_cold_user])
        print(f"Cold User AUC: {cold_user_auc:.4f}")
    
    if is_cold_episode.any():
        cold_episode_auc = roc_auc_score(val[target][is_cold_episode], val_pred[is_cold_episode])
        print(f"Cold Episode AUC: {cold_episode_auc:.4f}")

    # 特征重要性
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    print("\nCatBoost Feature Importance (Top 10):")
    print(importance_df.head(10))

    # 测试集预测
    print("\nPredicting on test set...")
    test_pred = model.predict_proba(test_pool)[:, 1]
    
    result = pd.DataFrame({
        'id': test['id'],
        'label': test_pred
    })
    result.to_csv('result_cat.csv', index=False)
    print("Results saved to result_cat.csv")

if __name__ == '__main__':
    train_model()
