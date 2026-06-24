import pandas as pd
import numpy as np
import os
import gc

def split_data():
    print("Loading processed data for splitting...")
    train = pd.read_pickle('train_processed.pkl')
    
    # 定义特征列（排除 ID 和 Label）
    drop_cols = ['uid', 'episode_id', 'label', 'id']
    features = [f for f in train.columns if f not in drop_cols]
    target = 'label'
    
    print(f"Features for splitting: {features}")
    
    # 采用基于时间的划分策略 (Time-based Split)
    # 使用 user_tenure 作为时间代理
    print("Performing time-based split (80/20)...")
    train = train.sort_values('user_tenure', ascending=False).reset_index(drop=True)
    split_idx = int(len(train) * 0.8)
    
    train_data = train.loc[:split_idx].copy()
    val_data = train.loc[split_idx:].copy()
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # 保存划分结果
    print("Saving split data to disk...")
    train_data.to_pickle('train_final.pkl')
    val_data.to_pickle('val_final.pkl')
    
    # 额外保存特征列表，方便其他模型调用
    import json
    with open('features.json', 'w') as f:
        json.dump(features, f)
        
    print("Data splitting complete! Files saved: train_final.pkl, val_final.pkl, features.json")

if __name__ == '__main__':
    split_data()
