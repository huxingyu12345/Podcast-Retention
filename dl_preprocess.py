import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler

# 修改 1: 适配当前项目路径和数据结构
DATA_DIR = r'/mnt/workspace/xiaoyuzhou/dataset_algo2025'

def dl_preprocess():
    print("Loading split data (train_final, val_final) for DL...")
    # 修改 2: 直接加载 split_data.py 的输出，确保划分一致性
    train = pd.read_pickle('train_final.pkl')
    val = pd.read_pickle('val_final.pkl')
    test = pd.read_pickle('test_processed.pkl')
    
    # 加载原始数据以获取 title（之前的 pkl 中为了省内存可能删除了文本）
    ep_add = pd.read_csv(os.path.join(DATA_DIR, 'episode_additional.csv'))
    title_map = ep_add.set_index('episode_id')['title'].to_dict()

    # --- 1. 深度学习专用特征增强 (基于 EDA) ---
    print("Applying DL-specific feature refinements...")
    
    for df in [train, val, test]:
        # 意图特征构建 (基于 EDA 发现的 collections 强相关性)
        df['is_high_intent'] = ((df['tab_name'] == 'collections') | 
                                (df['entrance_type'] == 'collections') | 
                                (df['entrance_type'] == 'play_list')).astype(int)
        
        # 缺失标识特征 (在 preprocess.py 中可能已经处理过一部分，这里确保完整)
        if 'age_is_missing' not in df.columns:
            df['age_is_missing'] = (df['age'] <= 0).astype(int)
        
        # 获取标题文本用于向量映射
        df['title'] = df['episode_id'].map(title_map).fillna('unknown')

    # --- 2. 集成 BGE 语义向量 ---
    print("Integrating BGE embeddings...")
    bge_path = 'title_bge_embeddings.pkl'
    bge_features = []
    if os.path.exists(bge_path):
        with open(bge_path, 'rb') as f:
            title2vec = pickle.load(f)
        default_vec = np.zeros(512)
        
        # 为了控制维度，我们只取前 16 维，或者在实际项目中可以使用全量
        embed_dim = 16
        for i in range(embed_dim):
            col_name = f'title_bge_{i}'
            train[col_name] = train['title'].apply(lambda x: title2vec.get(x, default_vec)[i])
            val[col_name] = val['title'].apply(lambda x: title2vec.get(x, default_vec)[i])
            test[col_name] = test['title'].apply(lambda x: title2vec.get(x, default_vec)[i])
            bge_features.append(col_name)
    else:
        print("Warning: title_bge_embeddings.pkl not found. Skipping text embeddings.")

    # --- 3. 定义特征类型 ---
    # 从统一的 features.json 读取基础特征
    with open('features.json', 'r') as f:
        base_features = json.load(f)
    
    # 重新分类 Sparse 和 Dense
    # Sparse: 类别型 ID 及其它标识
    sparse_features = ['tab_name', 'scene_name', 'entrance_type', 'address', 'sex', 
                       'rg_source', 'first_category', 'language', 'rg_weekday', 
                       'rg_month', 'age_bin', 'age_is_missing', 'is_high_intent']
    
    # Dense: 连续数值型
    dense_features = [f for f in base_features if f not in sparse_features] + bge_features
    # 移除 ID 类
    dense_features = [f for f in dense_features if f not in ['uid', 'episode_id']]

    # --- 4. 缺失值填充与归一化 ---
    print("Handling missing values and Scaling...")
    for feat in dense_features:
        # 使用训练集中位数填充
        median_val = train[feat].median()
        train[feat] = train[feat].fillna(median_val)
        val[feat] = val[feat].fillna(median_val)
        test[feat] = test[feat].fillna(median_val)

    # 深度学习对量纲敏感，必须归一化
    scaler = StandardScaler()
    train[dense_features] = scaler.fit_transform(train[dense_features])
    val[dense_features] = scaler.transform(val[dense_features])
    test[dense_features] = scaler.transform(test[dense_features])

    # --- 5. 类别特征索引化 (Label Encoding) ---
    print("Re-indexing sparse features for Embedding layer...")
    feature_max_idx = {}
    for feat in sparse_features:
        # 合并所有数据以建立统一字典
        combined = pd.concat([train[feat], val[feat], test[feat]]).astype(str)
        unique_vals = combined.unique()
        val2idx = {val: i for i, val in enumerate(unique_vals)}
        
        train[feat] = train[feat].astype(str).map(val2idx)
        val[feat] = val[feat].astype(str).map(val2idx)
        test[feat] = test[feat].astype(str).map(val2idx)
        feature_max_idx[feat] = len(unique_vals)

    # --- 6. 保存结果 ---
    print("Saving dl_ready_data.pkl...")
    dl_data = {
        'train': train[sparse_features + dense_features + ['label', 'uid', 'episode_id']],
        'val': val[sparse_features + dense_features + ['label', 'uid', 'episode_id']],
        'test': test[sparse_features + dense_features + ['id']],
        'sparse_features': sparse_features,
        'dense_features': dense_features,
        'feature_max_idx': feature_max_idx
    }
    
    with open('dl_ready_data.pkl', 'wb') as f:
        pickle.dump(dl_data, f)
    
    print("DL Preprocessing Complete! Data split is consistent with GBDT experiments.")

if __name__ == '__main__':
    dl_preprocess()


if __name__ == '__main__':
    dl_preprocess()
