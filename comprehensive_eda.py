import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = r'/mnt/workspace/xiaoyuzhou/dataset_algo2025'

def load_data(file_name):
    return pd.read_csv(os.path.join(DATA_DIR, file_name))

def check_missing(df, name):
    print(f"\n--- Missing Values in {name} ---")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.concat([missing, missing_pct], axis=1, keys=['Total', 'Percent'])
    print(missing_table[missing_table['Total'] > 0].sort_values('Total', ascending=False))

def analyze_distributions(df, cols, name):
    print(f"\n--- Distribution Analysis for {name} ---")
    for col in cols:
        if col in df.columns:
            print(f"\nFeature: {col}")
            print(df[col].describe())
            # 偏度与峰度
            if df[col].dtype in ['int64', 'float64']:
                print(f"Skewness: {df[col].skew():.2f}")
                print(f"Kurtosis: {df[col].kurt():.2f}")

def find_outliers(df, col):
    if col not in df.columns: return
    print(f"\n--- Outlier Detection for {col} ---")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    print(f"Number of Outliers: {len(outliers)} ({100 * len(outliers) / len(df):.2f}%)")
    if len(outliers) > 0:
        print("Sample Outlier Values:", outliers[col].head().values)

def analyze_categorical(df, cols, name):
    print(f"\n--- Categorical Distribution for {name} ---")
    for col in cols:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True)
            print(f"\nTop 10 categories in {col}:")
            print(counts.head(10))
            # 检查严重分布不均（头部效应）
            if counts.iloc[0] > 0.5:
                print(f"WARNING: Feature {col} is highly imbalanced (Top category: {counts.index[0]} at {counts.iloc[0]*100:.2f}%)")

def analyze_target_correlation(train, user_feat, ep_feat):
    print("\n--- Target Correlation (Label=1 vs Label=0) ---")
    # 合并少量数据进行分析
    df = train.sample(100000, random_state=42) # 抽样分析加速
    df = df.merge(user_feat[['uid', 'age', 'sex']], on='uid', how='left')
    
    # 类别特征与 Label 的关系
    for col in ['tab_name', 'scene_name', 'entrance_type']:
        print(f"\nMean Label by {col}:")
        print(df.groupby(col)['label'].mean().sort_values(ascending=False))

def main():
    print("Loading datasets for comprehensive analysis...")
    train = load_data('train.csv')
    user_feat = load_data('user_feature.csv')
    ep_feat = load_data('episode_feature.csv')

    # 1. 缺失值检查
    check_missing(train, "Train")
    check_missing(user_feat, "User Feature")
    check_missing(ep_feat, "Episode Feature")

    # 2. 数值分布与异常值
    analyze_distributions(user_feat, ['age'], "User Feature")
    find_outliers(user_feat, 'age')
    
    analyze_distributions(ep_feat, ['duration'], "Episode Feature")
    find_outliers(ep_feat, 'duration')

    # 3. 类别分布不均
    analyze_categorical(train, ['tab_name', 'scene_name', 'entrance_type'], "Train")
    analyze_categorical(user_feat, ['address', 'sex', 'rg_source'], "User Feature")
    analyze_categorical(ep_feat, ['first_category', 'language'], "Episode Feature")

    # 4. 潜在规律探索 (特征与目标的关系)
    analyze_target_correlation(train, user_feat, ep_feat)

if __name__ == '__main__':
    main()
