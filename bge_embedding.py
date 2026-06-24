import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

def generate_bge_embeddings():
    print("Loading episode additional info...")
    data_dir = r'/mnt/workspace/xiaoyuzhou/dataset_algo2025'
    ep_add = pd.read_csv(os.path.join(data_dir, 'episode_additional.csv'))
    
    # 提取唯一标题
    ep_add['title'] = ep_add['title'].fillna('未知标题').astype(str)
    unique_titles = ep_add['title'].unique()
    print(f"Total unique titles: {len(unique_titles)}")

    try:
        # 1. 导入必要的库
        from modelscope import snapshot_download
        from sentence_transformers import SentenceTransformer

        # 2. 【核心改动】使用 ModelScope 从国内源下载模型到本地 (只需下载一次)
        print("Downloading BGE model from ModelScope (China mirror)...")
        # cache_dir 指定本地缓存目录，下次运行会直接读取，不会重复下载
        model_dir = snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='./model_cache')
        
        # 3. 从本地路径加载模型
        print("Loading model from local path...")
        model = SentenceTransformer(model_dir)
        
        print("Model loaded successfully. Starting encoding...")
        
        # 分批处理以优化内存
        batch_size = 128  # 如果后续遇到内存溢出 (OOM)，请将此值改小，如 64 或 32
        embeddings = []
        
        for i in tqdm(range(0, len(unique_titles), batch_size), desc="Encoding Titles"):
            batch_titles = unique_titles[i:i+batch_size].tolist()
            
            # 使用 sentence-transformers 进行编码
            batch_embeddings = model.encode(
                batch_titles, 
                normalize_embeddings=True,  # BGE 模型官方建议开启归一化
                convert_to_numpy=True,      # 确保返回 numpy 数组
                show_progress_bar=False     # 外层已经有 tqdm 了，这里关闭避免重复
            )
            embeddings.append(batch_embeddings)
        
        # 合并结果并转换为 numpy
        all_embeddings = np.vstack(embeddings)
        
        # 构建 标题 -> 向量 的映射
        title2vec = {title: vec for title, vec in zip(unique_titles, all_embeddings)}
        
        # 保存结果
        with open('title_bge_embeddings.pkl', 'wb') as f:
            pickle.dump(title2vec, f)
        
        print("✅ BGE Embeddings generated and saved to title_bge_embeddings.pkl")

    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: python -m pip install sentence-transformers modelscope")
    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == '__main__':
    generate_bge_embeddings()