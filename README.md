## 1. 环境要求

- Python 3.8+
- PyTorch 2.0+ (用于深度学习模型)
- 核心库: `pandas`, `numpy`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `tqdm`

## 3. 运行步骤

请按照以下顺序执行脚本，以确保数据流的完整性：

### 第零阶段：数据探索与分析 (EDA)
在建模前，运行以下脚本以深入理解数据分布与业务痛点：
```bash
python comprehensive_eda.py   # 综合数据探索（缺失值、分布、相关性）
python cold_start_analysis.py # 冷启动现状分析（用户/内容冷启动比例）
```

### 第一阶段：数据准备
1. **生成基础特征**:
   ```bash
   python preprocess.py
   ```
2. **执行统一划分** (固定训练/验证集):
   ```bash
   python split_data.py
   ```

### 第二阶段：模型实验 (GBDT)
运行以下脚本进行树模型训练，结果将保存为 `result_*.csv`:
```bash
python train_lgb.py   # LightGBM 实验
python train_xgb.py   # XGBoost 实验
python train_cat.py   # CatBoost 实验
```

### 第三阶段：深度学习实验 (Deep Learning)
1. **生成语义向量** (可选):
   
   ```bash
   python bge_embedding.py
   ```
2. **深度学习专用预处理**:
   ```bash
   python dl_preprocess.py
   ```
3. **运行 DL 模型**:
   ```bash
   python deepfm.py    # DeepFM
   python dcn_v2.py    # DCN-V2
   python autoint.py   # AutoInt (Transformer-based)
   ```


