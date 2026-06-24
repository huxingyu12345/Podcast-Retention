import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- Dataset Definition ---
class RecDataset(Dataset):
    def __init__(self, data, sparse_cols, dense_cols, label_col=None):
        self.sparse_data = data[sparse_cols].values.astype(np.int64)
        self.dense_data = data[dense_cols].values.astype(np.float32)
        self.label = data[label_col].values.astype(np.float32) if label_col else None

    def __len__(self):
        return len(self.sparse_data)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.sparse_data[idx], self.dense_data[idx], self.label[idx]
        return self.sparse_data[idx], self.dense_data[idx]

# --- DeepFM Model with SENet and Bilinear Interaction (Enhanced) ---
class SENet(nn.Module):
    """
    Squeeze-and-Excitation Network to learn feature importance dynamically.
    """
    def __init__(self, num_fields, reduction_ratio=2):
        super(SENet, self).__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(num_fields, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, num_fields, embed_dim]
        # Squeeze: pooling across embed_dim
        z = torch.mean(x, dim=-1) # [batch_size, num_fields]
        # Excitation: compute weights
        a = self.fc(z) # [batch_size, num_fields]
        # Re-weight
        return x * a.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    """
    Bilinear Interaction instead of simple inner product.
    """
    def __init__(self, num_fields, embed_dim):
        super(BilinearInteraction, self).__init__()
        self.num_fields = num_fields
        # Simplified bilinear: shared weight matrix across fields to save memory
        self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        # x shape: [batch_size, num_fields, embed_dim]
        # Compute bilinear interactions
        vid_w = torch.matmul(x, self.weight) # [batch_size, num_fields, embed_dim]
        
        # Interaction between fields i and j
        # We can use the sum-pooling trick here for efficiency
        # (sum(v_i))^2 - sum(v_i^2)
        sum_embeddings = torch.sum(vid_w, dim=1)
        sum_square = sum_embeddings * sum_embeddings
        square_sum = torch.sum(vid_w * vid_w, dim=1)
        
        return 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

class DeepFM(nn.Module):
    def __init__(self, feature_max_idx, sparse_cols, dense_cols, embedding_dim=16, hidden_dims=[256, 128]):
        super(DeepFM, self).__init__()
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        num_sparse = len(sparse_cols)
        
        # 1. FM 一阶部分 (Linear)
        self.linear_sparse = nn.ModuleList([
            nn.Embedding(feature_max_idx[feat], 1) for feat in sparse_cols
        ])
        self.linear_dense = nn.Linear(len(dense_cols), 1)

        # 2. FM 二阶部分 (Embedding + SENet + Bilinear)
        self.embedding_sparse = nn.ModuleList([
            nn.Embedding(feature_max_idx[feat], embedding_dim) for feat in sparse_cols
        ])
        
        # 改进 1: 引入 SENet 动态调整特征权重 (针对缺失值和噪声)
        self.senet = SENet(num_sparse)
        
        # 改进 2: 引入双线性交互增强交叉表达
        self.bilinear = BilinearInteraction(num_sparse, embedding_dim)

        # 3. Deep 部分 (MLP + BN)
        # 改进 3: 数值特征投影层，使其与 Embedding 空间对齐
        self.dense_projection = nn.Linear(len(dense_cols), len(dense_cols))
        
        input_dim = num_sparse * embedding_dim + len(dense_cols)
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)

    def forward(self, x_sparse, x_dense):
        # --- FM 1st order ---
        linear_out = sum([self.linear_sparse[i](x_sparse[:, i]) for i in range(len(self.sparse_cols))])
        linear_out += self.linear_dense(x_dense)

        # --- FM 2nd order (Enhanced) ---
        embeddings = [self.embedding_sparse[i](x_sparse[:, i]).unsqueeze(1) for i in range(len(self.sparse_cols))]
        combined_embeddings = torch.cat(embeddings, dim=1) # [batch_size, num_sparse, embed_dim]
        
        # 应用 SENet 重新加权
        senet_embeddings = self.senet(combined_embeddings)
        
        # 应用双线性交互
        fm_second_order = self.bilinear(senet_embeddings)

        # --- Deep Part ---
        # 投影稠密特征
        projected_dense = F.relu(self.dense_projection(x_dense))
        
        deep_input = torch.cat([combined_embeddings.view(x_sparse.size(0), -1), projected_dense], dim=1)
        deep_out = self.deep_mlp(deep_input)

        # Final output
        logit = linear_out + fm_second_order + deep_out + self.bias
        return torch.sigmoid(logit).squeeze()

# --- Training Logic ---
def train_deepfm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DL ready data...")
    with open('dl_ready_data.pkl', 'rb') as f:
        dl_data = pickle.load(f)
    
    train_data = dl_data['train']
    val_data = dl_data['val']
    test_df = dl_data['test']
    sparse_cols = dl_data['sparse_features']
    dense_cols = dl_data['dense_features']
    feature_max_idx = dl_data['feature_max_idx']

    # 验证集评估准备 (用于冷启动分析)
    train_uids = set(train_data['uid'].unique())
    train_episodes = set(train_data['episode_id'].unique())
    is_cold_user = ~val_data['uid'].isin(train_uids)
    is_cold_episode = ~val_data['episode_id'].isin(train_episodes)

    train_dataset = RecDataset(train_data, sparse_cols, dense_cols, 'label')
    val_dataset = RecDataset(val_data, sparse_cols, dense_cols, 'label')
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    model = DeepFM(feature_max_idx, sparse_cols, dense_cols).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 引入学习率调度器 (移除已弃用的 verbose 参数)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    criterion = nn.BCELoss()

    best_auc = 0
    epochs = 10 # 增加训练轮数
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tk0 = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x_s, x_d, y in tk0:
            x_s, x_d, y = x_s.to(device), x_d.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x_s, x_d)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tk0.set_postfix(loss=total_loss/len(train_loader))

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_s, x_d, y in val_loader:
                x_s, x_d = x_s.to(device), x_d.to(device)
                y_pred = model(x_s, x_d)
                preds.extend(y_pred.cpu().numpy())
                targets.extend(y.numpy())
        
        auc = roc_auc_score(targets, preds)
        print(f"Epoch {epoch+1} Overall Val AUC: {auc:.4f}")
        
        # 增加冷启动评估维度
        if is_cold_user.any():
            cold_user_auc = roc_auc_score(np.array(targets)[is_cold_user], np.array(preds)[is_cold_user])
            print(f"Cold User AUC: {cold_user_auc:.4f}")
        
        if is_cold_episode.any():
            cold_episode_auc = roc_auc_score(np.array(targets)[is_cold_episode], np.array(preds)[is_cold_episode])
            print(f"Cold Episode AUC: {cold_episode_auc:.4f}")
        
        scheduler.step(auc) # 更新学习率
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_deepfm_enhanced.pth')
            print(f"New Best AUC found! Model saved.")

    # Inference
    print("Inference on test set...")
    model.load_state_dict(torch.load('best_deepfm_enhanced.pth'))
    model.eval()
    test_dataset = RecDataset(test_df, sparse_cols, dense_cols)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    test_preds = []
    with torch.no_grad():
        for x_s, x_d in tqdm(test_loader, desc="Predicting"):
            x_s, x_d = x_s.to(device), x_d.to(device)
            y_pred = model(x_s, x_d)
            test_preds.extend(y_pred.cpu().numpy())

    result = pd.DataFrame({'id': test_df['id'], 'label': test_preds})
    result.to_csv('result_deepfm_enhanced.csv', index=False)
    print("Enhanced results saved to result_deepfm_enhanced.csv")

if __name__ == '__main__':
    train_deepfm()


