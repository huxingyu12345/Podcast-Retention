import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json

# --- Dataset Definition (Consistent with DeepFM/DCN-v2) ---
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

# --- AutoInt Components ---

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention layer for feature interaction.
    Each feature is treated as a 'token'.
    """
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_linear = nn.Linear(embed_dim, embed_dim) # For residual connection if needed

    def forward(self, x):
        # x shape: [batch_size, num_features, embed_dim]
        batch_size, num_features, embed_dim = x.size()
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch_size, num_heads, num_features, num_features]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Context vector: [batch_size, num_heads, num_features, head_dim]
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_features, embed_dim)
        
        # Final output with residual
        out = F.relu(self.fc(context) + x)
        return out

class AutoInt(nn.Module):
    def __init__(self, feature_max_idx, sparse_cols, dense_cols, embed_dim=16, num_heads=2, num_attn_layers=3, hidden_dims=[256, 128]):
        super(AutoInt, self).__init__()
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        self.num_features = len(sparse_cols) + len(dense_cols)
        
        # 1. Embedding Layer for Sparse Features
        self.embedding_sparse = nn.ModuleList([
            nn.Embedding(feature_max_idx[feat], embed_dim) for feat in sparse_cols
        ])
        
        # 2. Linear Projection for Dense Features (to align with embed_dim)
        self.linear_dense = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in dense_cols
        ])
        
        # 3. Interacting Layer (Transformer-like)
        self.attn_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads) for _ in range(num_attn_layers)
        ])
        
        # 4. MLP Layer
        input_dim_mlp = self.num_features * embed_dim
        layers = []
        curr_dim = input_dim_mlp
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            curr_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction
        self.final_linear = nn.Linear(curr_dim, 1)
        
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
        # Sparse embeddings: [batch_size, len(sparse_cols), embed_dim]
        sparse_embeds = [self.embedding_sparse[i](x_sparse[:, i]).unsqueeze(1) for i in range(len(self.sparse_cols))]
        
        # Dense linear projection: [batch_size, len(dense_cols), embed_dim]
        dense_embeds = [self.linear_dense[i](x_dense[:, i].unsqueeze(1)).unsqueeze(1) for i in range(len(self.dense_cols))]
        
        # Combined features as tokens: [batch_size, num_features, embed_dim]
        x = torch.cat(sparse_embeds + dense_embeds, dim=1)
        
        # Self-Attention interactions
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        
        # Flatten for MLP
        x_flatten = x.view(x.size(0), -1)
        
        # MLP and output
        mlp_out = self.mlp(x_flatten)
        logit = self.final_linear(mlp_out)
        
        return torch.sigmoid(logit).squeeze()

# --- Training Logic ---

def train_autoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DL ready data for AutoInt...")
    with open('dl_ready_data.pkl', 'rb') as f:
        dl_data = pickle.load(f)
    
    train_data = dl_data['train']
    val_data = dl_data['val']
    test_df = dl_data['test']
    sparse_cols = dl_data['sparse_features']
    dense_cols = dl_data['dense_features']
    feature_max_idx = dl_data['feature_max_idx']

    # Cold start assessment preparation
    train_uids = set(train_data['uid'].unique())
    train_episodes = set(train_data['episode_id'].unique())
    is_cold_user = ~val_data['uid'].isin(train_uids)
    is_cold_episode = ~val_data['episode_id'].isin(train_episodes)

    train_dataset = RecDataset(train_data, sparse_cols, dense_cols, 'label')
    val_dataset = RecDataset(val_data, sparse_cols, dense_cols, 'label')
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    # Initialize AutoInt
    # Using embed_dim=16, 2 heads, 3 attention layers
    model = AutoInt(feature_max_idx, sparse_cols, dense_cols).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    criterion = nn.BCELoss()

    best_auc = 0
    epochs = 20
    
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
        
        # Cold start evaluation
        if is_cold_user.any():
            cold_user_auc = roc_auc_score(np.array(targets)[is_cold_user], np.array(preds)[is_cold_user])
            print(f"Cold User AUC: {cold_user_auc:.4f}")
        
        if is_cold_episode.any():
            cold_episode_auc = roc_auc_score(np.array(targets)[is_cold_episode], np.array(preds)[is_cold_episode])
            print(f"Cold Episode AUC: {cold_episode_auc:.4f}")
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_autoint.pth')
            print(f"New Best AUC found for AutoInt! Model saved.")

    # Inference
    print("Inference on test set using AutoInt...")
    model.load_state_dict(torch.load('best_autoint.pth'))
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
    result.to_csv('result_autoint.csv', index=False)
    print("AutoInt results saved to result_autoint.csv")

if __name__ == '__main__':
    train_autoint()
