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

# --- Dataset Definition (Reuse from DeepFM logic) ---
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

# --- DCN-V2 Components ---

class CrossNetworkV2(nn.Module):
    """
    DCN-V2 Cross Network: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    Uses Matrix Decomposition for efficiency.
    """
    def __init__(self, input_dim, num_layers):
        super(CrossNetworkV2, self).__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            # x_l+1 = x_0 * W*xl + xl
            xw = self.cross_weights[i](xl)
            xl = x0 * xw + xl
        return xl

class DCNv2(nn.Module):
    def __init__(self, feature_max_idx, sparse_cols, dense_cols, embedding_dim=16, cross_layers=3, hidden_dims=[256, 128]):
        super(DCNv2, self).__init__()
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        
        # 1. Embedding Layer
        self.embedding_sparse = nn.ModuleList([
            nn.Embedding(feature_max_idx[feat], embedding_dim) for feat in sparse_cols
        ])

        input_dim = len(sparse_cols) * embedding_dim + len(dense_cols)

        # 2. Cross Network
        self.cross_net = CrossNetworkV2(input_dim, cross_layers)

        # 3. Deep Network (MLP)
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            curr_dim = h_dim
        self.deep_mlp = nn.Sequential(*layers)

        # 4. Final Combination
        # Combines outputs of Cross and Deep parts
        self.final_linear = nn.Linear(input_dim + curr_dim, 1)
        
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
        # Embeddings
        embeddings = [self.embedding_sparse[i](x_sparse[:, i]) for i in range(len(self.sparse_cols))]
        combined_embeddings = torch.cat(embeddings, dim=1)
        
        # Concat with dense features
        x0 = torch.cat([combined_embeddings, x_dense], dim=1)

        # Cross Part
        cross_out = self.cross_net(x0)

        # Deep Part
        deep_out = self.deep_mlp(x0)

        # Concat Cross and Deep outputs
        stack_out = torch.cat([cross_out, deep_out], dim=1)
        
        logit = self.final_linear(stack_out)
        return torch.sigmoid(logit).squeeze()

# --- Training Logic ---

def train_dcn_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DL ready data for DCN-V2...")
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

    # Initialize DCN-V2
    model = DCNv2(feature_max_idx, sparse_cols, dense_cols).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    criterion = nn.BCELoss()

    best_auc = 0
    epochs = 10
    
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
            torch.save(model.state_dict(), 'best_dcn_v2.pth')
            print(f"New Best AUC found for DCN-V2! Model saved.")

    # Inference
    print("Inference on test set using DCN-V2...")
    model.load_state_dict(torch.load('best_dcn_v2.pth'))
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
    result.to_csv('result_dcn_v2.csv', index=False)
    print("DCN-V2 results saved to result_dcn_v2.csv")

if __name__ == '__main__':
    train_dcn_v2()
