import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from scipy.cluster.hierarchy import dendrogram, linkage


# 设置随机种子
def set_seed(seed_value=42):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # 设置NumPy的随机种子
    np.random.seed(seed_value)

    # 设置Python的随机种子
    random.seed(seed_value)

    # 使结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(1234)


def hac(cor):
    def mydist(p1, p2):
        x = int(p1)
        y = int(p2)
        return 1.0 - cor[x, y]
    x = list(range(cor.shape[0]))
    X = np.array(x)
    linked = linkage(np.reshape(X, (len(X), 1)), metric=mydist, method='single')
    result = dendrogram(linked,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    indexes = result.get('ivl')
    del result
    del linked
    index =[]
    for _, itm in enumerate(indexes):
        index.append(int(itm))

    return index


def inverse_C(C):
    return np.log(C) / -4


def reducer(data_matrix, method='pca', n_components=2, whiten=False):
    # if method == 'umap':
    #     reducer = umap.UMAP(n_components=n_components)
    if method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError('Unrecognized method: %s' % method)

    return reducer.fit_transform(data_matrix)



def plot_age(train_losses, val_losses, val_r2s, title='The simple MLP baseline age prediction result: train and test Loss/R2'):
    # Plot the training and validation
    max_r2 = max(val_r2s)
    # plot the training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.plot(val_r2s, label='Validation R2')
    # show the digits of max validation aupr in the figure
    plt.plot(val_r2s.index(max_r2), max_r2, 'ro')
    # show the digits of max validation aupr in the figure
    plt.annotate(f'{max_r2:.4f}', xy=(val_r2s.index(max_r2), max_r2), xytext=(val_r2s.index(max_r2), max_r2))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/R2')
    plt.legend(frameon=False)
    plt.show()

def conv_len_cal(L_in, kernel_size, stride, padding=0):
    return math.floor((L_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

class DeepPhyDataset(Dataset):
    def __init__(self, phy_embedding, data:np.array, label:np.array):
        self.embeddings = torch.cat([torch.zeros(1, phy_embedding.shape[1]), torch.from_numpy(phy_embedding)], dim=0)
        self.X_tensor = torch.from_numpy(data).float()
        self.y_tensor = torch.from_numpy(label).float()

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        nonzero_indices = torch.nonzero(self.X_tensor[idx])
        sample = {
            'X': self.X_tensor[idx],
            'y': self.y_tensor[idx],
            'nonzero_indices': nonzero_indices.squeeze(dim=1) + 1
            }
        return sample
    
    @ staticmethod
    def custom_collate_fn(batch):
        X_batch = [sample['X'] for sample in batch]
        y_batch = [sample['y'] for sample in batch]
        nonzero_indices_batch = [sample['nonzero_indices'] for sample in batch]

        # 找到最大的序列长度 # max(len(indices) for indices in nonzero_indices_batch)
        max_seq_len = max([len(indices) for indices in nonzero_indices_batch])
        # 填充非零索引，使它们具有相同的长度
        padded_indices_batch = []
        for indices in nonzero_indices_batch:
            padded_indices = torch.nn.functional.pad(indices, (0, max_seq_len - len(indices)))
            padded_indices_batch.append(padded_indices)

        # 将序列堆叠成一个批张量
        stacked_X_batch = torch.stack(X_batch)
        stacked_y_batch = torch.stack(y_batch)
        stacked_padded_indices_batch = torch.stack(padded_indices_batch)
        
        assert stacked_X_batch.shape[0] == stacked_y_batch.shape[0] == stacked_padded_indices_batch.shape[0]
        return {'X': stacked_X_batch, 'y': stacked_y_batch, 'nonzero_indices': stacked_padded_indices_batch}

class DeepPhy(nn.Module):
    def __init__(self, hidden_size, embeddings, kernel_size=7,num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.input_size = self.embedding.weight.size(0) - 1
        self.embedding_dim = self.embedding.weight.size(1)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.fc_abundance = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.Tanh(),
        )
        self.fc_phy = nn.Linear(self.embedding_dim, hidden_size)
        self.fc_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//8),
            nn.Tanh(),
            nn.Linear(hidden_size//8, 1)
        )


    def forward(self, x, ids):
        x_abundance = self.fc_abundance(x)
        embeddings_tensor = self.embedding(ids)
        embeddings_tensor = self.fc_phy(embeddings_tensor)
        embeddings_tensor = embeddings_tensor.transpose(1, 2)
        x_conv = self.conv(embeddings_tensor)
        # x_phy2vec = x_phy2vec.mean(dim=2, keepdim=False)
        x_conv = x_conv.max(dim=2, keepdim=False)[0]
        x = x_abundance * x_conv
        return self.fc_pred(x)


def train(X_train, Y_train, X_eval, Y_eval, phy_embedding):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 64
    kernal_size = 7
    num_layers = 1
    criterion = nn.MSELoss()
    batch_size = 16
    # Create DataLoader for training and validation data
    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.custom_collate_fn)
    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.custom_collate_fn)
    model = DeepPhy(hidden_size, train_dataset.embeddings, kernal_size, num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Training
    epochs = 500
    patience = 5
    best_val_loss = float("inf")
    counter = 0
    train_losses = []
    val_losses = []
    val_r2s = []
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices'])
            loss_train = criterion(y_pred_train, batch['y'])
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item() * batch['X'].size(0)

        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        y_val = []
        with torch.no_grad():
            for batch in val_loader:
                y_val.append(batch['y'].numpy())
                batch = {key: val.to(device) for key, val in batch.items()}
                y_pred_val =model(batch['X'], batch['nonzero_indices'])
                loss_val = criterion(y_pred_val, batch['y'])
                val_loss += loss_val.item() * batch['X'].size(0)
                val_preds.append(y_pred_val.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        # Calculate validation R2
        y_val = np.concatenate(y_val)
        val_preds = np.concatenate(val_preds)
        val_r2 = r2_score(y_val, val_preds)
        print(f'epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_r2: {val_r2:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    return train_losses, val_losses, val_r2s



if __name__ == '__main__':
    X_train = np.load('/home/wangbin/DeepPhy/data/data_mdeep/usa/X_train.npy')
    X_eval = np.load('/home/wangbin/DeepPhy/data/data_mdeep/usa/X_eval.npy')
    Y_train = np.load('/home/wangbin/DeepPhy/data/data_mdeep/usa/Y_train.npy')
    Y_eval = np.load('/home/wangbin/DeepPhy/data/data_mdeep/usa/Y_eval.npy')
    C = np.load('/home/wangbin/DeepPhy/data/data_mdeep/usa/c.npy')
    D = inverse_C(C)
    phy_embedding = reducer(C, 'pca', 64, whiten=True)

    # hac_index = hac(D)
    # phy_embedding = phy_embedding[hac_index,:]
    # X_train = X_train[:,hac_index]
    # X_eval = X_eval[:,hac_index]

    train_losses, val_losses, val_r2s = train(X_train, Y_train, X_eval, Y_eval, phy_embedding)
    plot_age(train_losses, val_losses, val_r2s, title='The DeepPhy model prediction on age dataset: train and test Loss/R2')


    # 绘制长度分布图
    # len_train = [len(np.nonzero(X_train[idx])[0]) for idx in range(len(X_train))]
    # plt.bar([i for i in range(len(len_train))], sorted(len_train))