import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import r2_score
# import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import random

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


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            # nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 8),
            # nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)


def train(X_train, Y_train, X_eval, Y_eval):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = X_train.shape[1]
    model = MLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3, weight_decay=0.05)
    # Convert data to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(Y_train).float().view(-1, 1)
    X_val_tensor = torch.from_numpy(X_eval).float()
    y_val_tensor = torch.from_numpy(Y_eval).float().view(-1, 1)
    # Create DataLoader for training and validation data
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred_train = model(batch_X)
            loss_train = criterion(y_pred_train, batch_y)
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        y_val = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                y_val.append(batch_y.numpy())
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                y_pred_val = model(batch_X)
                loss_val = criterion(y_pred_val, batch_y)
                val_loss += loss_val.item() * batch_X.size(0)
                val_preds.append(y_pred_val.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        # Calculate validation R2
        y_val = np.concatenate(y_val)
        val_preds = np.concatenate(val_preds)
        print(y_val.shape, val_preds.shape)
        val_r2 = r2_score(y_val, val_preds)
        print(val_r2)
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
    train_losses, val_losses, val_r2s = train(X_train, Y_train, X_eval, Y_eval)
    plot_age(train_losses, val_losses, val_r2s, title='The MLP model prediction on age dataset: train and test Loss/R2')
