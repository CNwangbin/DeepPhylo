import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from deepphylo.pre_dataset import set_seed
from deepphylo.plot import plot_age
from deepphylo.model import MLP_usa


def train(X_train, Y_train, X_eval, Y_eval):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = X_train.shape[1]
    model = MLP_usa(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.05)
    # Convert data to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(Y_train).float().view(-1, 1)
    X_val_tensor = torch.from_numpy(X_eval).float()
    y_val_tensor = torch.from_numpy(Y_eval).float().view(-1, 1)
    # Create DataLoader for training and validation data
    batch_size = args.batchsize
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training
    epochs = args.epochs
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
    set_seed(1234)
    parser = argparse.ArgumentParser(
        description='Command line tool for twin classification')

    parser.add_argument('--epochs',
                        default=500,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-l',
                        '--lr',
                        default=5e-3,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('-bs',
                        '--batchsize',
                        default=32,
                        type=int,
                        help='Batchsize size when encoding protein embedding with backbone')
    args = parser.parse_args()
    X_train = np.load('data_DeepPhylo/usa/X_train.npy')
    X_eval = np.load('data_DeepPhylo/usa/X_eval.npy')
    Y_train = np.load('data_DeepPhylo/usa/Y_train.npy')
    Y_eval = np.load('data_DeepPhylop/usa/Y_eval.npy')
    train_losses, val_losses, val_r2s = train(X_train, Y_train, X_eval, Y_eval)
    plot_age(train_losses, val_losses, val_r2s, title='The MLP model prediction on age dataset: train and test Loss/R2')