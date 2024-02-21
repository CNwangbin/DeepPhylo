import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from deepphylo.pre_dataset import set_seed, hac
from deepphylo.plot import plot_ss_curve, plot_pr_curve,plot_training
from deepphylo.evaluate import compute_metrics, select_best_epoch
from deepphylo.model import MLP_twin



def train(X_train, Y_train, X_eval, Y_eval):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = X_train.shape[1]
    model = MLP_twin(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
    metrics_dict = {'acc': [], 'mcc': [], 'roc_auc': [], 'aupr': [], 'precision':[], 'recall':[],'specificity':[], 'sensitivity':[]}
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred_train = model(batch_X)
            loss_train = criterion(y_pred_train[:,1].unsqueeze(-1), batch_y)
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
                loss_val = criterion(y_pred_val[:,1].unsqueeze(-1), batch_y)
                val_loss += loss_val.item() * batch_X.size(0)
                val_preds.append(y_pred_val[:,1].unsqueeze(-1).detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        # Calculate validation R2
        y_val = np.concatenate(y_val)
        val_preds = np.concatenate(val_preds)
        metrics = compute_metrics(y_val, val_preds)
        print(f"epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, acc: {metrics['acc']:.4f}, mcc:{metrics['mcc']:.4}, roc-auc:{metrics['roc_auc']:.4f}, aupr:{metrics['aupr']:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_dict['acc'].append(metrics['acc'])
        metrics_dict['mcc'].append(metrics['mcc'])
        metrics_dict['roc_auc'].append(metrics['roc_auc'])
        metrics_dict['aupr'].append(metrics['aupr'])
        metrics_dict['precision'].append(metrics['precision'])
        metrics_dict['recall'].append(metrics['recall'])
        metrics_dict['specificity'].append(metrics['specificity'])
        metrics_dict['sensitivity'].append(metrics['sensitivity'])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    return train_losses, val_losses, metrics_dict


if __name__ == '__main__':
    """
        # python3 src/MDeep.py --train --data_dir data/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 
        # --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2

    """
    #X_train = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/X_train.npy')
    #X_eval = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/X_eval.npy')
    #Y_train = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/Y_train.npy')
    #Y_eval = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/Y_eval.npy')
    #C = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/c.npy')
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
                        default=1e-4,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('-bs',
                        '--batchsize',
                        default=32,
                        type=int,
                        help='Batchsize size when encoding protein embedding with backbone')
    args = parser.parse_args()
    X_train = np.load('data_DeepPhylo/twin/X_train.npy')
    X_eval = np.load('data_DeepPhylo/twin/X_eval.npy')
    Y_train = np.load('data_DeepPhylo/twin/Y_train.npy')
    Y_eval = np.load('data_DeepPhylo/twin/Y_eval.npy')
    C = np.load('data_DeepPhylo/twin/c.npy')
    hac_index = hac(C)
    X_train = X_train[:,hac_index]
    X_eval = X_eval[:,hac_index]
    train_losses, val_losses, metrics_dict = train(X_train, Y_train, X_eval, Y_eval)
    plot_training(train_losses, val_losses, metrics_dict)
    best_epoch = select_best_epoch(metrics_dict, ['acc', 'mcc', 'roc_auc', 'aupr'])
    recall, precision = plot_pr_curve(metrics_dict['precision'][best_epoch], metrics_dict['recall'][best_epoch])
    specificity, sensitivity = plot_ss_curve(metrics_dict['sensitivity'][best_epoch], metrics_dict['specificity'][best_epoch])
    print(f"Best epoch: {best_epoch+1}")
    print(f"Best metrics: sens: {sensitivity:.4f}, spec:{specificity:.4f}, acc: {metrics_dict['acc'][best_epoch]:.4f}, p:{precision:.4f},mcc: {metrics_dict['mcc'][best_epoch]:.4f}, f1:{2*precision*recall / (precision+recall):.4f},roc_auc: {metrics_dict['roc_auc'][best_epoch]:.4f}, aupr: {metrics_dict['aupr'][best_epoch]:.4f}")