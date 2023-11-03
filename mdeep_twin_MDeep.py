import numpy as np
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from scipy.cluster.hierarchy import dendrogram, linkage
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, precision_recall_curve, auc, roc_curve
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

def compute_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred_hard = np.where(y_pred > 0.5, 1, 0)
    acc = accuracy_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    fpr, tpr, t = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # 从FPR和TPR计算特异度和敏感度
    specificity = 1 - fpr
    sensitivity = tpr
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    metric_dict = {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'precision':precision,'recall':recall, 'specificity':specificity, 'sensitivity':sensitivity}
    return metric_dict

def plot_training(train_losses, val_losses, metrics, title='The twin prediction result: train and test Loss/metrics'):
    def plot_metric(name, metric_values):
        max_metric = max(metric_values)
        plt.plot(metric_values, label=f"{name}")
        # show the digits of max validation aupr in the figure
        plt.plot(metric_values.index(max_metric), max_metric, 'ro')
        # show the digits of max validation aupr in the figure
        plt.annotate(f'{max_metric:.4f}', xy=(metric_values.index(max_metric), max_metric), xytext=(metric_values.index(max_metric), max_metric))
      
    # plot the training and validation loss
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plot_metric('acc', metrics['acc'])
    plot_metric('mcc', metrics['mcc'])
    plot_metric('roc_auc', metrics['roc_auc'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/AUC')
    plt.legend(frameon=False)
    plt.show()

def plot_pr_curve(precision, recall, title='Precision-Recall Curve'):
    # 绘制 Precision-Recall 曲线
    plt.figure()
    plt.step(recall, precision, color='black', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # 绘制y=x参考线
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # 计算最接近 y=x 的点
    diff = np.abs(np.array(precision) - np.array(recall))
    min_diff_idx = np.argmin(diff)
    intersect_x = recall[min_diff_idx]
    intersect_y = precision[min_diff_idx]
    
    # 绘制最接近 y=x 的点
    plt.scatter(intersect_x, intersect_y, color='r')
    plt.text(intersect_x, intersect_y, f'({intersect_x:.4f}, {intersect_y:.4f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title(title)
    plt.legend()
    plt.show()
    return intersect_x, intersect_y


def plot_ss_curve(sensitivity, specificity, title='Sensitivity-Specificity Curve'):
    # 绘制 Sensitivity-Specificity 曲线
    plt.figure()
    plt.step(specificity, sensitivity, color='black', alpha=0.2, where='post')
    plt.fill_between(specificity, sensitivity, step='post', alpha=0.2, color='lightskyblue')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # 绘制y=x参考线, 以及与曲线相交的点
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # 计算最接近 y=x 的点
    diff = np.abs(np.array(sensitivity) - np.array(specificity))
    min_diff_idx = np.argmin(diff)
    intersect_x = specificity[min_diff_idx]
    intersect_y = sensitivity[min_diff_idx]
    
    # 绘制最接近 y=x 的点
    plt.scatter(intersect_x, intersect_y, color='r', label='Intersection')
    plt.text(intersect_x, intersect_y, f'({intersect_x:.4f}, {intersect_y:.4f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title(title)
    plt.show()
    return intersect_x, intersect_y


def conv_len_cal(L_in, kernel_size, stride, padding=0):
    return math.floor((L_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

class MDeep(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, 128, 64),
            nn.BatchNorm1d(32), # 文章似乎是最后norm, 不过正常来说都是先norm再激活
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, 4, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        out_length = conv_len_cal(conv_len_cal(input_size, 128, 64), 4, 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(out_length * 32, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_conv = self.conv_layers(x.unsqueeze(1))
        out = self.fc_layers(out_conv.view(out_conv.size(0), -1))
        return out


def train(X_train, Y_train, X_eval, Y_eval):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = X_train.shape[1]
    model = MDeep(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
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

def select_best_epoch(metrics_dict, metrics_order):
    filtered_epochs = None
    for metric in metrics_order:
        if not metrics_dict[metric]:
            continue
        if len(metrics_dict[metric]) == 1:
            filtered_epochs = metrics_dict[metric][0]
            break
        if filtered_epochs is not None:
            filtered_metric = [(epoch, val) for epoch, val in enumerate(metrics_dict[metric]) if epoch in filtered_epochs]
            max_val = max(filtered_metric, key=lambda x: x[1])
            filtered_epochs = [epoch for epoch, val in filtered_metric if val == max_val[1]]
            if len(filtered_epochs) == 1:
                best_epoch = filtered_epochs[0]
                break
        else:
            max_val = max(enumerate(metrics_dict[metric]), key=lambda x: x[1])
            filtered_epochs = [epoch for epoch, val in enumerate(metrics_dict[metric]) if val == max_val[1]]
            if len(filtered_epochs) == 1:
                best_epoch = filtered_epochs[0]
                break
    return best_epoch

if __name__ == '__main__':
    """
        # python3 src/MDeep.py --train --data_dir data/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 
        # --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2

    """
    X_train = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/X_train.npy')
    X_eval = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/X_eval.npy')
    Y_train = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/Y_train.npy')
    Y_eval = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/Y_eval.npy')
    C = np.load('/home/wangbin/DeepPhy/data/data_mdeep/twin/c.npy')
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

