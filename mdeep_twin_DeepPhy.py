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
    def __init__(self, hidden_size, embeddings, kernel_size=7):
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
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
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
    criterion = nn.MSELoss()
    batch_size = 32
    # Create DataLoader for training and validation data
    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.custom_collate_fn)
    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.custom_collate_fn)
    model = DeepPhy(hidden_size, train_dataset.embeddings, kernal_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Training
    epochs = 200
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
    D = inverse_C(C)
    phy_embedding = reducer(C, 'pca', 64, whiten=True)

    train_losses, val_losses, metrics_dict = train(X_train, Y_train, X_eval, Y_eval, phy_embedding)
    plot_training(train_losses, val_losses, metrics_dict)
    best_epoch = select_best_epoch(metrics_dict, ['acc', 'mcc', 'roc_auc', 'aupr'])
    recall, precision = plot_pr_curve(metrics_dict['precision'][best_epoch], metrics_dict['recall'][best_epoch])
    specificity, sensitivity = plot_ss_curve(metrics_dict['sensitivity'][best_epoch], metrics_dict['specificity'][best_epoch])
    print(f"Best epoch: {best_epoch+1}")
    print(f"Best metrics: sens: {sensitivity:.4f}, spec:{specificity:.4f}, acc: {metrics_dict['acc'][best_epoch]:.4f}, p:{precision:.4f},mcc: {metrics_dict['mcc'][best_epoch]:.4f}, f1:{2*precision*recall / (precision+recall):.4f},roc_auc: {metrics_dict['roc_auc'][best_epoch]:.4f}, aupr: {metrics_dict['aupr'][best_epoch]:.4f}")

