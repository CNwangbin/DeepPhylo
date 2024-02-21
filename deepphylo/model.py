import math
import torch.nn as nn

class DeepPhy_twin(nn.Module):
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

class DeepPhy_usa(nn.Module):
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

def conv_len_cal(L_in, kernel_size, stride, padding=0):
    return math.floor((L_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
class MDeep_twin(nn.Module):
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

class MDeep_usa(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 8, 4),
            nn.BatchNorm1d(64), # 文章似乎是最后norm
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 8, 4),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 8, 4),
            nn.BatchNorm1d(64),
            nn.Tanh(),
        )

        out_length = conv_len_cal(conv_len_cal(conv_len_cal(input_size, 8, 4), 8, 4), 8, 4)
        self.fc_layers = nn.Sequential(
            nn.Linear(out_length * 64, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64, 8),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out_conv = self.conv_layers(x.unsqueeze(1))
        out = self.fc_layers(out_conv.view(out_conv.size(0), -1))
        return out
    
class MLP_twin(nn.Module):
    def __init__(self, input_size):
        super(MLP_twin, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class MLP_usa(nn.Module):
    def __init__(self, input_size):
        super(MLP_usa, self).__init__()
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