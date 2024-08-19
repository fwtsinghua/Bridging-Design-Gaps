import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the CSV file
df = pd.read_csv('historical_data.csv')
# 去掉 'UID' 列
df.drop(columns=['UID'], inplace=True)


# Define the assembly graph (example)
assembly_graph = nx.Graph()
assembly_graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])  # Example edges source -> target


# edges = [('Arriving Time', 'Normal Clinic Service Time'),
#          ('Arriving Time', 'Fever Clinic Service Time'),
#          ('Normal Clinic Service Time', 'Pharmacy Service Time'),
#          ('Fever Clinic Service Time', 'Pharmacy Service Time'),
#          ('Pharmacy Service Time', 'Exit Time'),
#          ('Exit Time', 'Total Time'), ]
class ParametricGraphConstructor:
    def __init__(self, df, assembly_graph):
        self.df = df
        self.assembly_graph = assembly_graph

    def build_graph(self, row):
        # Create a graph using NetworkX
        G = nx.Graph()

        # Add nodes and edges based on the DataFrame
        for i, col_name in enumerate(self.df.columns):
            if np.isnan(row[col_name]):
                G.add_node(i, feature=0)
            else:
                G.add_node(i, feature=row[col_name])

        # Add edges based on the assembly graph
        for u, v in self.assembly_graph.edges:  # 两个节点的占位符 ： u 是边的起始节点，v 是边的结束节点
            G.add_edge(u, v)

        # Convert to PyTorch Geometric Data object
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        x = torch.tensor([G.nodes[i]['feature'] for i in range(len(G))], dtype=torch.float)

        return Data(x=x, edge_index=edge_index)


# Initialize the constructor and build the graphs
constructor = ParametricGraphConstructor(df, assembly_graph)
graphs = [constructor.build_graph(row) for _, row in df.iterrows()]


class FeaturePositionalEncoder(torch.nn.Module):
    def __init__(self, num_features, feature_embedding_dim, pos_embedding_dim):
        super().__init__()
        self.feature_tokenizer = Linear(num_features, feature_embedding_dim)
        self.positional_encoder = Linear(num_features, pos_embedding_dim)

    def forward(self, x):
        feature_embedding = self.feature_tokenizer(x)
        positional_embedding = self.positional_encoder(x)
        return feature_embedding, positional_embedding


class ParametricGCNEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # 在构造Data对象时，使用unsqueeze(-1)来确保x张量的形状为(num_nodes, 1)。
        x = self.conv1(x, edge_index)  # GCNConv层期望输入张量x的形状为(num_nodes, num_features)，而edge_index的形状为(2, num_edges)。
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        """
        - `x` 是一个张量，表示节点特征矩阵，形状为 `(N, F)`，其中 `N` 是所有图中节点的总数，`F` 是节点特征的维度。
        - `batch` 是一个张量，表示每个节点所属的图的索引，形状为 `(N,)`。它的每个值都标识了节点属于哪个图。
        - `size` 是一个可选参数，表示图的总数 `B`。如果提供了，它帮助函数知道有多少个图。如果未提供，函数会自动计算。
        """
        x = global_mean_pool(x, data.batch)  # data.batch
        return x
        # 函数的目标是将每个图的节点特征通过平均池化操作压缩成一个单一的图级特征向量。
        # 这种操作在处理批量图数据时非常有用，可以帮助将图数据从节点级别的特征转化为图级别的特征。


class CrossAttentionModule(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        """
        (seq_len, batch, embed_dim)
        :param query:
        :param key:
        :param value:
        :return:
        """
        output, _ = self.cross_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
        return output.squeeze(0)


class ParametricDataset(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


dataset = ParametricDataset(graphs)
dataloader = DataLoader(dataset, batch_size=1)

# Define models
# 将GCNConv的第一层的输入通道数设置为1，因为每个节点只有一个特征值。
encoder = ParametricGCNEncoder(num_features= 1, hidden_channels=64, out_channels=64) # 假设每个节点的特征维度为 1
feature_pos_encoder = FeaturePositionalEncoder(num_features=len(df.columns), feature_embedding_dim=64,
                                               pos_embedding_dim=64)
cross_attention = CrossAttentionModule(embed_dim=64, num_heads=4)
feature_pos_encoder.to('cuda')
encoder.to('cuda')
cross_attention.to('cuda')

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(feature_pos_encoder.parameters()) + list(cross_attention.parameters()), lr=0.01)


def train():
    encoder.train()
    feature_pos_encoder.train()
    cross_attention.train()

    for data in dataloader:
        """ 
        data 为 DataBatch(x=[6], edge_index=[2, 6], batch=[6], ptr=[2]) 
        在 `pytorch-geometric` 中，`DataBatch` 对象包含图数据的多个部分，下面是你提供的 `DataBatch` 的各个变量及其含义：
        - `x=[6]`: 这是节点特征矩阵。这里 `x` 是一个形状为 `[6]` 的张量，表示图中有 6 个节点，每个节点具有一个特征（特征维度为 1）。如果节点特征有多个维度，这个维度会更高，比如 `[6, F]`，其中 `F` 是特征的维度。
        - `edge_index=[2, 6]`: 这是边索引矩阵。`edge_index` 是一个形状为 `[2, 6]` 的张量，其中每一列表示图中一条边的源节点和目标节点。`2` 表示边的两个端点（源节点和目标节点），`6` 表示图中总共有 6 条边。因此，`edge_index` 的每一列包含一个边的两个端点的信息。
        - `batch=[6]`: 这是批处理索引张量。如果你的数据包含多个图并被批量处理，`batch` 用于标识每个节点属于哪个图。在这个例子中，`batch=[6]` 表示图中每个节点都属于同一个图（或者说只处理了一个图），因为每个节点的批次索引都是相同的。`batch` 的长度等于节点的数量。
        - `ptr=[2]`: 这是图的指针索引张量，用于在批量数据中快速定位每个图的起始位置。在这个例子中，`ptr=[2]` 表示只有一个图，所以它的指针索引仅包含两个值，这两个值标识了图的开始和结束位置。如果有多个图，`ptr` 的长度为图的数量加一，表示每个图在批处理数据中的起始位置。
        这些变量在处理图数据时非常重要，特别是当你需要对批处理的图进行操作时。它们帮助你管理图的结构和节点特征，并确保你可以在批处理操作中正确地访问和更新图数据。
        """
        optimizer.zero_grad()

        graph_embedding = encoder(data.to('cuda'))  # [1, 64]
        feature_embedding, positional_embedding = feature_pos_encoder(data.x.to('cuda'))
        feature_embedding = feature_embedding.unsqueeze(0)  # feature_embedding: [1, 64],
        positional_embedding = positional_embedding.unsqueeze(0)  # positional_embedding: [1, 64]

        # # Combine embeddings
        combined_embedding = torch.cat([graph_embedding, feature_embedding, positional_embedding], dim=0)

        # Cross attention
        multimodal_embedding = cross_attention(graph_embedding, feature_embedding, positional_embedding)

        # Loss calculation (example loss function)
        loss = torch.norm(multimodal_embedding - combined_embedding)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")


# Train the model
for epoch in range(1, 210):
    train()
