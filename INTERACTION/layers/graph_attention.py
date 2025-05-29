from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout
import torch_geometric
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
from utils import init_weights
from torch_geometric.nn import HypergraphConv
from torch.nn import functional as F 
class GraphAttention(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 has_edge_attr: bool,
                 if_self_attention: bool,
                 **kwargs) -> None:
        super(GraphAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)
        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr)
        x_dst = x_dst + self._ffn_layer(self.ffn_prenorm(x_dst))
        return x_dst

    def message(self,
                x_dst_i: torch.Tensor,
                x_src_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim)
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)
        scale = self.head_dim ** 0.5
        weight = (query_i * key_j).sum(dim=-1) / scale
        weight = softmax(weight, index, ptr)
        weight = self.attn_drop(weight)
        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads*self.head_dim)

    def _mha_layer(self,
                   x_src: torch.Tensor,
                   x_dst: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src)

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    
class DynamicHyperGraphAttention(nn.Module):
    def __init__(self, hidden_dim=128, k_nearest=10):
        super(DynamicHyperGraphAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.k_nearest = k_nearest  # 控制每个节点的超边数
        self.hyperedge_cache = {}  # 缓存超边关联矩阵

        # 超图卷积层
        self.hypergraph_conv1 = HypergraphConv(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.MHA = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

    def forward(self, embs1, embs2, m_embs, edge_index):
        x = torch.cat([embs1, embs2], dim=0)  # (num_nodes, hidden_dim)

        # 动态生成超边关联矩阵
        hyperedge_incidence_matrix = self.generate_dynamic_hyperedge_incidence_matrix(x)

        # 动态生成超边特征
        num_hyperedges = hyperedge_incidence_matrix.size(1)
        hyperedge_features = torch.randn(num_hyperedges, self.hidden_dim).to(x.device)

        # 计算超边的注意力权重
        hyperedge_features, _ = self.attention(hyperedge_features.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        hyperedge_features = hyperedge_features.squeeze(1)

        # 使用稀疏矩阵的超边关联矩阵
        hyperedge_interaction = torch.sparse.mm(hyperedge_incidence_matrix, hyperedge_features)

        # 将节点特征和交互特征相加
        x = x + hyperedge_interaction

        x = self.hypergraph_conv1(x, edge_index)
        x = F.elu(self.bn(x))

        # 使用自注意力机制将所有节点特征融合到 t_embs.size(0) 个输出中
        x = x.unsqueeze(1)
        query = m_embs.unsqueeze(1)
        attn_output, _ = self.MHA(query, x, x)
        m_embs = attn_output.squeeze(1)

        return m_embs

    def generate_dynamic_hyperedge_incidence_matrix(self, x):
        device = x.device  # 确保获取当前输入张量所在的设备
        num_nodes = x.size(0)
        
        # 计算距离矩阵，并选取k近邻
        with torch.no_grad():
            distance_matrix = torch.cdist(x, x, p=2)
            _, nearest_indices = torch.topk(distance_matrix, self.k_nearest, largest=False)
        
        # 创建行和列索引
        row_indices = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, self.k_nearest).flatten()
        col_indices = nearest_indices.flatten().to(device)  # 确保 nearest_indices 也在相同设备上
        
        # 将所有索引和相关张量都放置在同一个设备上
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.ones(indices.size(1), device=device)
        hyperedge_incidence_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)
        
        return hyperedge_incidence_matrix
