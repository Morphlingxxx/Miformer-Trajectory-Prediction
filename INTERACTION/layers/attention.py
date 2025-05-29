import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


class Attention(nn.Module):
    def __init__(self,
                 hidden_dim: int) -> None:
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # 初始化 Query、Key 和 Value 的权重矩阵
        self.W_q_i = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 用于计算 Query 的线性层
        self.W_k_i = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 用于计算 Key 的线性层
        self.W_v_i = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 用于计算 Value 的线性层
        self.W_q_j = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 用于计算 Query 的线性层
        self.W_k_j = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 用于计算 Key 的线性层
        self.W_v_j = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 用于计算 Value 的线性层
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 归一化层
        
        self.apply(init_weights)

    def forward(self, embs_query, embs_key_value):
        # 计算 Query, Key, Value
        query_i = self.W_q_i(embs_query)  # (batch_size, sequence_length, hidden_dim)
        key_i = self.W_k_i(embs_key_value)    # (batch_size, sequence_length, hidden_dim)
        value_i = self.W_v_i(embs_key_value)  # (batch_size, sequence_length, hidden_dim)

        # 计算注意力分数，Q 和 K 点积之后除以 sqrt(hidden_dim)
        attention_scores = torch.bmm(query_i, key_i.transpose(1, 2))  # (batch_size, sequence_length, sequence_length)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        # 通过 softmax 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, sequence_length, sequence_length)

        # 使用注意力权重对 value 进行加权求和
        embs1_enhanced = torch.bmm(attention_weights, value_i)  # (batch_size, sequence_length, hidden_dim)

        # 残差连接和归一化
        embs_query = embs_query + embs_key_value + embs1_enhanced  # 残差连接
        embs_query = self.layer_norm(embs_query)  # 归一化

        embs = embs_query 

        return embs