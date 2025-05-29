import torch
import torch.nn as nn
import torch.fft
from torch.nn import functional as F 

class TrajectoryModelWithFourierDecoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, cutoff_frequency, sampling_rate):
        super(TrajectoryModelWithFourierDecoupling, self).__init__()
        # 初始化傅里叶嵌入解耦模块
        self.fourier_decoupling = FourierEmbeddingDecoupling(cutoff_frequency=cutoff_frequency, sampling_rate=sampling_rate)
        
        # 高频和低频特征提取器
        self.high_freq_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.low_freq_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 频段融合模块（自适应权重）
        self.fusion_weights = nn.Linear(hidden_dim * 2, 2)  # 学习高频和低频的融合权重
        
        # 注意力融合模块
        self.attention_fusion = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, embedding):
        # 对嵌入进行傅里叶变换解耦
        low_freq, high_freq = self.fourier_decoupling(embedding)

        # 高频和低频特征提取
        high_freq_features = self.high_freq_extractor(high_freq)
        low_freq_features = self.low_freq_extractor(low_freq)

        # 自适应融合
        combined_features = torch.cat([high_freq_features, low_freq_features], dim=-1)  # [batch_size, seq_len, 2 * hidden_dim]
        weights = torch.sigmoid(self.fusion_weights(combined_features))  # 权重范围在 [0, 1] 之间
        fused_features = weights[:, :, 0].unsqueeze(-1) * low_freq_features + weights[:, :, 1].unsqueeze(-1) * high_freq_features

        # 使用注意力机制进一步融合特征
        attn_output, _ = self.attention_fusion(fused_features, fused_features, fused_features)
        # torch.cuda.empty_cache()

        # 融合后的最终特征
        return attn_output

class FourierEmbeddingDecoupling(nn.Module):
    def __init__(self, cutoff_frequency: float, sampling_rate: float):
        super(FourierEmbeddingDecoupling, self).__init__()
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate

    def forward(self, embedding):
        # 假设输入的嵌入维度为 [batch_size, sequence_length, embedding_dim]
        # 对时间轴（sequence_length）进行傅里叶变换
        embedding_fft = torch.fft.fft(embedding, dim=1)  # 频域转换，沿着序列长度的维度进行变换

        # 构建频率掩码，根据 cutoff_frequency 进行频率分离
        frequencies = torch.fft.fftfreq(embedding.size(1), d=1/self.sampling_rate)
        low_freq_mask = torch.abs(frequencies) <= self.cutoff_frequency
        high_freq_mask = torch.abs(frequencies) > self.cutoff_frequency

        # 分别获取低频和高频成分
        low_freq_embedding = embedding_fft * low_freq_mask.unsqueeze(-1).to(embedding.device)
        high_freq_embedding = embedding_fft * high_freq_mask.unsqueeze(-1).to(embedding.device)

        # 逆傅里叶变换，将频域转换回时域
        low_freq_embedding_time = torch.fft.ifft(low_freq_embedding, dim=1).real
        high_freq_embedding_time = torch.fft.ifft(high_freq_embedding, dim=1).real

        return low_freq_embedding_time, high_freq_embedding_time

class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5  # 缩放因子
        self.num_heads = num_heads

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query_seq, key_seq, value_seq, mask=None):
        # 线性变换以获得 query, key 和 value
        query = self.query_proj(query_seq)  # [batch_size, seq_len_q, hidden_dim]
        key = self.key_proj(key_seq)        # [batch_size, seq_len_k, hidden_dim]
        value = self.value_proj(value_seq)  # [batch_size, seq_len_k, hidden_dim]

        # 使用多头注意力机制计算注意力输出
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=mask)

        return attn_output
