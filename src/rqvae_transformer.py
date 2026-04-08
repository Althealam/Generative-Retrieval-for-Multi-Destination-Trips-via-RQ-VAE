import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        # 创建一个位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 将位置编码加到 Embedding 上
        return x + self.pe[:, :x.size(1), :]

class RQVAETransformer(nn.Module):
    def __init__(self, 
                 num_codes: int = 33, 
                 d_model: int = 256,  # Transformer 内部向量维度
                 nhead: int = 4,      # 多头注意力的头数
                 num_layers: int = 2, # Transformer 层的层数
                 dim_feedforward: int = 256, # 前馈网络的维度
                 max_len: int = 100):
        super().__init__()
        
        self.d_model = d_model
        # 1. Embedding 层 (和 GRU 版本一样)
        self.embedding = nn.Embedding(num_codes, d_model, padding_idx=32)
        
        # 2. 位置编码层 (Transformer 必需)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        # 3. Transformer Decoder 块 (这里我们用 EncoderLayer 来构建 Decoder-only 架构)
        # 注意：使用 batch_first=True 简化数据处理
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 4. 输出头 (和 GRU 版本一样，分别预测 Code1 和 Code2)
        self.fc_code1 = nn.Linear(d_model, 32)
        self.fc_code2 = nn.Linear(d_model, 32)

    def _generate_causal_mask(self, sz: int, device):
        # 生成上三角掩码，确保 $t$ 时刻看不到 $t+1$ 时刻
        # 格式为: True 表示被掩盖，False 表示可见
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, x):
        # x: [batch_size, seq_len] 这里的 seq_len 是 Code 的长度 (如 2*N)
        batch_size, seq_len = x.size()
        device = x.device
        
        # 1. 创建 Padding Mask (告诉 Attention 哪些是 Padding)
        # padding_mask shape: [batch_size, seq_len]
        padding_mask = (x == 32)
        
        # 2. 创建自回归因果掩码 (Causal Mask)
        # causal_mask shape: [seq_len, seq_len]
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # 3. Embedding + 位置编码
        # 注意：Embedding 后要乘 sqrt(d_model) 是 Transformer 的标准操作
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 4. 通过 Transformer 块
        # 这里必须同时传入因果掩码和 Padding 掩码
        # output shape: [batch_size, seq_len, d_model]
        output = self.transformer_block(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # 5. 提取序列的最后一个输出 (即预测下一站的基础)
        # 我们不再用 hn[-1]，而是直接取 seq_len 维度的最后一位
        last_hidden = output[:, -1, :] # shape: [batch_size, d_model]
        
        # 6. 输出 Code1 和 Code2 的预测 Logits
        out1 = self.fc_code1(last_hidden)
        out2 = self.fc_code2(last_hidden)
        
        return out1, out2