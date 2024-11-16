import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_size=130, d_model=64, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256):
        super(TransformerAutoEncoder, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # 輸入投影層
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer編碼器和解碼器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 輸出投影層
        self.output_projection = nn.Linear(d_model, input_size)

    def forward(self, src):
        # 輸入形狀: (batch_size, seq_len, input_size)
        # Transformer期望的輸入形狀: (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        
        # 輸入投影
        src = self.input_projection(src)
        
        # 添加位置編碼
        src = self.pos_encoder(src)
        
        # Transformer編碼
        memory = self.transformer_encoder(src)
        
        # Transformer解碼
        # 在自編碼任務中,目標序列與輸入序列相同
        output = self.transformer_decoder(memory, memory)
        
        # 輸出投影
        output = self.output_projection(output)
        
        # 將輸出轉回原始形狀: (batch_size, seq_len, input_size)
        output = output.permute(1, 0, 2)
        
        return output

# 使用示例
# model = TransformerAutoEncoder(input_size=130)
# x = torch.randn(32, 100, 130)  # (batch_size, seq_len, input_size)
# output = model(x)
# print(output.shape)  # 應該輸出 torch.Size([32, 100, 130])