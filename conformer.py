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

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=3, padding=1):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(1, 2, 0)  # (seq_len, batch_size, d_model) -> (batch_size, d_model, seq_len)
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # (batch_size, d_model, seq_len) -> (seq_len, batch_size, d_model)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ConformerAutoEncoder(nn.Module):
    def __init__(self, input_size=130, d_model=64, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256):
        super(ConformerAutoEncoder, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器和解码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Convolutional模块
        self.conv_module = ConvModule(d_model)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, input_size)

    def forward(self, src):
        # 输入形状: (batch_size, seq_len, input_size)
        # Transformer期望的输入形状: (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        
        # 输入投影
        src = self.input_projection(src)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        memory = self.transformer_encoder(src)
        
        # Convolutional模块
        memory = self.conv_module(memory)
        
        # Transformer解码
        # 在自编码任务中，目标序列与输入序列相同
        output = self.transformer_decoder(memory, memory)
        
        # 输出投影
        output = self.output_projection(output)
        
        # 将输出转回原始形状: (batch_size, seq_len, input_size)
        output = output.permute(1, 0, 2)
        
        return output

