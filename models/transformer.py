import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ManualMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(ManualMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projection and split into heads
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # 3. Concatenate heads
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = ManualMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Sublayer 1: Attention + Add & Norm
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Sublayer 2: Feed Forward + Add & Norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class TransformerClassifier(nn.Module):
    """
    Complete Transformer Encoder for Classification.
    Uses [CLS] token as requested.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_classes, n_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        # Use only [CLS] token output for classification
        cls_out = x[:, 0, :]
        return self.fc(cls_out)