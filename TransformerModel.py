
import torch.nn as nn
import torch
from einops import rearrange
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerTranslator(nn.Module):
    def __init__(self,in_token,out_token,ninp,nhead,nhid,nlayers,dropout=0.5):
        super(TransformerTranslator, self).__init__()
        self.ninp = ninp
        self.encoder_embedding = nn.Embedding(in_token,ninp)
        self.decoder_embedding = nn.Embedding(out_token,ninp)
        self.pos_encoder = PositionalEncoding(ninp,dropout)
        self.transformer = nn.Transformer(d_model=ninp,nhead=nhead,num_encoder_layers=nlayers,num_decoder_layers=nlayers,dim_feedforward=nhid,dropout=dropout)
        self.fc = nn.Linear(ninp,out_token)
        
    def generate(self,out):
        output = rearrange(out,'t n e -> n t e')
        output = self.fc(output)
        return output

    def forward(self,src,src_pad_mask,tgt,tgt_mask,tgt_pad_mask,mem_mask):
        src = self.encoder_embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = rearrange(src,'n s e-> s n e')
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        tgt = rearrange(tgt, 'n t e-> t n e')
        
        out = self.transformer(src=src,src_key_padding_mask=src_pad_mask,tgt=tgt,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=mem_mask)
        return self.generate(out)