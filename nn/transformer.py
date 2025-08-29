import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, max_len, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        pos_emb = self.pos_emb(x)
        return self.drop_out(pos_emb)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.linear_q(q)   
        k = self.linear_k(k)
        v = self.linear_v(v)
        
        def split_heads(x):
            return x.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.bool() == False, float('-inf'))

        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(output)
        
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)
    

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(q=norm_x, k=norm_x, v=norm_x, mask=src_mask)
        x = x + self.dropout1(attn_out)

        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout2(ffn_out)

        return x
    

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.norm2 = nn.LayerNorm(d_model)
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        x = self.norm1(dec)
        x, _ = self.self_attention(q=x, k=x, v=x, mask=trg_mask)
        x = _x + self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.norm2(x) 
            x, _ = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = _x + self.dropout2(x)

        _x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = _x + self.dropout3(x)

        return x


class Transformer_Encoder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, max_len, drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    

class Transformer_Decoder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, max_len, drop_prob)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        return trg
    

class Transformer(nn.Module):
    def __init__(self, seg_len, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob=0.1):
        super().__init__()
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.seg_len = seg_len

        self.encoder = Transformer_Encoder(max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob)
        self.decoder = Transformer_Decoder(max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob)

        self.trg_attention_mask_cache = {}

    def create_src_mask(self, src, eps=1e-6):
        mask = (src.norm(dim=-1) > eps).unsqueeze(1).unsqueeze(2)
        return mask

    def create_trg_mask(self, trg, eps=1e-6):
        seq_len = trg.size(1)
        if seq_len not in self.trg_attention_mask_cache:
            self.trg_attention_mask_cache[seq_len] = ~torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=trg.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
        trg_attention_mask = self.trg_attention_mask_cache[seq_len]
        trg_pad_mask = (trg.norm(dim=-1) > eps).unsqueeze(1).unsqueeze(2)
        combined_mask = trg_pad_mask & trg_attention_mask
        return combined_mask

    def forward(self, src, trg):
        src_mask = self.create_src_mask(src)

        start_token_batch = self.start_token.expand(trg.size(0), 1, -1)
        trg = torch.cat([start_token_batch, trg[:, :-1, :]], dim=1)

        trg_mask = self.create_trg_mask(trg)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        
        return dec_output

    def evaluate(self, src):
        src_mask = self.create_src_mask(src)
        enc_output = self.encoder(src, src_mask)
        
        batch_size = src.size(0)
        dec_input = self.start_token.expand(batch_size, 1, -1)
        
        for _ in range(self.seg_len):
            trg_mask = self.create_trg_mask(dec_input)
            
            dec_output = self.decoder(dec_input, enc_output, trg_mask, src_mask)
            
            next_vector = dec_output[:, -1:, :]
            
            dec_input = torch.cat((dec_input, next_vector), dim=1)
            
        return dec_input[:, 1:, :]


if __name__ == '__main__':
    model = Transformer(seg_len=2, max_len=1000, d_model=64, ffn_hidden=256, n_head=4, n_layers=2, drop_prob=0.1)
    x = torch.randn(2, 4, 64)
    z = torch.randn(2, 2, 64)
    y = model(x, z)
    torch.save(model.state_dict(), "model/test_model.pth")
    print(y.shape)