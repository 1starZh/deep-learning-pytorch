import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class multiheadattention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(multiheadattention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        
    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    
class positionalencoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=100):
        super(positionalencoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
class positionwiseFFn(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_output, **kwargs):
        super(positionwiseFFn, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)
        
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
class addnormal(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(addnormal, self).__init__(**kwargs)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        return self.ln(self.drop(Y) + X)
    
class encoderblock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(encoderblock, self).__init__(**kwargs)
        self.attention = multiheadattention(key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnormal1 = addnormal(norm_shape, dropout)
        self.ffn = positionwiseFFn(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnormal2 = addnormal(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnormal1(X, self.attention(X, X, X, valid_lens))
        return self.addnormal2(Y, self.ffn(Y))
    
class transformerencoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(transformerencoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positionalencoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), encoderblock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
            
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
    
class decoderblock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(decoderblock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = multiheadattention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnormal1 = addnormal(norm_shape, dropout)
        self.attention2 = multiheadattention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnormal2 = addnormal(norm_shape, dropout)
        self.ffn = positionwiseFFn(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnormal3 = addnormal(norm_shape, dropout)
        
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else: 
            dec_valid_lens = None
            
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnormal1(X, X2)
        
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnormal2(Y, Y2)
        
        return self.addnormal3(Z, self.ffn(Z)), state
    
class transformerdecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(transformerdecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positionalencoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                decoderblock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights
    
class encoderdecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(encoderdecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
    
def train(num_hiddens, num_layers, dropout, batch_size, num_steps,
            lr, num_epochs, ffn_num_input, ffn_num_hiddens,
            num_heads, key_size, query_size, value_size,
            norm_shape, device=None):
    device = device or d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = transformerencoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = transformerdecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = encoderdecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    return net, src_vocab, tgt_vocab

net, src_vocab, tgt_vocab = train(
    num_hiddens=32, num_layers=2, dropout=0.1,
    batch_size=64, num_steps=10,
    lr=0.005, num_epochs=200,
    ffn_num_input=32, ffn_num_hiddens=64,
    num_heads=4, key_size=32, query_size=32, value_size=32,
    norm_shape=[32]
)
