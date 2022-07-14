import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import random

import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def input_max_len(feature):
    src_len = 0 # enc_input max sequence length
    for i in range(len(feature)):
        if len(feature[i])>src_len:
            src_len = len(feature[i])
    return src_len

src_len = 10

src_vocab = {'Empty':0, 'A': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
                'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 
                'T': 17, 'V': 18, 'W': 19, 'Y': 20}
src_vocab_size = len(src_vocab)

def make_data(features,src_len):
    enc_inputs = []
    for i in range(len(features)):
        enc_input = [[src_vocab[n] for n in list(features[i])]]
        while len(enc_input[0])<src_len:
            enc_input[0].append(0)
        enc_inputs.extend(enc_input)
    return torch.LongTensor(enc_inputs)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs,labels):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.labels = labels

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.labels[idx]


# Transformer Parameters
d_model = 512  # Embedding size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder and Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        # div_term [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    :return:
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B,S,D) - proj -> (B,S,D_new) -split -> (B, S, H, W) -> trans -> (B,H,S,W)

        Q = self.W_Q(input_Q).view(batch_size,-1, n_heads, d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, n_heads, d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, n_heads, d_v).transpose(1,2) # V:[batch_size, n_heads, len_v(=len_k, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask: [batch_size,n_heads, seq_len, seq_len]

        # [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q,K,V, attn_mask)
        context = context.transpose(1,2).reshape(batch_size, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)

        return nn.LayerNorm(d_model).to(device)(output+residual),attn # Layer Normalization


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output+residual) #[batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,enc_inputs, enc_self_attn_mask):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return:
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return:
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_layer = [512,256,64,32,1]
        self.e1 = nn.Linear(input_dim, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e5 = nn.Linear(hidden_layer[3], hidden_layer[4])
        self.batchnorm0 = nn.BatchNorm1d(hidden_layer[0])
        self.batchnorm1 = nn.BatchNorm1d(hidden_layer[1])
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer[2])
        self.batchnorm3 = nn.BatchNorm1d(hidden_layer[3])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,dec_input):
        h_1 = F.leaky_relu(self.batchnorm0(self.e1(dec_input)), negative_slope=0.05, inplace=True)
        h_2 = F.leaky_relu(self.batchnorm1(self.e2(h_1)), negative_slope=0.05, inplace=True)
        h_3 = F.leaky_relu(self.e3(h_2), negative_slope=0.1, inplace=True)
        h_4 = F.leaky_relu(self.e4(h_3), negative_slope=0.1, inplace=True)
        y = self.e5(h_4)
        return y

class TRN(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Classifier(src_len*d_model).to(device)

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''

        # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs,enc_self_attns = self.encoder(enc_inputs)

        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1)) 
        pred = self.decoder(dec_inputs)

        return pred

# load the TRN parameters
model_load = TRN()
checkpoint = torch.load('model_TRN.pt')
model_load.load_state_dict(checkpoint)
model_load.eval()

# read the peptides
test_file = 'deca_peptides.csv'
df_test = pd.read_csv(test_file)
test_feat = np.array(df_test["Feature"])

# predict in batches
predict = []
for itr in range(1000):
    test_feat_slide = test_feat[itr*1000:(itr+1)*1000]
    test_enc_inputs = make_data(test_feat_slide,src_len).to(device)
    predict_test = model_load(test_enc_inputs)
    predict = predict + predict_test.squeeze(1).cpu().detach().numpy().tolist()

# save file
df_test_save = pd.DataFrame()
df_test_save['Feature'] = df_test["Feature"]
df_test_save['Predict'] = predict
df_test_save.to_csv('AP_predict.csv')
