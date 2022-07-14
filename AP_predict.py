import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import random

import pandas as pd

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

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

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1], pos向量
        # div_term [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位赋值 [max_len,d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # 技术位赋值 [max_Len,d_model/2]
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len,1,d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :] # 直接将pos_embedding 和 vocab_embedding相加
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
    # 举个例子，输入为 seq_data = [1, 2, 3, 4, 0]，seq_data.data.eq(0) 就会返回 [False, False, False, False, True]
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

        # 分解为MultiHead Attention
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

        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm0 = nn.BatchNorm1d(hidden_layer[0])
        self.batchnorm1 = nn.BatchNorm1d(hidden_layer[1])
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer[2])
        self.batchnorm3 = nn.BatchNorm1d(hidden_layer[3])

        self.sigmoid = nn.Sigmoid()
    def forward(self,dec_input):

        # dec_input = self.dropout(dec_input)
        h_1 = F.leaky_relu(self.batchnorm0(self.e1(dec_input)), negative_slope=0.05, inplace=True)
        # h_1 = F.leaky_relu(self.e1(dec_input), negative_slope=0.1, inplace=True)
        h_1 = self.dropout(h_1)
        h_2 = F.leaky_relu(self.batchnorm1(self.e2(h_1)), negative_slope=0.05, inplace=True)
        # h_2 = F.leaky_relu(self.e2(h_1), negative_slope=0.1, inplace=True)
        h_2 = self.dropout(h_2)
        # h_3 = F.leaky_relu(self.batchnorm2(self.e3(h_2)), negative_slope=0.05, inplace=True)
        h_3 = F.leaky_relu(self.e3(h_2), negative_slope=0.1, inplace=True)
        h_3 = self.dropout(h_3)
        # h_4 = F.leaky_relu(self.batchnorm3(self.e4(h_3)), negative_slope=0.05, inplace=True)
        h_4 = F.leaky_relu(self.e4(h_3), negative_slope=0.1, inplace=True)
        # h_4 = self.dropout(h_4)
        y = self.e5(h_4)


        return y

class Transformer(nn.Module):
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

        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, n_heads,tgt_len, src_len]
        pred = self.decoder(dec_inputs)

        return pred

criterion = nn.CrossEntropyLoss(ignore_index=0)

def calculate_acc(label, predict):
    sample_num = label.shape[0]
    index = np.argmax(predict,1)
    acc = (index == label).sum()/sample_num
    return acc

def training(seed_id,data_size,model,optimizer):

    # df_train = pd.read_csv('10000dataAP-pentaAPcsv-zihan.csv')
    

    # order = list(range(0,df_train.shape[0]))
    # valid_size = int(df_train.shape[0]/10)
    # random.shuffle(order)

    # test_scale = 2000
    # train_valid_scale = df_train.shape[0] - test_scale
    # if data_size <= train_valid_scale:
    #     train_valid_scale = data_size
    
    # valid_scale = int(train_valid_scale/5)
    # train_scale = train_valid_scale - valid_scale

    # train_order = order[test_scale+valid_scale:test_scale+valid_scale+train_scale]
    # valid_order = order[test_scale:test_scale+valid_scale]

    # train_feat = np.array(df_train["Feature"])[train_order]
    # valid_feat = np.array(df_train["Feature"])[valid_order]

    # train_enc_inputs = make_data(train_feat,src_len).to(device)
    # valid_enc_inputs = make_data(valid_feat,src_len).to(device)

    # train_label = torch.Tensor(np.array(df_train["Label"])[train_order]).to(device).unsqueeze(1)
    # valid_label = torch.Tensor(np.array(df_train["Label"])[valid_order]).to(device).unsqueeze(1)

    # train_loader = Data.DataLoader(MyDataSet(train_enc_inputs,train_label), 512, True)

    # valid_best = 100
    # loss_func = torch.nn.MSELoss()
    # for epoch in range(200):
    #     model.train()
    #     for enc_inputs,labels in train_loader:
    #         enc_inputs = enc_inputs.to(device)
            
    #         outputs = model(enc_inputs)
    #         loss = loss_func(outputs, labels)

    #         # print('Epoch:','%04d' % (epoch+1), 'loss =','{:.6f}'.format(loss))

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     if (epoch+1) % 1 == 0:
    #         model.eval()
    #         predict = model(valid_enc_inputs)

    #         MSE_valid = loss_func(predict,valid_label).item()

    #         if valid_best > MSE_valid:
    #             valid_best = MSE_valid
    #             print('Epoch:',epoch+1)
    #             print('Validation MSE:',MSE_valid)
    #             torch.save(model.state_dict(),'model_{}.pt'.format(data_size))

    test_files = ['3.2million_data1_A.csv','3.2million_data2_C.csv','3.2million_data3_D.csv','3.2million_data4_E.csv',
                    '3.2million_data5_F.csv','3.2million_data6_G.csv','3.2million_data7_H.csv','3.2million_data8_I.csv',
                    '3.2million_data9_K.csv','3.2million_data10_L.csv','3.2million_data11_M.csv','3.2million_data12_N.csv',
                    '3.2million_data13_P.csv','3.2million_data14_Q.csv','3.2million_data15_R.csv','3.2million_data16_S.csv',
                    '3.2million_data17_T.csv','3.2million_data18_V.csv','3.2million_data19_W.csv','3.2million_data20_Y.csv']

    model_load = Transformer()
    checkpoint = torch.load('model_{}.pt'.format(data_size))
    model_load.load_state_dict(checkpoint)
    model_load.eval()

    file_id = 0
    for test_file in test_files:
        file_id = file_id + 1
        df_test = pd.read_csv(test_file)
        test_feat = np.array(df_test["Feature"])

        predict = []
        for itr in range(100):
            print('We are now running file no.{} iter no.{}'.format(file_id,itr))
            test_feat_slide = test_feat[itr*1600:(itr+1)*1600]
            test_enc_inputs = make_data(test_feat_slide,src_len).to(device)
            predict_test = model_load(test_enc_inputs)
            predict = predict + predict_test.squeeze(1).cpu().detach().numpy().tolist()
            
        df_test_save = pd.DataFrame()
        df_test_save['Feature'] = df_test["Feature"]
        df_test_save['Predict'] = predict
        df_test_save.to_csv('results/32e6_data{}.csv'.format(file_id))

data_sizes = [10000000]
seed = [5]

for seed_id in seed:
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed_id)
    for data_size in data_sizes:
        model = Transformer().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.2)
        training(seed_id,data_size,model,optimizer)