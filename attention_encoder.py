import torch
import torch.nn as nn
import numpy as np
from Layers import EncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        # 放到内存，对于每次不同的输入句子，位置词向量是不需要改变的（不需要去学习改变的）
        # 之后使用就用的是self.pos_table

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        # 先整除2再乘以2，是因为对于奇数维也变成相应的偶数维

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # [:, 0::2]前一个：对所有的行，后面是对偶数列，抽出来构成一个新的array

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
        #unsqueeze表示增加一个维度（加一个[])，将参数作为一个整体

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
        # .clone.detach(),开辟一个新内存，复制内容，从计算图中抽出来，不计算梯度
        # x:[batch_size,seq_len,embed_size] pos_table:[1,200,embed_size]

class Attention_encoder(nn.Module):
    '''
    d_model: the dim of input and out
    n_head: head of muliti-head attention
    d_x: the dim of one head
    n_layers: num of layer of the encoder
    n_positions: max len of the sentence

    borrow from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer
    '''
    def __init__(self,d_model=512,
                 n_head=8,
                 n_layers=6,
                 d_inner=2048,
                 n_positions=200,
                 dropout=0.1):
        super().__init__()
        self.d_k=self.d_v=self.d_q=d_model//n_head
        self.position_enc = PositionalEncoding(d_model, n_position=n_positions)
        # self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, self.d_k, self.d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,embed):
        '''
        :param embed: [batch_size,seq_len,embed_size]
        :return: [batch_size,seq_len,embed_size]
        '''
        x_posed=self.position_enc(embed)
        enc_output=self.layer_norm(x_posed)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=None)

        return enc_output



