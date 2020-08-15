import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class charlstm(nn.Module):
    def __init__(self,char_num,d_embed,d_out,pad_index=0):
        super().__init__()
        self.char_num=char_num
        self.d_embed=d_embed
        self.d_out=d_out
        self.pad_idx=pad_index

        self.embed=nn.Embedding(char_num,d_embed,padding_idx=pad_index)

        self.lstm=nn.LSTM(input_size=d_embed,hidden_size=d_out//2,bidirectional=True)

    def forward(self,x):
        '''
        :param x: [batch_size,seq_len,fix_len]
        :return: [batch_size,seq_len,d_out]
        将一个单词的字母序列作为bilstm的输入，将正向和反向的最后位置的hidden合并，作为该单词的charlstm表示
        '''
        batch_size=x.size(0)
        seq_len=x.size(1)
        fix_len=x.size(2)
        x = self.embed(x)
        # x:[batch_size,seq_len,fix_len,d_embed]
        x=x.view(-1,fix_len,self.d_embed)
        mask1=x.ne(0)
        sum1=mask1.sum(-1)
        mask2=sum1.ne(0)
        words_len=mask2.sum(-1)
        x = pack_padded_sequence(x, words_len, batch_first=True, enforce_sorted=False)
        x, (h, _) = self.lstm(x)
        # h:[2,batch_size*seq_len,d_out//2]
        # h=h.transpose(0,1).contiguous().view(batch_size*seq_len,-1)
        h = torch.cat(torch.unbind(h), -1)
        # h:[batch_size*seq_len,d_out]
        return h.view(batch_size,seq_len,-1)

if __name__ == '__main__':
    cl = charlstm(5, 10, 20, 0)
    x = torch.randint(0, 2, (2, 3, 4))
    print(x)
    x = cl(x)
    print(x)

