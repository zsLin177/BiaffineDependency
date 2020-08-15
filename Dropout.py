import torch
import torch.nn as nn


class IndependentDroupout(nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p

    def forward(self,x,y):
        '''

        :param x: [batch_size,seq_len,dim1]
        :param y: [batch_size,seq_len,dim2]
        :return: x,y with same size as input, but compensated
        '''

        if (self.training): #只有训练的时候才要dropout
            batch_size=x.size(0)
            seq_len=x.size(1)
            masks=[torch.ones((batch_size,seq_len)).bernoulli_(1 - self.p) for _ in range(2)]
            total=sum(masks)
            scale = 2 / total.max(torch.ones_like(total))
            # 来计算每份分多少，之后乘多少。用这种来避免除0
            masks=[mask*scale for mask in masks]
            x=x*masks[0].unsqueeze(-1)
            y=y*masks[1].unsqueeze(-1)
        return x,y

class SharedDropout(nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p

    def forward(self,x):
        '''
        每个batch用相同的dropout
        :param x: [batch_size,seq_len,dim]
        :return:
        '''
        if(self.training):
            batch_size=x.size(0)
            dim=x.size(2)
            masks=[torch.zeros((1,dim)).bernoulli_(1 - self.p)/(1-self.p) for _ in range(batch_size)]
            for i in range(batch_size):
                x[i]=x[i]*masks[i]
        return x


if __name__ == '__main__':
    # idr=IndependentDroupout(0.5)
    sd=SharedDropout(0.2)
    # ssd=SSD(0.3)
    x = torch.randint(1, 2, (2, 3, 5))
    y = torch.rand(2,3,5)
    print(y)
    # new_x=ssd(x)
    # print(new_x)
    y=sd(y)
    print(y)
    # idr(x,y)

