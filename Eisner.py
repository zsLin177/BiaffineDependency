import torch
from supar.utils.fn import pad

def eisner(scores, mask):
    '''
    :param scores: [batch_size,seq_len,seq_len]
    :param mask: [batch_size,seq_len]
    The first column with pseudo words as roots should be set to False.
    :return: tree: [batch_size,seq_len]
    '''
    batch_size, seq_len, _ = scores.shape
    scores = scores.transpose(1,2).contious()
    lens = mask.sum(-1)
    # lens: [batch_size]
    # exclude root:0
    scores = scores.cpu().unbind()
    # scores: a tuple with batch_size [seq_len,seq_len]

    def backtrack(p_i,p_c,pred,i,j):
        '''
        i: the head of the arc
        j: the tail of the arc
        pred: store the result
        '''
        if(i == j):
            return
        r = p_c[i, j].tolist()[0]
        pred[r] = i

        backtrack(p_i,p_c,pred,r,j)
        r2=p_i[i, r].tolist()[0]
        if(i < j): # 正向
            backtrack(p_i,p_c,pred,i,r2)
            backtrack(p_i,p_c,pred,r,r2+1)
        else: # 逆向
            backtrack(p_i,p_c,pred,r,r2)
            backtrack(p_i,p_c,pred,i,r2+1)

    preds = []
    for i in range(batch_size):
        s_c = torch.full_like(scores[i], float('-inf'))
        s_c.diagonal().fill_(0)
        s_i = torch.full_like(scores[i], float('-inf'))
        p_c = torch.zeros_like(scores[i]).long()
        p_i = torch.zeros_like(scores[i]).long()
        len=int(lens.tolist()[i])
        if(len == 1):
            preds.append(torch.LongTensor([0,0]))
            continue
        for j in range(2, len + 2):
            # j is the len of span
            # positive
            for k in range(0, len - j + 2):
                # imcomplete
                temp=[s_c[k,r] + s_c[k+j-1,r+1] + scores[i][k, k+j-1] for r in range(k,k+j-1)]
                s_i[k,k+j-1] = max(temp)
                p_i[k,k+j-1] = temp.index(max(temp))+k
                # complete
                temp = [s_i[k,r] + s_c[r,k+j-1] for r in range(k+1,k+j)]
                s_c[k,k+j-1] = max(temp)
                p_c[k,k+j-1] = temp.index(max(temp))+k+1

            # negative
            for k in range(len, j-1, -1):
                # imcomplete
                temp = [s_c[k-j+1, r] + s_c[k,r+1] + scores[i][k, k-j+1] for r in range(k-j+1,k)]
                s_i[k, k-j+1] = max(temp)
                p_i[k, k-j+1] = temp.index(max(temp))+ k-j+1
                # complete
                temp = [s_c[r, k-j+1] + s_i[k, r] for r in range(k-j+1,k)]
                s_c[k,k-j+1] = max(temp)
                p_c[k,k-j+1] = temp.index(max(temp)) + k-j+1

        # backtrack
        pred=[0]*(len+1)
        backtrack(p_i,p_c,pred,0,len)
        pred=torch.LongTensor(pred)
        preds.append(pred)

    return pad(preds, total_length=seq_len).to(mask.device)

if __name__ == '__main__':
    x=torch.LongTensor([[1,2,3],[0,1,2]])
    print(x)
    x[0][0]=0
    print(x)





