import torch
from supar.utils.field import Field, SubwordField
from supar.utils import Dataset, Embedding
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL



# print(f"\nTotal words: {len(WORD.vocab)}\nTotal words in train data: {WORD.vocab.n_init}")
# print(f"Total characters: {len(CHAR.vocab)}")
# print(f"Total labels: {len(REL.vocab)}")
#
# print(f"Numericalize 'John saw Marry': {WORD.transform([['John', 'saw', 'Marry']])}")
# print(WORD.vocab[[2, 11312, 17403, 12647]])
# print(REL.vocab.stoi)

def get_init(dim=100):
    WORD = Field('words', pad='<pad>', unk='<unk>', bos='<bos>', lower=True)
    CHAR = SubwordField('chars', pad='<pad>', unk='<unk>', bos='<bos>', fix_len=20)
    ARC = Field('arcs', bos='<bos>', use_vocab=False, fn=CoNLL.get_arcs)
    REL = Field('rels', bos='<bos>')

    transform = CoNLL(FORM=(WORD, CHAR), HEAD=ARC, DEPREL=REL)
    # 这个form=(WORD, CHAR)，HEAD=ARC, DEPREL=REL就是指明了各个域所作用的列
    train = Dataset(transform, 'data/ptb/train.conllx')
    # 建立train数据集
    if(dim==100):
        WORD.build(train, 2, Embedding.load('data/glove.6B.100d.txt', 'unk'))
    elif(dim==300):
        WORD.build(train, 2, Embedding.load('data/glove.6B.300d.txt', 'unk'))
    # 用训练集初始化word词典，出现次数低于2的视为未知词，加载预训练词向量：放在WORD.embed中，当然词典中包含预训练的词
    CHAR.build(train)
    REL.build(train)
    return WORD,CHAR,ARC,REL,transform

def get_dataset(transform):
    print("Load the data")
    train = Dataset(transform, 'data/ptb/train.conllx')
    dev = Dataset(transform, 'data/ptb/dev.conllx')
    test = Dataset(transform, 'data/ptb/test.conllx')
    print("Numericalize the data")
    train.build(batch_size=5000, n_buckets=32, shuffle=True)
    dev.build(batch_size=5000, n_buckets=32)
    test.build(batch_size=5000, n_buckets=32)

    # print(f"\ntext:\n{train.sentences[1]}")
    # print(f'\nwords:\n{train.words[1]}')
    # print(f'\nchars:\n{train.chars[1]}')
    # print(f"arcs: {train.arcs[1]}")
    # print(f"rels: {train.rels[1]}")
    return train,dev,test

if __name__ == '__main__':
    get_dataset()