import torch.nn as nn
import torch
from supar.modules import MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.utils import Config
from supar.utils.alg import eisner, eisner2o, mst
from supar.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attention_encoder import Attention_encoder,PositionalEncoding

class BiaffineDependencyModel(nn.Module):
    '''
    n_words: num of word in train
    n_feats: num of char in train
    n_rels: num of tag in train
    '''
    def __init__(self,
                 n_words,
                 n_feats,
                 n_rels,
                 encoder='lstm',
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 n_att_layers=6,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()
        self.args = Config().update(locals())

        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        # can trained embed:word in train

        self.feat_embed = CharLSTM(n_chars=n_feats,
                                   n_embed=n_char_embed,
                                   n_out=n_feat_embed,
                                   pad_index=feat_pad_index)

        self.embed_dropout = IndependentDropout(p=embed_dropout)
        # 输入层的dropout，采用独立dropout

        self.encoder_type=encoder
        if(encoder=='lstm'):
            self.encoder = BiLSTM(input_size=n_embed + n_feat_embed,
                               hidden_size=n_lstm_hidden,
                               num_layers=n_lstm_layers,
                               dropout=lstm_dropout)
            self.lstm_dropout = SharedDropout(p=lstm_dropout)
            # 编码层lstm以及shared dropout

        elif(encoder=='att'):
            d_input=n_embed + n_feat_embed
            self.linear1=nn.Linear(d_input,n_lstm_hidden * 2) # 前加
            self.encoder=Attention_encoder(d_model=n_lstm_hidden * 2,n_layers=n_att_layers)
            # self.linear2=nn.Linear(512,n_lstm_hidden * 2,bias=False) # 后加

        self.mlp_arc_d = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        # 四个不同的全连接层，映射到对应的维度

        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, feats):
        """
        Args:
            words (LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].

        Returns:
            s_arc (Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        # tensor.ne()
        # 这个mask和words一样的shape，在self.pad_index的地方为false，别的真实的地方为true
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            # 这个ext_mask是和word一样的shape，在小于self.word_embed.num_embeddings的地方为false，大于等于为true
            ext_words = words.masked_fill(ext_mask, self.unk_index)
            # 在mask为true的地方，对应在ext_words的位置上改为self.unk_index


        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        if(self.encoder_type=='lstm'):
            x = pack_padded_sequence(embed, mask.sum(1), True, False)
            # batch_first=True,enforce_sorted=False

            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=seq_len)
            x = self.lstm_dropout(x)
        elif(self.encoder_type=='att'):
            x=self.linear1(embed) # 前加
            x=self.encoder(x)
            # x=self.linear2(x) #后加

        # apply MLPs to the output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1) #permute交换维度
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        """
        Args:
            s_arc (Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            rels (LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (BoolTensor): [batch_size, seq_len]
                Mask for covering the unpadded tokens.

        Returns:
            loss (Tensor): scalar
                The training loss.
        """
        s_arc, arcs = s_arc[mask], arcs[mask]
        # s_arc:[L,seq_len],arcs:[L]每个词对应head的号码
        s_rel, rels = s_rel[mask], rels[mask]
        # rels:[L]每个词对应弧的种类的标号
        # s_rel: [batch_size, seq_len, seq_len, n_labels]
        # s_rel[mask] -> [L, seq_len, n_labels]
        # s_rel[torch.arange(len(arcs)), arcs] -> [L, n_labels]

        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        # 举个例子，对于第n个词，取出来的是原来s_rel[n,arc[n]]一个label是序列
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)
        # 交叉熵要求输入是分数，不是概率。且s_arc是二维的（包含所有位置分数），arcs是一维的（包含所有正确位置的标号）

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """
        Args:
            s_arc (Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            mask (BoolTensor): [batch_size, seq_len]
                Mask for covering the unpadded tokens.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.

        Returns:
            arc_preds (Tensor): [batch_size, seq_len]
                The predicted arcs.
            rel_preds (Tensor): [batch_size, seq_len]
                The predicted labels.
        """

        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        # 具体我也不懂啊，看效果
        arc_preds = mst(s_arc, mask)
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    @classmethod
    def load(cls, path):
        # 这个cls学到了
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        model = cls(**state['args'])
        # 输入字典，初始化生成model
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
