from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from load_data import get_dataset,get_init
from model import BiaffineDependencyModel
from ruler import AttachmentMetric
import os
from datetime import datetime, timedelta
import torch.nn as nn
import torch
from supar.utils.fn import ispunct
from supar.utils.metric import Metric
from Optm import ScheduledOptim




def train_loader(model, optimizer, scheduler, loader,WORD):
    model.train()

    # metric = AttachmentMetric()

    for words, feats, arcs, rels in loader:
        optimizer.zero_grad()

        mask = words.ne(WORD.pad_index)
        # ignore the first token of each sentence
        mask[:, 0] = 0
        # print(words.device)
        # print(model.device)
        s_arc, s_rel = model(words, feats)
        loss = model.loss(s_arc, s_rel, arcs, rels, mask)
        loss.backward()
        if(scheduler!=None):
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
        else:
            optimizer.step_and_update_lr()
        # arc_preds, rel_preds = model.decode(s_arc, s_rel, mask)
        # ignore all punctuation if not specified
        # mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
        # metric(arc_preds, rel_preds, arcs, rels, mask)

# @torch.no_grad()
def evaluate_loader(model, loader,WORD,puncts):
    model.eval()

    total_loss, metric = 0, AttachmentMetric()

    for words, feats, arcs, rels in loader:
        mask = words.ne(WORD.pad_index)
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = model(words, feats)
        loss = model.loss(s_arc, s_rel, arcs, rels, mask)
        arc_preds, rel_preds = model.decode(s_arc, s_rel, mask)
        total_loss += loss.item()
        mask &= words.unsqueeze(-1).ne(puncts).all(-1)
        metric(arc_preds, rel_preds, arcs, rels, mask)
    total_loss /= len(loader)

    return total_loss, metric

def train_parser(train, dev, test, model, optimizer,transform,WORD,puncts,encoder,
          path='model',
          epochs=5000,
          patience=3):
    transform.train()

    elapsed = timedelta()
    best_e, best_metric = 1, Metric()
    if(encoder=='lstm'):
        scheduler = ExponentialLR(optimizer, .75 ** (1 / epochs))
    else:
        scheduler=None
    f=open('temp_log','w')
    for epoch in range(1, epochs + 1):
        start = datetime.now()

        print(f"Epoch {epoch} / {epochs}:")
        f.write(f"Epoch {epoch} / {epochs}:"+'\n')
        train_loader(model, optimizer, scheduler, train.loader,WORD)

        # scheduler.step()

        loss, dev_metric = evaluate_loader(model, dev.loader,WORD,puncts)
        print(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
        f.write(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}"+'\n')
        loss, test_metric = evaluate_loader(model, test.loader,WORD,puncts)
        print(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")
        f.write(f"{'test:':6} - loss: {loss:.4f} - {test_metric}"+'\n')
        f.flush()

        t = datetime.now() - start
        # save the model if it is the best so far
        if dev_metric > best_metric:
            best_e, best_metric = epoch, dev_metric
            model.save(path)
            print(f"{t}s elapsed (saved)\n")
            f.write(f"{t}s elapsed (saved)\n")
        else:
            print(f"{t}s elapsed\n")
            f.write(f"{t}s elapsed\n")
        f.flush()
        elapsed += t
        if epoch - best_e >= patience:
            break
    loss, metric = evaluate_loader(model.load(path), test.loader,WORD,puncts)
    f.close()

    print(f"Epoch {best_e} saved")
    print(f"{'dev:':6} - {best_metric}")
    print(f"{'test:':6} - {metric}")
    print(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")


def train(WORD, CHAR, ARC, REL, transform,encoder,epoch=60,word_dim=100):
    model = BiaffineDependencyModel(n_words=WORD.vocab.n_init,
                                    n_feats=len(CHAR.vocab),
                                    n_rels=len(REL.vocab),
                                    pad_index=WORD.pad_index,
                                    unk_index=WORD.unk_index,
                                    bos_index=WORD.bos_index,
                                    feat_pad_index=CHAR.pad_index,
                                    encoder=encoder,
                                    n_embed=word_dim)
    model.load_pretrained(WORD.embed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    puncts = torch.tensor([i for s, i in WORD.vocab.stoi.items() if ispunct(s)]).to(device)

    train, dev, test = get_dataset(transform)
    # print(len(train.sentences))
    train.sentences = train.sentences[:3]
    dev.sentences = dev.sentences[:200]
    test.sentences = test.sentences[:200]
    print('train sentences:%d dev sentences:%d test sentences:%d' % (
    len(train.sentences), len(dev.sentences), len(test.sentences)))
    if (encoder == 'lstm'):
        optimizer = Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.9), eps=1e-12)
    else:
        optimizer = ScheduledOptim(Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, 800, 4000)

    train_parser(train, dev, test, model, optimizer,transform, WORD, puncts, encoder,epochs=epoch,
                      path=encoder+'_model')

def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(177233)
    WORD, CHAR, ARC, REL, transform=get_init(100)
    train(WORD, CHAR, ARC, REL, transform,'lstm',epoch=600,word_dim=100)