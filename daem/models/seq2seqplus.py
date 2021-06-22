from functools import partial
import tqdm
import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import Precision, Recall, F1
from daem.models.nn import SentenceConv
from daem.utils.cache_decorator import CacheDecoreator
from daem.models.base import EntityMatchingModule

class Seq2SeqDifference(nn.Module):
    def __init__(self, dim, dense_hidden=60):
        super(Seq2SeqDifference, self).__init__()
#         self.embeddings = embeddings_layer
        self.dim = dim
        self.conv1 = SentenceConv(1, 100, self.dim)
        self.conv2 = SentenceConv(2, 100, self.dim)
        self.conv3 = SentenceConv(3, 100, self.dim)
        self.dense = nn.Sequential(
            nn.Linear(100 * 6, dense_hidden),
            nn.ReLU()
        )
    def forward(self, l_lengths, l_vectors, r_lengths, r_vectors):
        device = l_lengths.device

        features = []
        
        if torch.max(l_lengths) == 0 or torch.max(r_lengths) == 0:
            for i in range(0, 6):
                features.append(torch.zeros((l_lengths.shape[0], 100), device=device))
        else:
            l_mask = pad_sequence([torch.ones(k) for k in l_lengths]).to(device)
            r_mask = pad_sequence([torch.ones(k) for k in r_lengths]).to(device)

            attention = torch.bmm(l_vectors.transpose(0, 1), r_vectors.transpose(0, 1).transpose(1, 2))
            attention = attention * l_mask.T.unsqueeze(2) * r_mask.T.unsqueeze(1)

            attention_r = torch.nn.functional.softmax(attention - 10 * (1 - l_mask.T.unsqueeze(2)), dim=1)
            attention_r = attention_r * l_mask.T.unsqueeze(2)
            attention_r = attention_r / (attention_r.sum(dim=1, keepdim=True) + 1e-13)

            attention_l = torch.nn.functional.softmax(attention - 10 * (1 - r_mask.T.unsqueeze(1)), dim=2)
            attention_l = attention_l * r_mask.T.unsqueeze(1)
            attention_l = attention_l / (attention_l.sum(dim=2, keepdim=True) + 1e-13)

            att = attention_l
            l_mean = att.sum(2) / r_lengths.unsqueeze(1)
            l_w = (att * att).sum(2) / r_lengths.unsqueeze(1) - l_mean * l_mean
            l_w = l_w / l_mean.clamp_min(0.001)
            l_a, l_i = torch.topk(att, 1, 2)
            l_i = l_i * l_mask.T.int().unsqueeze(2)
            l_a = l_a / l_a.sum(2, keepdim=True).clamp_min(0.001)

            l_counter = torch.zeros(l_vectors.shape, device=device)
            for j in range(0, 1):
                indices = l_i[:, :, j].transpose(0, 1).unsqueeze(2).expand(-1, -1, self.dim)
                l_counter += l_a[:, :, j].T.unsqueeze(2) * r_vectors.gather(0, indices)
            l_c = l_w.T.unsqueeze(2) * (l_vectors - l_counter).abs() * l_mask.unsqueeze(2)

            att = attention_r
            r_mean = att.sum(1) / l_lengths.unsqueeze(1)
            r_w = (att * att).sum(1) / l_lengths.unsqueeze(1) - r_mean * r_mean
            r_w = r_w / r_mean.clamp_min(0.001)

            r_a, r_i = torch.topk(att, 1, 1)
            r_i = r_i * r_mask.T.int().unsqueeze(1)
            r_a = r_a / r_a.sum(1, keepdim=True).clamp_min(0.001)

            r_counter = torch.zeros(r_vectors.shape, device=device)
            for j in range(0, 1):
                indices = r_i[:, j, :].T.unsqueeze(2).expand(-1, -1, self.dim)
                r_counter += r_a[:, j, :].T.unsqueeze(2) * l_vectors.gather(0, indices)
            r_c = r_w.T.unsqueeze(2) * (r_vectors - r_counter).abs() * r_mask.unsqueeze(2)
            for the_c, the_mask in [(l_c, l_mask), (r_c, r_mask)]:
                features.append(self.conv1(the_c, the_mask))
                if the_c.shape[0] >= 2:
                    features.append(self.conv2(the_c, the_mask))
                else:
                    features.append(torch.zeros_like(features[-1]))
                if the_c.shape[0] >= 3:
                    features.append(self.conv3(the_c, the_mask))
                else:
                    features.append(torch.zeros_like(features[-1]))
        return self.dense(torch.cat(features, 1))


class Seq2SeqPlusMatcher(nn.Module):
    def __init__(self, word_embeddings, columns, dense_hidden=60):
        super(Seq2SeqPlusMatcher, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        self.dim = self.embeddings.embedding_dim
        self.matchers = nn.ModuleDict()
        self.left_encoders = nn.ModuleDict()
        self.right_encoders = nn.ModuleDict()
        l_att = nn.TransformerEncoderLayer(d_model=self.dim, nhead=5)
        r_att = nn.TransformerEncoderLayer(d_model=self.dim, nhead=5)
        for col in columns:
            self.matchers[col] = Seq2SeqDifference(self.dim, dense_hidden)
            self.left_encoders[col] = l_att
            self.right_encoders[col] = r_att
        self.columns = columns
        self.dense = nn.Sequential(
            nn.Linear(dense_hidden * len(columns), dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 2),
        )
    def forward(self, batch):
        features = []
        self.left_vectors = []
        self.right_vectors = []

        for col in self.columns:
            left = batch['left'][col]
            right = batch['right'][col]
            l_lengths, l_tokens = left['size'], left['seq']
            r_lengths, r_tokens = right['size'], right['seq']
            l_vectors = self.embeddings(l_tokens)
            r_vectors = self.embeddings(r_tokens)
            l_vectors = self.left_encoders[col](l_vectors, left['attention_mask'])#, ~left['key_padding_mask'])
            r_vectors = self.right_encoders[col](r_vectors, right['attention_mask'])#, ~right['key_padding_mask'])
            if True: # matching layer
                features.append(self.matchers[col](l_lengths, l_vectors, r_lengths, r_vectors))
            else:
                l_vectors = (l_vectors * left['key_padding_mask'].T.unsqueeze(2)).sum(0) / left['key_padding_mask'].sum(1).unsqueeze(1)
                r_vectors = (r_vectors * right['key_padding_mask'].T.unsqueeze(2)).sum(0) / right['key_padding_mask'].sum(1).unsqueeze(1)
                features.append((l_vectors * r_vectors).sum(1).unsqueeze(1))
            self.left_vectors.append(l_vectors)
            self.right_vectors.append(r_vectors)
        self.features = features
        return self.dense(torch.cat(features, 1))



import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

class PlusMatcher(EntityMatchingModule, pl.LightningModule):
    def __init__(self, word_embeddings, columns, dense_hidden=60, max_seq_len=128):
        super(PlusMatcher, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        mask = torch.tril(torch.ones((max_seq_len + 1, max_seq_len)))
        mask = mask - torch.eye(max_seq_len + 1, max_seq_len)
        self.register_buffer('src_mask', mask.bool())
        self.dim = self.embeddings.embedding_dim
        self.matchers = nn.ModuleDict()
        self.left_encoders = nn.ModuleDict()
        self.right_encoders = nn.ModuleDict()
        l_att = nn.TransformerEncoderLayer(d_model=self.dim, nhead=5, dim_feedforward=self.dim * 4)
        l_att = nn.TransformerEncoder(l_att, 1)
        r_att = nn.TransformerEncoderLayer(d_model=self.dim, nhead=5, dim_feedforward=self.dim * 4)
        r_att = nn.TransformerEncoder(r_att, 1)
        for col in columns:
            self.matchers[col] = Seq2SeqDifference(self.dim, dense_hidden)
            self.left_encoders[col] = l_att
            self.right_encoders[col] = r_att
        self.columns = columns
        self.dense = nn.Sequential(
            nn.Linear(dense_hidden * len(columns), dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 2),
        )

        self.train_prec = Precision()
        self.train_rec = Recall()
        self.train_f1 = F1()

        self.valid_prec = Precision()
        self.valid_rec = Recall()
        self.valid_f1 = F1()

    def forward(self, batch):
        features = []
        self.left_vectors = []
        self.right_vectors = []

        for col in self.columns:
            left = batch['left'][col]
            right = batch['right'][col]
            l_lengths, l_tokens = left['size'], left['seq']
            r_lengths, r_tokens = right['size'], right['seq']
            l_vectors = self.embeddings(l_tokens)
            r_vectors = self.embeddings(r_tokens)
            # print(l_vectors.shape, l_lengths.shape)
            # print(self.src_mask[l_lengths, :l_vectors.shape[0]].T.shape)
            # print(l_vectors.shape)
            l_vectors = self.left_encoders[col](l_vectors, src_key_padding_mask=~self.src_mask[l_lengths, :l_vectors.shape[0]])
            r_vectors = self.right_encoders[col](r_vectors, src_key_padding_mask=~self.src_mask[r_lengths, :r_vectors.shape[0]])
            # print(l_vectors.shape)
            # r_vectors = self.right_encoders[col](r_vectors)#, ~right['key_padding_mask'])
            if True: # matching layer
                features.append(self.matchers[col](l_lengths, l_vectors, r_lengths, r_vectors))
            else:
                l_vectors = (l_vectors * left['key_padding_mask'].T.unsqueeze(2)).sum(0) / left['key_padding_mask'].sum(1).unsqueeze(1)
                r_vectors = (r_vectors * right['key_padding_mask'].T.unsqueeze(2)).sum(0) / right['key_padding_mask'].sum(1).unsqueeze(1)
                features.append((l_vectors * r_vectors).sum(1).unsqueeze(1))
            self.left_vectors.append(l_vectors)
            self.right_vectors.append(r_vectors)
        self.features = features
        return self.dense(torch.cat(features, 1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    

def main():
    task_name = 'Fodors-Zagats'
    cache = CacheDecoreator('data/cache', task_name)
    @cache.category('entity_pair_dataset', force=False)
    def DeepMatcherDataset(base_dir):
        from daem.data.dataset import EntityPairDataset
        from pathlib import Path
        import fastText as fasttext
        base_dir = Path(base_dir)
        train = EntityPairDataset(base_dir / 'train.csv')
        valid = EntityPairDataset(base_dir / 'validation.csv')
        test = EntityPairDataset(base_dir / 'test.csv')
        tokens = [c.get_tokens() for c in [train, valid, test]]
        tokens = np.unique(np.concatenate(tokens))
        print('Token set size:', tokens.shape[0])
        from daem.data.vocabulary_table import VocabularyTable
        word_embedding_model = fasttext.load_model('wiki.en.bin')
        vocab = VocabularyTable(tokens, word_embedding_model)
        train.inject_vocab(vocab)
        valid.inject_vocab(vocab)
        test.inject_vocab(vocab)
        return train, valid, test, vocab
    # data
    from daem.data.collate import entity_pad_collate
    train, valid, test, vocab = DeepMatcherDataset('datasets/Structured/' + task_name)
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(train, batch_size=256, collate_fn=entity_pad_collate, num_workers=8)
    val_loader = DataLoader(valid, batch_size=256, collate_fn=entity_pad_collate, num_workers=8)
    test_loader  = DataLoader(test, batch_size=256, collate_fn=entity_pad_collate, num_workers=8)

    # model
    model = PlusMatcher(vocab.vectors, train.columns)

    # training
    trainer = pl.Trainer(gpus=1, max_epochs=50, progress_bar_refresh_rate=310)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
    
