from daem.impl import inter_ffe_injection
from daem.models.batch_generator import PaddedPairDatasetContainer
import fastText
import os

import pandas as pd
import nltk
import os
from pathlib import Path

cache_dir = './data/cache'
base_dir = Path('./') / 'datasets/Structured-raw'
task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'Fodors-Zagats', 'Walmart-Amazon']

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
from daem.models.functional import packed_sequence_mean
from better_lstm import LSTM
from daem.models.nn.sentence import SentenceConv


class DAALMEncoder(nn.Module):
    def __init__(self, word_embeddings, columns):
        super(DAALMEncoder, self).__init__()
        self.columns = columns
        # token embeddings
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        self.conv = SentenceConv(1, dim=self.embeddings.embedding_dim, pooling='max')
        
        mask_matrix = pad_sequence([torch.ones(k) for k in range(0, 256 + 1)]).T
        self.register_buffer('mask_matrix', mask_matrix)
        self.dense = nn.Sequential(
            nn.Linear(len(columns) * 100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.Sigmoid()
        )


    def forward(self, X):
        representations = []

        for column in self.columns:
            seq = X[column]['seq']
            size = X[column]['size']
            seq = self.embeddings(seq)
            representations.append(self.conv(seq, self.mask_matrix[size, :size.max()].T))
        return torch.cat(representations, 1)

import warnings
warnings.filterwarnings("ignore")


def blocking_kfold(true_matches):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(true_matches.index):
        yield true_matches.iloc[train_index], true_matches.iloc[test_index]

def negative_sampling(matches, destination, src_name, dst_name, sample_size=5):
    for src_id, dst_ids in matches.groupby(by='left_id'):
        dst_ids = dst_ids[dst_name].values
        neg_ids = list(np.random.choice(destination, sample_size + len(dst_ids), replace=False))
        for pos in dst_ids:
            if pos in neg_ids:
                neg_ids.remove(pos)
        neg_ids = neg_ids[:sample_size]
        yield pd.DataFrame({
            src_name: [src_id] * (sample_size + len(dst_ids)), 
            dst_name: list(dst_ids) + neg_ids,
            'label': [1] * len(dst_ids) + [0] * sample_size
        })
        
if True:
    import fastText
    from daem.utils.vocabulary_table import VocabularyTable
    word_embedding_model = fastText.load_model(os.environ['ACTIVEER_DIR'] + '/wiki.en.bin')
    for task_name in task_names:
        task_dir = base_dir / task_name
        left = pd.read_csv(task_dir / "tableA.csv")
        right = pd.read_csv(task_dir / "tableB.csv")
        gold = pd.concat([pd.read_csv(task_dir / (part + '.csv')) for part in ['train', 'test', 'valid']])
        gold = gold.rename(columns={'ltable_id': 'id1', 'rtable_id': 'id2'})
        assert (left.columns == right.columns).all()
        columns = list(left.columns)[1:]
        print(task_name, columns, len(left), len(right), len(gold))
        for col in columns:
            for where in [left, right]:
                where[col] = where[col].fillna('').apply(str).apply(nltk.word_tokenize)
        vocabs = VocabularyTable(pd.concat([left, right]), columns, word_embedding_model)
        pd.to_pickle((left, right, gold, columns, vocabs), '%s/dataset-%s.pkl' % (cache_dir, task_name))
if True:
    import numpy as np
    for task_name in task_names:
        print(task_name)
        left, right, gold, columns, vocabs = pd.read_pickle('%s/dataset-%s.pkl' % (cache_dir, task_name))

        if len(left) < len(right):
            left, right = right, left
            gold = gold.rename(columns=dict(id1='id2', id2='idt')).rename(columns=dict(idt='id1'))
        print(len(left), len(right))
        columns = columns[:1]
        for useless_property in ['type', 'phone', 'class']:
            if useless_property in columns:
                columns.remove(useless_property)
        print(task_name, columns, len(left), len(right), len(gold), gold['label'].mean())
        gold_pairs = gold.rename(columns=dict(id1='left_id', id2='right_id'))
        true_matches = gold_pairs[gold_pairs['label'] == 1][['left_id', 'right_id']]
        container = PaddedPairDatasetContainer(vocabs, columns, left, right)
        
        from daem.helpers import performance_raw, performance_batch, pretty_print_example, pretty_print_examples
        import torch
        import torch.nn as nn
        import torch.optim as optim
        device = 'cuda'
        we = torch.Tensor(vocabs.vectors)
        we = we / torch.norm(we, 2, 1).unsqueeze(1)
        from daem.models.pure_plus import PurePlusMatcher as Seq2SeqPlusMatcher
        
        left_batches = container.split_tuples('left', left['id'].values, batch_size=1, device='cuda')
        # right_batches = container.split_tuples('right', right['id'].values, batch_size=1, device='cuda')
        for train, test in blocking_kfold(true_matches):
            entity_encoder = DAALMEncoder(we, columns).to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(entity_encoder.parameters(), lr=2e-3)
            for epoch in range(0, 15):
                tr_loss = 0.0
                for batches in negative_sampling(true_matches, right['id'].values, 'left_id', 'right_id'):
                    optimizer.zero_grad()
                    left_rep = entity_encoder(left_batches[batches.values[0][0]])
                    right_batch = container.split_tuples('right', batches.values[:, 1], batch_size=len(batches), device=device)[0]
                    right_rep = entity_encoder(right_batch)
                    batch_prob = torch.softmax(torch.cosine_similarity(left_rep, right_rep), 0).unsqueeze(0)
                    batch_label = torch.LongTensor(batches.values[:, 2]).to(device)
                    loss = criterion(batch_prob, torch.LongTensor([0]).to(device))
#                     loss = torch.sum(batch_prob * batch_label) - torch.sum(batch_prob * (1 - batch_label))
                    loss.backward()
                    optimizer.step()
                    del right_batch
                    tr_loss += loss.item()
            with torch.no_grad():
                k = int(np.sqrt([len(left), len(right), 1e6]).max())
                left_reps, right_reps = [], []
                for left_ent in left_batches:
                    left_reps.append(entity_encoder(left_ent).detach().to('cpu').numpy())
                left_reps = np.vstack(left_reps)
                left_reps = left_reps / np.linalg.norm(left_reps, 2, 1, True)
                for right_ent in container.split_tuples('right', right['id'].values, batch_size=1, device='cpu'):
                    for col in columns:
                        right_ent[col]['seq'] = right_ent[col]['seq'].to('cuda')
                        right_ent[col]['size'] = right_ent[col]['size'].to('cuda')
                    right_reps.append(entity_encoder(right_ent).detach().to('cpu').numpy())
                right_reps = np.vstack(right_reps)
                right_reps = right_reps / np.linalg.norm(right_reps, 2, 1, True)

                scores = np.matmul(left_reps, right_reps.T)

                pairs = []
                for i in range(0, len(left)):
                    if np.sum(scores[i, :] >= 0.8) < k:
                        right_ids = np.arange(0, len(right))[scores[i, :] >= 0.8]
                    else:
                        right_ids = np.argpartition(-scores[i, :], k)[:k]
                    if len(right_ids) > 0:
                        left_ids = np.array([i] * len(right_ids))
                        pairs.append((left_ids, right_ids))
                pairs = pd.DataFrame({
                    'left_id': np.concatenate([v for v, _ in pairs]),
                    'right_id': np.concatenate([v for _, v in pairs])
                })
                print('task=%s,recall=%.3f' % (task_name, len(pd.merge(pairs, true_matches)) / len(true_matches)), len(pairs))
