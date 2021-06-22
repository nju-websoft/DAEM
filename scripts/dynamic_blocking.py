from daem.impl import inter_ffe_injection
from daem.models.batch_generator import PaddedPairDatasetContainer
import fastText
import os
import time
import pandas as pd
import nltk
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
from daem.models.functional import packed_sequence_mean
from better_lstm import LSTM
from daem.models.nn.sentence import SentenceConv
from daem.models.blocker import DAALMEncoder, static_blocking
from daem.helpers import performance_raw, performance_batch, pretty_print_example, pretty_print_examples
from daem.models.pure_plus import PurePlusMatcher as Seq2SeqPlusMatcher
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
device = 'cuda'

cache_dir = Path('.') / 'cache'
base_dir = Path('.') / 'datasets/Structured-raw'
task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'Fodors-Zagats', 'Walmart-Amazon']
task_name = task_names[-1]

import warnings
warnings.filterwarnings("ignore")


def assign_attributes(pairs, left, right, columns):
    for prefix, where in [('left_', left), ('right_', right)]:
        pairs = pd.merge(pairs, where.rename(columns=dict((c, prefix + c) for c in ['id'] + columns)), on=prefix + 'id')
    return pairs


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

def retrain_network(net, loop_cnt, batches, test_batches, weight=None):
    if weight is None:
        weight = torch.Tensor([1, 20])
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    with tqdm(range(0, loop_cnt * (len(batches) * 3 + len(test_batches)))) as pbar:
        best_validation = 0.0
        for epoch in range(0, loop_cnt):
            net.train()
            tr_loss = 0
            for batch, label in batches:
                optimizer.zero_grad()
                output = net(batch)
                # cross entropy loss
                loss = criterion(output, label)
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(2)
            with torch.no_grad():
                net.eval()
                p0 = performance_batch(net, batches)
                pbar.update(len(batches))
                p1 = p0
                if epoch == loop_cnt - 1:
                    p2 = performance_batch(net, test_batches)
                else:
                    p2 = p0
                pbar.update(len(test_batches))
                template = 'epoch=%d loss=%3.3f  train=%.3f,%.3f,%.3f  valid=%.3f,%.3f,%.3f  test=%.3f,%.3f,%.3f'
                pbar.set_postfix_str(template % tuple([epoch, tr_loss] + list(p0) + list(p1) + list(p2)))
    return p2

def run_active_learning(pairs, net, gold_pairs, cache_name):
    from daem.helpers import predict_batch
    pool = pd.merge(pairs, gold_pairs[['left_id', 'right_id', 'label']], how='left').fillna(0.0)
    pool = assign_attributes(pool, left, right, columns)
    pool = list(pool.itertuples())
    picked = np.zeros(len(pool), dtype=np.bool)
    pool_batches = container.split_pairs(pool, 16, 'cuda')

    net.load_state_dict(torch.load(cache_name))
    results = []
    from tqdm.notebook import tqdm
    for i in tqdm(range(0, 50)):
        score = predict_batch(net, pool_batches)
        order = np.argsort(score)
        neg_margin = order[(score[order] <= 0.5) & ~picked][-8:]
        pos_margin = order[(score[order] > 0.5) & ~picked][:12]
        margin = np.concatenate([neg_margin, pos_margin])
        picked[margin] = True
        acc = np.array([pool[i].label >= 0.5 for i in pos_margin] + [pool[i].label < 0.5 for i in neg_margin]).mean()

        batches = container.split_pairs([pool[i] for i in range(0, len(picked)) if picked[i]], 16, 'cuda')
        net.load_state_dict(torch.load(cache_name))
        f1 = retrain_network(net, 25, batches, pool_batches)[2]
        results.append(((i * 20), f1, acc))
        if len(results) >= 3 and results[-3][2] == results[-2][2] and results[-2][2] == results[-1][2] and results[-1][1] >= 0.8:
            break
    return net, pool_batches, results
# container = PaddedPairDatasetContainer(vocabs, columns, left, right
def generate_fake_training_data(net, test_batches, pairs):
    score = predict_batch(net, test_batches)
    train = pd.DataFrame([[p.left_id, p.right_id] for p in pairs.itertuples()], columns=['left_id', 'right_id'])
    train['score'] = score
    train['label'] = float('nan')
    train.loc[train['score'] >= 0.5, 'label'] = 1
    train.loc[train['score'] < 0.5, 'label'] = 0
    train = train[train['label'].notna()].copy()
    train['label'] = train['label'].astype(int)
    train = assign_attributes(train, left, right, columns)
    return train

def train_blocker(encoder, batches, loop_cnt=25):
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(entity_encoder.parameters(), lr=2e-3)
    for epoch in range(0, loop_cnt):
        tr_loss = 0.0
        for batch, label in batches:
            optimizer.zero_grad()
            left_rep = entity_encoder(batch['left'])
            right_rep = entity_encoder(batch['right'])
            batch_prob = 0.5 + 0.49 * torch.cosine_similarity(left_rep, right_rep)
            loss = criterion(batch_prob, label.float())
            loss.backward()
            optimizer.step()
    return encoder

def get_score(encoder, left_batches, right_batches):
    left_reps, right_reps = [], []
    for left_ent in left_batches:
        left_reps.append(encoder(left_ent).detach().to('cpu').numpy())
    left_reps = np.vstack(left_reps)
    left_reps = left_reps / np.linalg.norm(left_reps, 2, 1, True)
    for right_ent in container.split_tuples('right', right['id'].values, batch_size=1, device='cpu'):
        for col in encoder.columns:
            right_ent[col]['seq'] = right_ent[col]['seq'].to('cuda')
            right_ent[col]['size'] = right_ent[col]['size'].to('cuda')
        right_reps.append(encoder(right_ent).detach().to('cpu').numpy())
    right_reps = np.vstack(right_reps)
    right_reps = right_reps / np.linalg.norm(right_reps, 2, 1, True)

    scores = np.matmul(left_reps, right_reps.T)
    return scores

import numpy as np
print(task_name)
left, right, gold, columns, vocabs = pd.read_pickle('%s/dataset-%s.pkl' % (cache_dir, task_name))
left = pd.merge(left, gold[['id1']].drop_duplicates().rename(columns={'id1': 'id'}))
right = pd.merge(right, gold[['id2']].drop_duplicates().rename(columns={'id2': 'id'}))
left['id1'] = np.arange(0, len(left))
gold = pd.merge(gold.rename(columns={'id1': 'id'}), left[['id', 'id1']])[['id1', 'id2', 'label']]
right['id2'] = np.arange(0, len(right))
gold = pd.merge(gold.rename(columns={'id2': 'id'}), right[['id', 'id2']])[['id1', 'id2', 'label']]
left['id'] = left['id1']
right['id'] = right['id2']
# left, right = right, left
# gold = gold.rename(columns={'id1': 'id3'}).rename(columns={'id3': 'id2', 'id2': 'id1'})
if 'type' in columns:
    columns.remove('type')
len(left), len(right), gold['label'].sum()
from daem.impl import inter_ffe_injection_entity
inter_ffe_injection_entity(task_name, [left, right])


def inter_ffe_full(train, parts, columns):
    for part in parts:
        full_entries = []
        for i in range(0, len(part)):
            full = []
            for col in columns:
                full += part.loc[i, col]
            full_entries.append(full)
        part['full'] = full_entries
# inter_ffe_full(task_name, [left, right], columns)

print(task_name, columns, len(left), len(right), len(gold), gold['label'].mean())
gold_pairs = gold.rename(columns=dict(id1='left_id', id2='right_id'))
true_matches = gold_pairs[gold_pairs['label'] == 1][['left_id', 'right_id', 'label']]

container = PaddedPairDatasetContainer(vocabs, columns, left, right)
we = torch.Tensor(vocabs.vectors)
we = we / torch.norm(we, 2, 1).unsqueeze(1)

# initialize blocker
from daem.active.seed import generate_train_examples
examples = generate_train_examples(left, right, columns, 40, 4)
container = PaddedPairDatasetContainer(vocabs, columns, left, right)
batches = container.split_pairs(examples, 128, 'cuda')

entity_encoder = DAALMEncoder(we, [columns[0]]).to(device)
train_blocker(entity_encoder, batches, 50)

# static blocking
pairs = static_blocking(entity_encoder, container, columns, left, right, 0.8)
rec, pe = len(pd.merge(pairs, true_matches)) / len(true_matches), len(pairs)
pre = len(pd.merge(pairs, true_matches)) / len(pairs)
print(rec, pre)

# initialize matcher
pairs = pd.merge(pairs, true_matches, how='left').fillna(0.0)
gold_pairs = assign_attributes(pairs, left, right, columns)
from daem.helpers import performance_raw, performance_batch, predict_batch
import torch
import torch.nn as nn
import torch.optim as optim
test_batches = container.split_pairs(list(gold_pairs.itertuples()), 128, 'cuda')
device = 'cuda'
net = Seq2SeqPlusMatcher(vocabs.vectors, columns).to(device)
retrain_network(net, 25, batches, test_batches, weight=None)
cache_name = '../data/tmp/pretrain-%s.pt' % task_name
torch.save(net.state_dict(), cache_name)

# main loop
last_duplicates = 1
for main_loop_idx in range(0, 10):
    # initial pairs
    left_batches = container.split_tuples('left', left['id'].values, batch_size=128, device='cuda')
    right_batches = container.split_tuples('right', right['id'].values, batch_size=128, device='cuda')
    scores = get_score(entity_encoder, left_batches, right_batches)
    k = 50
    pairs = []
    for i in range(0, len(left)):
        right_ids = np.argpartition(-scores[i, :], k)[:k]
        right_ids = right_ids[np.argsort(-scores[i][right_ids])]
        if len(right_ids) > 0:
            left_ids = np.array([i] * len(right_ids))
            pairs.append((left_ids, right_ids, np.arange(0, len(right_ids))))
    pairs = pd.DataFrame({
        'left_id': np.concatenate([v for v, _, __ in pairs]),
        'right_id': np.concatenate([v for _, v, __ in pairs]),
        'rank': np.concatenate([v for _, __, v in pairs])
    })
    pairs = assign_attributes(pairs, left, right, columns).assign(label=1)
    
    # generate gap
    from daem.helpers import predict_batch
    pair_batches = container.split_pairs(list(pairs.itertuples()), 128, 'cuda')
    pairs['mp'] = predict_batch(net, pair_batches)
    mp = pairs[['left_id', 'right_id', 'rank', 'mp']]
    mpp = mp.copy()
    mpp['rank'] -= 1
    gap = pd.merge(mp, mpp, on=['left_id', 'rank']).sort_values(by=['left_id', 'rank'])
    gap['gap'] = gap['mp_x'] - gap['mp_y']
    gap.loc[gap['gap'] < 0, 'gap'] = 0
    gap = gap[['left_id', 'right_id_x', 'rank', 'gap']]
    gap = gap.rename(columns={'right_id_x': 'right_id'})
    
    size = gap['left_id'].nunique()
    gap_cache = np.zeros((size, gap['rank'].max() * 2))
    gap_size = np.zeros(size, dtype=int)
    gap_left_id = np.zeros(size)
    for i, (left_id, g) in enumerate(gap.groupby(by='left_id')):
        gap_left_id[i] = left_id
        gap_size[i] = len(g)
        gap_cache[i, :len(g)] = g['gap'].values
    
    last_rec = 0.0
    last_pre = 0.0
    for duplicates in range(last_duplicates, 20):
        tau = min(len(left), len(right)) * duplicates
        k = np.ones(size, dtype=np.int) * (tau // size)
        k[:tau % size] += 1
        o = 0.0
#         with tqdm(range(0, tau + size)) as pbar:
        for loop_var in range(0, tau + size):
            new_o = sum(gap_cache[i, min(k[i], gap_size[i] - 1)] for i in range(0, size))
            if new_o - o < 1e-5:
                break
            o = new_o
#             pbar.set_postfix_str(str(o))
#             pbar.update(1)
            inc = np.array([gap_cache[i, min(k[i] + 1, gap_size[i] - 1)] - gap_cache[i, k[i]] for i in range(0, size)])
            inc[k >= gap_size] = -1e6
            dec = np.array([gap_cache[i, max(k[i] - 1, 0)] - gap_cache[i, k[i]] for i in range(0, size)])
            dec[k == 0] = -1e6
            i_inc = np.argmax(inc)
            i_dec = np.argmax(dec)
            if i_inc == i_dec:
                inc[i_dec] = 0.0
                i_inc = np.argmax(inc)
            if inc[i_inc] + dec[i_dec] > 1e-6:
                k[i_inc] += 1
                k[i_dec] -= 1
#                 pbar.update(1)
            else:
                break

        pairs = []
        for i, (left_id, g) in enumerate(gap.groupby(by='left_id')):
            if k[i] > 0:
                pairs.append(([left_id] * min(k[i], len(g)), g['right_id'].values[:k[i]]))
        pairs = pd.DataFrame({
            'left_id': np.concatenate([v for v, _ in pairs]),
            'right_id': np.concatenate([v for _, v in pairs]),
        })
        rec, pe = len(pd.merge(pairs, true_matches)) / len(true_matches), len(pairs)
        pre = len(pd.merge(pairs, true_matches)) / len(pairs)
        if (rec < 0.9 and rec - last_rec < 0.01) or (rec > 0.9 and rec - last_rec < 0.005) or pre < last_pre * 0.1:
            print(rec, pre, last_rec, last_pre, rec - last_rec < 0.01, last_pre * 0.1)
            break
        last_rec, last_pre = rec, pre
    print(last_rec, last_pre)
    last_duplicates = duplicates + 1
    # active learnig
    pairs = pd.merge(pairs, true_matches, how='left').fillna(0.0)
    gold_pairs = assign_attributes(pairs, left, right, columns)
    net, pool_batches, results = run_active_learning(pairs, net, gold_pairs, cache_name)
    print(results[-1])
    # retrain blocker
    from daem.active.seed import generate_train_examples
    # examples = train
    train = generate_fake_training_data(net, pool_batches, gold_pairs)
    batches = container.split_pairs(list(train.itertuples()), 128, 'cuda')
    entity_encoder = DAALMEncoder(we, columns[:1]).to(device)
    train_blocker(entity_encoder, batches, 50)

