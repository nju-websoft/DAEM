import pandas as pd
import nltk
import os
import sys
import numpy as np
sys.path.append('./')
sys.path.append('./external')
sys.path.append('./external/DeepMatcher')

from daem.impl import inter_ffe_injection
from daem.models.batch_generator import PaddedPairDatasetContainer
try:
    import fasttext as fastText
except ImportError:
    import fastText
import os
import deepmatcher as dm
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import math

cache_dir = './data/cache/'

from daem.impl import inter_ffe_injection
from daem.models.batch_generator import PaddedPairDatasetContainer
from daem.models.pure import PureMatcher as Seq2SeqPlusMatcher

def generate_train_examples(left, right, columns, npos=20, nneg=4):
    examples = []
    for x, x_prefix, y, y_prefix in [(left, 'left_', right, 'right_'), (right, 'right_', left, 'left_')]:
        x = x[columns].sample(npos).rename(columns=dict((c, x_prefix + c) for c in columns))
        x = pd.concat([x] * nneg, ignore_index=True)
        y = y[columns].sample(npos * nneg).rename(columns=dict((c, y_prefix + c) for c in columns))
        y.index = x.index
        examples.append(pd.merge(x, y, left_index=True, right_index=True, how='inner').assign(label=0))
        for col in columns:
            x[y_prefix + col] = x[x_prefix + col].apply(lambda l: [x for x in l if random.random() > 0.1])
        examples.append(x.assign(label = 1))
    examples = pd.concat(examples, sort=False)
    examples = list(examples.itertuples())
    idx = np.random.permutation(len(examples))
    examples = [examples[i] for i in idx]
    return examples

def retrain_network(net, loop_cnt, batches, test_batches, weight=None, realtime=False):
    if weight is None:
        weight = torch.Tensor([1, 5])
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-5)
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
                # p1 = p0
                if epoch == loop_cnt - 1 or realtime:
                    p2 = performance_batch(net, test_batches)
                else:
                    p2 = p0
                pbar.update(len(test_batches))
                template = 'epoch=%d loss=%3.3f  train=%.3f,%.3f,%.3f  test=%.3f,%.3f,%.3f'
                pbar.set_postfix_str(template % tuple([epoch, tr_loss] + list(p0) + list(p2)))
    return p2

def inter_ffe_full(train, valid, test, columns):
    for part in [train, valid, test]:
        for i in range(0, len(part.examples)):
            for where in ['left_', 'right_']:
                full = []
                for col in columns:
                    full += getattr(part.examples[i], where + col)
                setattr(part.examples[i], where + 'full', full)

batch_size = 16
task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'iTunes-Amazon', 'Walmart-Amazon', 'Beer', 'Fodors-Zagats']
# task_names = [task_names[-1]]
base_dir =  './datasets/Structured/'

import sys
# task_names = [sys.argv[-1]]
loop_cnt = 50

for task_name in task_names:
    print(task_name)
    results = []
    train, validation, test = dm.data.process(path=base_dir + task_name, train='train.csv', validation='validation.csv', test='test.csv', pca=False)
    #inter_ffe_injection(task_name, train, validation, test)
    inter_ffe_full(train, validation, test, train.canonical_text_fields)
    vocab = train.vocabs[train.all_text_fields[0]]
    # columns = train.canonical_text_fields + ['full']
    columns = train.canonical_text_fields + ['full']
    columns = ['full']
    if 'type' in columns:
        columns.remove('type')
    container = PaddedPairDatasetContainer(vocab, columns)
    # batches = container.split_pairs(train.examples, batch_size=batch_size, device='cuda')
    # valid_batches = container.split_pairs(validation.examples, batch_size=128, device='cuda')
    # test_batches = container.split_pairs(test.examples, batch_size=128, device='cuda')

    from daem.helpers import performance_raw, performance_batch, pretty_print_example, pretty_print_examples
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # hidden_size = int(arguments['--hidden_size'] or 60)
    device = 'cuda'
    we = vocab.vectors
    we = we / torch.norm(we, 2, 1).unsqueeze(1)
    from daem.models.pure_plus import PurePlusMatcher as Seq2SeqPlusMatcher
    net = Seq2SeqPlusMatcher(we, columns, fixed_matcher=True).to(device)
    start_cache_name = './data/tmp/da-pretrain-%s-start.pt' % task_name
    torch.save(net.state_dict(), start_cache_name)

    examples = sum([x.examples for x in [train, validation, test]], [])
    df = pd.DataFrame([e.__dict__ for e in examples])
    left = df[['left_' + col for col in columns]]
    right = df[['right_' + col for col in columns]]
    left = left.rename(columns=dict(('left_' + k, k) for k in columns))
    right = left.rename(columns=dict(('right_' + k, k) for k in columns))
    for col in columns:
        left[col] = left[col].apply(tuple)
        right[col] = right[col].apply(tuple)
    left.drop_duplicates()
    right.drop_duplicates()
    # from daem.active.seed import generate_train_examples
    pretrain = generate_train_examples(left, right, columns)
    # break
    batches = container.split_pairs(pretrain, batch_size=50, device='cuda')
    test_batches = container.split_pairs(examples, batch_size=128, device='cuda')
    retrain_network(net, 50, batches, test_batches, realtime=True)
    cache_name = './data/tmp/da-pretrain-%s.pt' % task_name
    torch.save(net.state_dict(), cache_name)

    for sample_size in [int(sys.argv[-3])]:
    # for sample_size in [400]:
        for rep in range(0, 10):
            import numpy as np
            idx = np.random.permutation(np.arange(0, len(examples)))
            train = [examples[i] for i in idx[:sample_size]]
            test = [examples[i] for i in idx[sample_size:]]

            batches = container.split_pairs(train, batch_size=10, device='cuda')
            test_batches = container.split_pairs(test, batch_size=128, device='cuda')

            net = Seq2SeqPlusMatcher(we, columns, fixed_matcher=True).to(device)
            net.load_state_dict(torch.load(start_cache_name))
            f = retrain_network(net, loop_cnt, batches, test_batches, realtime=False)
            results.append([task_name, sample_size, rep, '-da'] + list(f))

            net = Seq2SeqPlusMatcher(we, columns, fixed_matcher=True).to(device)
            net.load_state_dict(torch.load(cache_name))
            f = retrain_network(net, loop_cnt, batches, test_batches, realtime=False)
            results.append([task_name, sample_size, rep, '+da'] + list(f))
            pd.to_pickle(results, '../da-%s.log' % task_name)