import pandas as pd
import nltk
import os
import sys
sys.path.append('../')
sys.path.append('../external')

cache_dir = '../data/cache/'
base_dir = '../datasets/Structured-raw'
# task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'iTunes-Amazon', 'Walmart-Amazon']
task_names = ['DBLP-GoogleScholar']
task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'iTunes-Amazon', 'Walmart-Amazon', 'Fodors-Zagats', 'Beer']
task_names = ['Walmart-Amazon']

from daem.models.batch_generator import PaddedPairDatasetContainer
from daem.models.pure import PureMatcher as Seq2SeqPlusMatcher
from daem.active.seed import generate_train_examples

def retrain_network(net, loop_cnt, batches, test_batches, weight=None):
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
                p1 = p0
                if epoch == loop_cnt - 1:
                    p2 = performance_batch(net, test_batches)
                else:
                    p2 = p0
                pbar.update(len(test_batches))
                template = 'epoch=%d loss=%3.3f  train=%.3f,%.3f,%.3f  valid=%.3f,%.3f,%.3f  test=%.3f,%.3f,%.3f'
                pbar.set_postfix_str(template % tuple([epoch, tr_loss] + list(p0) + list(p1) + list(p2)))
    return p2

if False:
    import fastText as fastText
    from daem.utils.vocabulary_table import VocabularyTable
    word_embedding_model = fastText.load_model('../wiki.en.bin')
    for task_name in task_names:
        left = pd.read_csv('{}/{}/tableA.csv'.format(base_dir, task_name))
        right = pd.read_csv('{}/{}/tableB.csv'.format(base_dir, task_name))
        gold = pd.concat([pd.read_csv('{}/{}/{}.csv'.format(base_dir, task_name, part)) for part in ['train', 'test', 'valid']])
        gold = gold.rename(columns={'ltable_id': 'id1', 'rtable_id': 'id2'})
        assert (left.columns == right.columns).all()
        columns = list(left.columns)[1:]
        print(task_name, columns, len(left), len(right), len(gold))
        for col in columns:
            for where in [left, right]:
                where[col] = where[col].fillna('').apply(str).apply(nltk.word_tokenize)
        vocabs = VocabularyTable(pd.concat([left, right]), columns, word_embedding_model)
        pd.to_pickle((left, right, gold, columns, vocabs), '%s/dataset-%s.pkl' % (cache_dir, task_name))
else:
    import numpy as np
    for task_name in task_names:
        left, right, gold, columns, vocabs = pd.read_pickle('%s/dataset-%s.pkl' % (cache_dir, task_name))
        print(task_name, columns, len(left), len(right), len(gold), gold['label'].mean())
#         break
#         
        gold_pairs = gold.rename(columns=dict(id1='left_id', id2='right_id'))
        for prefix, where in [('left_', left), ('right_', right)]:
            gold_pairs = pd.merge(gold_pairs, where.rename(columns=dict((c, prefix + c) for c in ['id'] + columns)), on=prefix + 'id')
        gold_pairs = list(gold_pairs.itertuples())
        if 'type' in columns:
            columns.remove('type')
        if 'price' in columns:
            columns.remove('price')
        
        examples = generate_train_examples(left, right, columns)
        
        container = PaddedPairDatasetContainer(vocabs, columns, left, right)
        batches = container.split_pairs(examples, 128, 'cuda')
        test_batches = container.split_pairs(gold_pairs, 128, 'cuda')
        
        
        from daem.helpers import performance_raw, performance_batch, predict_batch
        import torch
        import torch.nn as nn
        import torch.optim as optim
        # hidden_size = int(arguments['--hidden_size'] or 60)
        device = 'cuda'
        vocabs.vectors = vocabs.vectors / np.sqrt((vocabs.vectors * vocabs.vectors).sum(1)).reshape((-1, 1))
        net = Seq2SeqPlusMatcher(vocabs.vectors, columns).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5])).to(device)
        optimizer = optim.Adam(net.parameters(), lr=2e-5)

        # optimizer, scheduler = build_optimizer(net, 15, 2e-5, 1e-08, 5, 0.0)

        loop_cnt = 15

        from tqdm.notebook import tqdm

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
        #             print(loss)
                    loss.backward()
                    optimizer.step()
        #             scheduler.step()
                    pbar.update(2)
                with torch.no_grad():
                    net.eval()
                    p0 = performance_batch(net, batches)
                    pbar.update(len(batches))
                    p1 = p0
        #             p1 = performance_batch(net, valid_batches)
#                     pbar.update(len(valid_batches))
                    p2 = performance_batch(net, test_batches)
#                     p2 = p0
                    pbar.update(len(test_batches))
                    template = 'epoch=%d loss=%3.3f  train=%.3f,%.3f,%.3f  valid=%.3f,%.3f,%.3f  test=%.3f,%.3f,%.3f'
                    # torch.cuda.empty_cache()
                    pbar.set_postfix_str(template % tuple([epoch, tr_loss] + list(p0) + list(p1) + list(p2)))
#                     print(template % tuple([epoch, tr_loss] + list(p0) + list(p1) + list(p2)))
        df = pd.DataFrame(dict(p=predict_batch(net, test_batches), y=[t.label for t in gold_pairs]))
        # print(pd.concat([df.sort_values('p').head(20), df.sort_values('p').tail(20)])['y'].mean())
        cache_name = '../data/tmp/pretrain-%s.pt' % task_name
        torch.save(net.state_dict(), cache_name)

from itertools import combinations
masked_batches = []
container = PaddedPairDatasetContainer(vocabs, columns)
for i, comb in enumerate(combinations(columns, 2)):
    masked_batches.append(container.split_pairs(gold_pairs, 128, 'cuda', set(comb)))

from daem.helpers import predict_batch
import cupy as cp
picked = np.zeros(len(gold_pairs), dtype=np.bool)
net.load_state_dict(torch.load('/dev/shm/pretrain.pt'))
from daem.online_matcher.partial_order_matcher import MaximumInferenceClassifier, MaximumExpectedMatchClassifier
for i in range(0, 20):
    sv = []
    for batches in masked_batches:
        sv.append(predict_batch(net, batches))
#     sv.append(predict_batch(net, test_batches))
    sv = np.vstack(sv).T
    y = np.array([t.label for t in gold_pairs])
    mic = MaximumExpectedMatchClassifier(sv, predict_batch(net, test_batches))
    margin = []
    for next_question in np.arange(0, len(picked))[picked]:
        mic.update_model(np.array([next_question]), np.array([y[next_question] * 2 - 1]))
    for loop_value in range(0, 20):
        next_question = mic.acquisition_function()
        if next_question is None:
            break
        next_question = int(next_question[0])
        margin.append(next_question)
        mic.update_model([next_question], [y[next_question] * 2 - 1])
    margin = np.array(margin)
    picked[margin] = True
#     break
    if False:
        gold_pairs_fake = pd.DataFrame(list(gold_pairs))
        gold_pairs_fake['label'] = cp.asnumpy(mic.predicted)
        gold_pairs_fake = gold_pairs_fake[gold_pairs_fake['label'] != 0]
        gold_pairs_fake['label'] = (gold_pairs_fake['label'] + 1) // 2
        print(picked.sum(), len(gold_pairs_fake))
        batches = container.split_pairs(list(gold_pairs_fake.itertuples()), 16, 'cuda')
    else:
        batches = container.split_pairs([gold_pairs[i] for i in range(0, len(picked)) if picked[i]], 16, 'cuda')
#     net.load_state_dict(torch.load('/dev/shm/pretrain.pt'))
    retrain_network(net, 30, batches, test_batches, weight=torch.Tensor([1, int(1 / gold_pairs_fake['label'].mean())]))