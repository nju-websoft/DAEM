import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class PaddedPairDatasetContainer():
    def __init__(self, vocab, columns, left=None, right=None):
        self.vocab = vocab
        self.columns = columns
        if left is not None and right is not None:
            self.left_entites = {}
            self.right_entites = {}
            self._preprocess_entities(left, self.left_entites)
            self._preprocess_entities(right, self.right_entites)
    
    def _seq2tensor(self, seq):
        return [self.vocab.stoi[t] for t in seq] if len(seq) > 0 else [0]

    def _example2tensor(self, example, part, col):
        tensor = [self.vocab.stoi[t] for t in getattr(example, part + '_' + col)]
        while len(tensor) < 1:
            tensor.append(1)
        return tensor
    
    def _preprocess_entities(self, source, target):
        for _, row in source.iterrows():
            ent = dict((col, None) for col in self.columns)
            for col in self.columns:
                seq = torch.LongTensor(self._seq2tensor(row[col]))
                ent[col] = seq
            target[row['id']] = ent
                
    def tuples_iterator(self, where, entity_ids, batch_size=120, device='cpu', mask=None):
        if mask is None:
            mask = set()
        assert type(mask) == set
        assert where in ('left', 'right')
        entities = self.right_entites if where == 'right' else self.left_entites

        for i in range(0, entity_ids.shape[0], batch_size):
            begin = i
            end = min(entity_ids.shape[0], i + batch_size)
            batch = dict()

            for col in self.columns:
                if col not in mask:
                    seq = [entities[idx][col] for idx in entity_ids[begin:end]]
                else:
                    seq = [torch.LongTensor([0]) for _ in range(begin, end)]
                batch[col] = dict()
                batch[col]['seq'] = pad_sequence(seq).to(device)
                batch[col]['size'] = torch.LongTensor([len(t) for t in seq]).to(device)
            yield batch

    def split_tuples(self, where, entity_ids, batch_size=120, device='cpu', mask=None):
        return tuple(self.tuples_iterator(where, entity_ids, batch_size, device, mask))

    def split_pairs(self, examples, batch_size=120, device='cpu', mask=None):
        if mask is None:
            mask = set()
        assert type(mask) == set
        output = []
        label_tensor = torch.LongTensor([e.label for e in examples]).to(device)

        for i in range(0, label_tensor.shape[0], batch_size):
            begin = i
            end = min(label_tensor.shape[0], i + batch_size)
            batch = {'left': {}, 'right': {}}
            label = label_tensor[begin:end]

            for col in self.columns:
                for where in ('left', 'right'):
                    if col not in mask:
                        seq = [self._example2tensor(e, where, col) for e in examples[begin:end]]
                    else:
                        seq = [[0] for _ in range(begin, end)]
                    batch[where][col] = dict()
                    batch[where][col]['seq'] = pad_sequence([torch.LongTensor(t) for t in seq]).to(device)
                    batch[where][col]['size'] = torch.LongTensor([len(t) for t in seq]).to(device)
            output.append((batch, label))
        return output