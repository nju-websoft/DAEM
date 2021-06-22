import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from daem.models.functional import packed_sequence_mean
from better_lstm import LSTM


def _tokens2tensor(tokens, vocabs):
    tokens = [vocabs.stoi[c] for c in tokens]
    if len(tokens) == 0:
        tokens = [vocabs.stoi['<unk>']]

    return torch.LongTensor(tokens)

class MatchingDatasetContainer():
    def __init__(self, dataset):
        self.vocabs = dataset.vocabs[dataset.all_text_fields[0]]
        self.columns = dataset.canonical_text_fields
    def _tokens2tensor(self, tokens):
        tokens = [self.vocabs.stoi[c] for c in tokens]
        if len(tokens) == 0:
            tokens = [self.vocabs.stoi['<unk>']]
        return torch.LongTensor(tokens)
    def split_and_pack(self, examples, batch_size=120, device='cpu'):
        output = []
        left_tensors = {}
        right_tensors = {}
        label_tensor = torch.LongTensor([e.label for e in examples])
        empty_tensors = {}
        for column in self.columns:
            left_tensors[column] = [(self._tokens2tensor(getattr(e, 'left_' + column))) for e in examples]
            right_tensors[column] = [(self._tokens2tensor(getattr(e, 'right_' + column))) for e in examples]
            empty_tensors[column] = np.array(
                [len(getattr(e, 'left_' + column)) == 0 or len(getattr(e, 'right_' + column)) == 0
                for e in examples])
        for i in range(0, label_tensor.shape[0], batch_size):
            begin = i
            end = min(label_tensor.shape[0], i + batch_size)
            batch = {'left': {}, 'right': {}, 'empty': {}}
            label = torch.LongTensor(label_tensor[begin:end]).to(device)
            for column in self.columns:
                batch['left'][column] = pack_sequence(left_tensors[column][begin:end], enforce_sorted=False).to(device)
                batch['right'][column] = pack_sequence(right_tensors[column][begin:end], enforce_sorted=False).to(device)
                batch['empty'][column] = torch.LongTensor(1 - empty_tensors[column][begin:end]).to(device)
            output.append((batch, label))
        return output


class DeepERLSTM(nn.Module):
    def __init__(self, word_embeddings, columns, composition='lstm', difference='cosine', classification='50-2', dropout=0.05):
        super(DeepERLSTM, self).__init__()
        self.composition = composition
        self.difference = difference
        self.classification = classification
        self.columns = columns
        # token embeddings
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        # token composition layer
        if self.composition == 'lstm':
            self.lstm = LSTM(self.embeddings.embedding_dim, 150, dropoutw=dropout)

        if self.difference == 'cosine':
            self.feature_cnt = len(columns)
        elif self.difference == 'abs_diff':
            self.feature_cnt = len(columns) * 150
        else:
            assert False

        if self.classification == 'linear':
            self.linear = nn.Linear(len(self.columns), 1)
            if self.difference == 'cosine':
                with torch.no_grad():
                    self.linear.weight = torch.nn.Parameter(torch.ones(1, self.feature_cnt), True)
                    self.linear.bias = torch.nn.Parameter(torch.zeros(1), True)
        else:
            sequence = []
            in_dim = self.feature_cnt
            for out_dim in self.classification.split('-'):
                out_dim = int(out_dim)
                sequence += [nn.Linear(in_dim, out_dim), nn.Dropout(dropout), nn.ReLU()]
                in_dim = out_dim
            sequence += [nn.Linear(out_dim, 2)]
            self.dense = nn.Sequential(*sequence)

    def _dense(self, out):
        if hasattr(self, 'dense'):
            return self.dense(out)
        else:
            if self.difference == 'cosine':
                out = F.linear(out, self.linear.weight * self.linear.weight, self.linear.bias)
            else:
                out = self.linear(out)
            return torch.stack([-out, out], 1).view(-1, 2)
    
    def _composite(self, seq):
        if self.composition == 'lstm':
            _, (hidden, _) = self.lstm(PackedSequence(self.embeddings(seq.data), seq.batch_sizes))
            return hidden[-1]
        else:
            return packed_sequence_mean(seq, self.embeddings)

    def forward(self, X):
        similarities = []

        for column in self.columns:
            seq_left = X['left'][column]
            seq_right = X['right'][column]
            if self.difference == 'cosine':
                similarities.append(F.cosine_similarity(
                    self._composite(seq_left)[seq_left.unsorted_indices],
                    self._composite(seq_right)[seq_right.unsorted_indices]) * X['empty'][column])
            else:
                similarities.append(torch.abs(
                    self._composite(seq_left)[seq_left.unsorted_indices] -
                    self._composite(seq_right)[seq_right.unsorted_indices]) * X['empty'][column].unsqueeze(1))
        if self.difference == 'cosine':
            out = torch.stack(similarities, 1)
        else:
            out = torch.cat(similarities, 1)
        return self._dense(out)