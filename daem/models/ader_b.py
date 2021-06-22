from daem.models.nn import SentenceConv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class MatchingDatasetContainer():
    def __init__(self, dataset):
        self.vocabs = dataset.vocabs[dataset.all_text_fields[0]]
        self.columns = dataset.canonical_text_fields

    def _example2tensor(self, example, part, col):
        tensor = [self.vocabs.stoi[t] for t in getattr(example, part + '_' + col)]
        while len(tensor) < 1:
            tensor.append(1)
        return tensor

    def split(self, examples, batch_size=120, device='cpu'):
        output = []
        label_tensor = torch.LongTensor([e.label for e in examples]).to(device)

        for i in range(0, label_tensor.shape[0], batch_size):
            begin = i
            end = min(label_tensor.shape[0], i + batch_size)
            batch = {'left': {}, 'right': {}}
            label = label_tensor[begin:end]

            for col in self.columns:
                for where in ('left', 'right'):
                    seq = [self._example2tensor(e, where, col) for e in examples[begin:end]]
                    batch[where][col] = dict()
                    batch[where][col]['seq'] = pad_sequence([torch.LongTensor(t) for t in seq]).to(device)
                    batch[where][col]['size'] = torch.LongTensor([len(t) for t in seq]).to(device)
                    batch[where][col]['key_padding_mask'] = pad_sequence([torch.ones(len(t)) for t in seq]).T.bool().to(device)
                    max_size = max([len(t) for t in seq])
                    attn_mask = torch.eye(max_size)
                    attn_mask[1:, :-1] += torch.eye(max_size - 1)
                    attn_mask[:-1, 1:] += torch.eye(max_size - 1)
#                     attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
                    batch[where][col]['attention_mask'] = attn_mask.to(device)
            output.append((batch, label))
        return output


class Seq2SeqDifference(nn.Module):
    def __init__(self, dim, dense_hidden=60):
        super(Seq2SeqDifference, self).__init__()
#         self.embeddings = embeddings_layer
        self.dim = dim
        self.conv1 = SentenceConv(1, 100, self.dim)
#         self.conv2 = SentenceConv(2, 100, self.dim)
#         self.conv3 = SentenceConv(3, 100, self.dim)
        self.dense = nn.Sequential(
            nn.Linear(100 * 2, dense_hidden),
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
#             l_vectors[torch.isnan(l_vectors)] = 0
#             r_vectors[torch.isnan(r_vectors)] = 0

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
        return self.dense(torch.cat(features, 1))

class Seq2SeqPlusMatcher(nn.Module):
    def __init__(self, word_embeddings, columns, dense_hidden=60):
        super(Seq2SeqPlusMatcher, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        self.dim = self.embeddings.embedding_dim
        self.matchers = nn.ModuleDict()
        self.left_encoders = nn.ModuleDict()
        self.right_encoders = nn.ModuleDict()
        self.attr_convs = nn.ModuleDict()
        self.attr_denses = nn.ModuleDict()
        l_att = nn.TransformerEncoderLayer(d_model=self.dim, nhead=5)
        l_att = nn.TransformerEncoder(l_att, num_layers=1)
        r_att = nn.TransformerEncoderLayer(d_model=self.dim, nhead=5)
        r_att = nn.TransformerEncoder(r_att, num_layers=1)
        matcher = Seq2SeqDifference(self.dim, dense_hidden)
#         r_att = l_att
        for col in columns:
            self.matchers[col] = matcher
            self.left_encoders[col] = l_att
            self.right_encoders[col] = r_att
            self.attr_convs[col] = SentenceConv(ngram=1, duplicate=dense_hidden, dim=self.dim)
            self.attr_denses[col] = nn.Sequential(
                nn.Linear(dense_hidden * 2, 1),
                nn.ReLU(),
            )
        self.columns = columns
        self.dense = nn.Sequential(
            nn.Linear(dense_hidden * len(columns), dense_hidden),
            nn.ReLU(),
#             nn.Linear(dense_hidden * len(columns), dense_hidden),
#             nn.ReLU(),
            nn.Linear(dense_hidden, 2),
        )
    def forward(self, batch):
        features = []
        weights = []
#         self.left_vectors = []
#         self.right_vectors = []

        for col in self.columns:
            left = batch['left'][col]
            right = batch['right'][col]
            l_lengths, l_tokens = left['size'], left['seq']
            r_lengths, r_tokens = right['size'], right['seq']
            l_vectors = self.embeddings(l_tokens)
            r_vectors = self.embeddings(r_tokens)
            l_vectors = self.left_encoders[col](l_vectors, 1 - left['attention_mask'], ~left['key_padding_mask'])
            r_vectors = self.right_encoders[col](r_vectors, 1 - right['attention_mask'], ~right['key_padding_mask'])
#             print(l_vectors.shape, left['key_padding_mask'].shape)
            f = torch.cat([
                self.attr_convs[col](l_vectors, left['key_padding_mask'].T), 
                self.attr_convs[col](r_vectors, right['key_padding_mask'].T)
            ], 1)
            weights.append(self.attr_denses[col](f))
            features.append(self.matchers[col](l_lengths, l_vectors, r_lengths, r_vectors))
        return self.dense(torch.cat(features, 1) * torch.repeat_interleave(torch.softmax(torch.cat(weights, 1), 1), 60, 1))