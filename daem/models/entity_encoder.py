import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
from daem.models.functional import packed_sequence_mean
from better_lstm import LSTM

class DeepERLSTMEncoder(nn.Module):
    def __init__(self, word_embeddings, columns, composition='lstm', difference='cosine', classification='50-2', dropout=0.05):
        super(DeepERLSTMEncoder, self).__init__()
        self.composition = composition
        self.difference = difference
        self.classification = classification
        self.columns = columns
        # token embeddings
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        # token composition layer
        if self.composition == 'lstm':
            self.lstm = LSTM(self.embeddings.embedding_dim, 150, dropoutw=dropout)

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
        representations = []

        for column in self.columns:
            seq = X[column]['seq']
            size = X[column]['size']
            packed = pack_padded_sequence(seq, size, enforce_sorted=False)
            representations.append(self._composite(packed)[packed.unsorted_indices])
        return torch.cat(representations, 1)


class AutoBlockEncoder(nn.Module):
    def __init__(self, word_embeddings, hidden_size=60, rho=1.0, max_len=128):
        super(AutoBlockEncoder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings))
        self.lstm = nn.LSTM(self.embeddings.embedding_dim, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1, bias=False)
        self.rho = rho
        mask_matrix = torch.log(pad_sequence([torch.ones(k) for k in range(0, max_len + 1)]))
        self.register_buffer('mask_matrix', mask_matrix)
    
    def forward(self, seq, size):
        packed = pack_padded_sequence(seq, size.to('cpu'), enforce_sorted=False)
        output, (_, _) = self.lstm(PackedSequence(self.embeddings(packed.data), packed.batch_sizes))
        output, _ = pad_packed_sequence(PackedSequence(self.linear(output.data), output.batch_sizes))
        output = output[:, packed.unsorted_indices, :] + self.mask_matrix[:size.max(), size].unsqueeze(2)
        output = self.rho * torch.softmax(output, 1) + (1 - self.rho) / size.unsqueeze(0).unsqueeze(2).float()
        output = (output * self.embeddings(seq)).sum(0)
        return output