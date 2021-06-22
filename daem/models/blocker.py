import numpy as np
import pandas as pd
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
        self.conv = SentenceConv(1, duplicate=100, dim=self.embeddings.embedding_dim, pooling='max')
        
        mask_matrix = pad_sequence([torch.ones(k) for k in range(0, 256 + 1)]).T
        self.register_buffer('mask_matrix', mask_matrix)

    def forward(self, X):
        representations = []

        for column in self.columns:
            seq = X[column]['seq']
            size = X[column]['size']
            seq = self.embeddings(seq)
            representations.append(self.conv(seq, self.mask_matrix[size, :size.max()].T))
        return torch.cat(representations, 1)


def static_blocking(encoder, container, columns, left, right, threshold=0.8):
    k = int(np.sqrt([len(left), len(right), 1e2]).min())
    left_batches = container.split_tuples('left', left['id'].values, batch_size=128, device='cuda')
    right_batches = container.split_tuples('right', right['id'].values, batch_size=128, device='cuda')
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

    pairs = []
    for i in range(0, len(left)):
        if np.sum(scores[i, :] >= threshold) < k:
            right_ids = np.arange(0, len(right))[scores[i, :] >= threshold]
        else:
            right_ids = np.argpartition(-scores[i, :], k)[:k]
        if len(right_ids) > 0:
            left_ids = np.array([i] * len(right_ids))
            pairs.append((left_ids, right_ids))
    pairs = pd.DataFrame({
        'left_id': np.concatenate([v for v, _ in pairs]),
        'right_id': np.concatenate([v for _, v in pairs])
    })
    
    return pairs