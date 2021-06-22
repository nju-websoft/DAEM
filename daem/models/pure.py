import torch
import torch.nn as nn
from .nn import SentenceDifferenceEncoding

class PureMatcher(nn.Module):
    def __init__(self, word_embeddings, columns, dense_hidden=60, fixed_matcher=True):
        super(PureMatcher, self).__init__()
        self.columns = columns
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings), freeze=True)
        self.dim = self.embeddings.embedding_dim
        self.matchers = nn.ModuleDict()

        if fixed_matcher:
            matcher = SentenceDifferenceEncoding(self.dim, weights_dim=[dense_hidden])
            for col in columns:
                self.matchers[col] = matcher
        else:
            for col in columns:
                self.matchers[col] = SentenceDifferenceEncoding(self.dim, weights_dim=[dense_hidden])
        self.dense = nn.Sequential(
            nn.Linear(dense_hidden * len(columns), dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 2),
        )
        
    def forward(self, batch):
        features = []
        for col in self.columns:
            left = batch['left'][col]
            right = batch['right'][col]
            l_lengths, l_tokens = left['size'], left['seq']
            r_lengths, r_tokens = right['size'], right['seq']
            l_vectors = self.embeddings(l_tokens)
            r_vectors = self.embeddings(r_tokens)
            features.append(self.matchers[col](l_lengths, l_vectors, r_lengths, r_vectors))
        return self.dense(torch.cat(features, 1))
