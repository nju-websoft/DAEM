# from __future__ import print_function, division
import os
import re
# import torch as th
import pandas as pd
import numpy as np
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

NON_WHITE_SPACE_REG_EXP = re.compile(r'\S+')


def white_space_tokenize(sent):
    if sent is None:
        return sent
    else:
        sent = str(sent)
        return NON_WHITE_SPACE_REG_EXP.findall(sent)


class EntityPairDataset(Dataset):
    """"Entity pair dataset."""

    def __init__(self, csv_file, columns=None, tokenize=None):
        df = pd.read_csv(csv_file)
        if columns is None:
            columns = [c[len('left_'):]
                       for c in df.columns if c.startswith('left_')]
        if tokenize is None:
            tokenize = white_space_tokenize
        self.columns = columns
        self.df = df
        for col in self.columns:
            for where in ('left_', 'right_'):
                df[where + col] = df[where + col].apply(tokenize)
        self.labels = df['label'].values

    def get_tokens(self):
        tokens = []
        for col in self.columns:
            for where in ('left_', 'right_'):
                tokens.append(np.concatenate(self.df[where + col].values))
        tokens = np.concatenate(tokens)
        return np.unique(tokens)

    def inject_vocab(self, vocab):
        self.items = []
        for _, row in self.df.iterrows():
            item = {'left': dict(), 'right': dict()}
            for col in self.columns:
                for where in ('left', 'right'):
                    token_ids = [vocab.stoi[c] for c in row[where + '_' + col]]
                    item[where][col] = np.array(token_ids, dtype=np.long)
            self.items.append(item)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx], self.labels[idx]


def main():
    class MockedEmbedModel(object):
        def get_dimension(self):
            return 100

        def get_word_vector(self, t):
            return np.zeros(100)

    print(NON_WHITE_SPACE_REG_EXP.findall('     '))
    print(NON_WHITE_SPACE_REG_EXP.findall('   34234  34  '))
    from pathlib import Path
    base_dir = Path('datasets/Structured/DBLP-GoogleScholar')
    train = EntityPairDataset(base_dir / 'train.csv')
    valid = EntityPairDataset(base_dir / 'validation.csv')
    test = EntityPairDataset(base_dir / 'test.csv')
    tokens = [c.get_tokens() for c in [train, valid, test]]
    tokens = np.unique(np.concatenate(tokens))
    print('Token set size:', tokens.shape[0])
    from daem.data.vocabulary_table import VocabularyTable
    vocab = VocabularyTable(tokens, MockedEmbedModel())
    train.inject_vocab(vocab)
    valid.inject_vocab(vocab)
    test.inject_vocab(vocab)
    print(train.items[0])


if __name__ == '__main__':
    main()
