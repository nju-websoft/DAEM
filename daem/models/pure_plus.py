from daem.models.nn.neural import sequential_neural_network
from daem.models.nn import SentenceConv
from daem.models.nn import PositionalEncoding
from daem.models.base import EntityMatchingModule
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

with_similarity_vectors = 1
with_difference_vectors = 1
with_token_weight = True
with_transfomer = True


class TokenWeight(nn.Module):
    def __init__(self, we_dim, weights_dim=None):
        super(TokenWeight, self).__init__()
        if weights_dim is None:
            weights_dim = [400, 20]
        self.token_weight = sequential_neural_network([we_dim] + weights_dim, nn.SELU, 0.05)
    def forward(self, sequence, size):
        return self.token_weight(sequence).max(2).values.unsqueeze(2)


class LSTMTokenWeight(nn.Module):
    def __init__(self, we_dim, hidden_size=60, max_len=128):
        super(LSTMTokenWeight, self).__init__()
        self.lstm = nn.LSTM(we_dim, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1, bias=False)
        mask_matrix = torch.log(pad_sequence([torch.ones(k) for k in range(0, max_len + 1)]))
        self.register_buffer('mask_matrix', mask_matrix)
    
    def forward(self, seq, size):
        packed = pack_padded_sequence(seq, size, enforce_sorted=False)
        output, (_, _) = self.lstm(PackedSequence(packed.data, packed.batch_sizes))
        output, _ = pad_packed_sequence(PackedSequence(self.linear(output.data), output.batch_sizes))
        output = output[:, packed.unsorted_indices, :] + self.mask_matrix[:size.max(), size].unsqueeze(2)
        output = torch.sigmoid(output)
        # output = torch.softmax(output, 1)
        return output


class SentenceComparison(nn.Module):
    def __init__(self, max_length=128):
        super(SentenceComparison, self).__init__()
        mask_matrix = pad_sequence([torch.ones(k) for k in range(0, max_length + 1)]).T
        self.register_buffer('mask_matrix', mask_matrix)
    def forward(self, left_vectors, left_lengths, right_vectors, right_lengths):
        # print(left_vectors.shape)
        # print(right_vectors.shape)
        left_v = left_vectors / torch.sqrt(torch.pow(left_vectors, 2).sum(1)).unsqueeze(1)
        right_v = right_vectors / torch.sqrt(torch.pow(right_vectors, 2).sum(1)).unsqueeze(1)
        # torch.sqrt(torch.pow(we, 2).sum(1)).unsqueeze(1)
        # assert False
        left_mask = self.mask_matrix[left_lengths, :left_vectors.shape[0]]
        right_mask = self.mask_matrix[right_lengths, :right_vectors.shape[0]]
        attention = torch.bmm(left_v.transpose(0, 1), right_v.transpose(0, 1).transpose(1, 2))
        attention = attention * left_mask.unsqueeze(2) * right_mask.unsqueeze(1)
        self.attention = attention

        left_sim, indices = torch.max(attention, 2)
        indices = indices.transpose(0, 1).unsqueeze(2).expand(-1, -1, left_vectors.shape[2])
        left_difference = left_vectors - right_vectors.gather(0, indices)
        left_difference *= left_mask.T.unsqueeze(2)

        right_sim, indices = torch.max(attention, 1)
        indices = indices.T.unsqueeze(2).expand(-1, -1, left_vectors.shape[2])
        right_difference = right_vectors - left_vectors.gather(0, indices)
        right_difference *= right_mask.T.unsqueeze(2)
        
        return left_difference, right_difference, left_mask, right_mask, left_sim.T, right_sim.T

class WeightedSentenceComparison(nn.Module):
    def __init__(self, we_dim, weights_dim=None, max_length=128, token_weight=None):
        super(WeightedSentenceComparison, self).__init__()
        self.sd = SentenceComparison(max_length)
        if weights_dim is None:
            weights_dim = [400, 20]
        if token_weight is None:
            token_weight = TokenWeight(we_dim)
        self.token_weight = token_weight
    def forward(self, left_vectors, left_lengths, right_vectors, right_lengths, l_weights, r_weights):
        left_difference, right_difference, left_mask, right_mask, ls, rs = self.sd(left_vectors, left_lengths, right_vectors, right_lengths)
        left_difference *= l_weights
        right_difference *= r_weights
        
        return left_difference, right_difference, left_mask, right_mask, ls, rs


class SentenceComparisonEncoding(nn.Module):
    def __init__(self, we_dim, sent_diff=None, hidden_dim=100, weights_dim=None):
        super(SentenceComparisonEncoding, self).__init__()
        self.dim = we_dim
        self.conv1 = SentenceConv(1, hidden_dim, self.dim)
        self.sim_conv1 = SentenceConv(1, hidden_dim, self.dim)
        if sent_diff is None:
            self.sd = WeightedSentenceComparison(self.dim)
        else:
            self.sd = sent_diff
        if weights_dim is None:
            weights_dim = [60]
        self.dense = sequential_neural_network([hidden_dim * 2] + weights_dim, nn.SELU)
    def forward(self, l_lengths, l_vectors, r_lengths, r_vectors, l_weights, r_weights):
        features = []
        sim_features = []
        l_c, r_c, l_mask, r_mask, l_s, r_s = self.sd(l_vectors, l_lengths, r_vectors, r_lengths)
        for the_c, token_weight, the_mask in [(l_c, l_weights, l_mask), (r_c, r_weights, r_mask)]:
            features.append(self.conv1(the_c.abs() * token_weight, the_mask.T))
        for the_c, the_weight, token_weight, the_mask in [(l_vectors, l_s, l_weights, l_mask), (r_vectors, r_s, r_weights, r_mask)]:
            v = the_c *  token_weight * nn.SELU()(the_weight.unsqueeze(2))
            sim_features.append(self.sim_conv1(v, the_mask.T))
        
        return self.dense(torch.cat(features, 1)) * with_difference_vectors, self.dense(torch.cat(sim_features, 1)) * with_similarity_vectors


class PurePlusMatcher(EntityMatchingModule, pl.LightningModule):
    def __init__(self, word_embeddings, columns, dense_hidden=60, fixed_matcher=True, max_length=128, encoder_layer_num=3):
        super(PurePlusMatcher, self).__init__()
        self._prepare_buffer(max_length)
        self.columns = columns
        we = torch.Tensor(word_embeddings)
        we = we / torch.sqrt(torch.pow(we, 2).sum(1)).unsqueeze(1)
        self.embeddings = nn.Embedding.from_pretrained(we, freeze=True)
        self.dim = self.embeddings.embedding_dim
        self.matchers = nn.ModuleDict()
        sent_diff = SentenceComparison(self.dim)
        self.token_weight = TokenWeight(self.dim)
        if False:
            matcher = SentenceComparisonEncoding(self.dim, sent_diff=sent_diff, weights_dim=[dense_hidden])
            for col in columns:
                self.matchers[col] = matcher
        else:
            for col in columns:
                self.matchers[col] = SentenceComparisonEncoding(self.dim, sent_diff=sent_diff, weights_dim=[dense_hidden])
        
        self.dense = nn.Sequential(
            nn.Linear(dense_hidden * len(columns) * 2, dense_hidden),
            nn.SELU(),
            nn.Linear(dense_hidden, 2),
        )
    
    def _prepare_buffer(self, max_length):
        if True:
            mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        else:
            mask = torch.eye(max_length, max_length)
            mask[1:, :-1] += torch.eye(max_length - 1, max_length - 1)
            mask[:-1, 1:] += torch.eye(max_length - 1, max_length - 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('attention_mask', mask)
        mask = (1 - pad_sequence([torch.ones(k) for k in range(0, max_length + 1)]).T).bool()
        self.register_buffer('src_key_padding_mask', mask)
        
    def forward(self, batch):
        features = []
        sim_features = []
        for col in self.columns:
            left = batch['left'][col]
            right = batch['right'][col]
            l_lengths, l_tokens = left['size'], left['seq']
            r_lengths, r_tokens = right['size'], right['seq']
            l_vectors = self.embeddings(l_tokens)
            r_vectors = self.embeddings(r_tokens)
            if with_token_weight:
                l_weights = self.token_weight(l_vectors, l_lengths)
                r_weights = self.token_weight(r_vectors, r_lengths)
            else:
                l_weights = 1.0
                r_weights = 1.0
            diff, sim = self.matchers[col](l_lengths, l_vectors, r_lengths, r_vectors, l_weights, r_weights)
            features.append(diff)
            sim_features.append(sim)
        return self.dense(torch.cat(features + sim_features, 1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer



def test_model(task_name):
    from daem.utils.cache_decorator import CacheDecoreator
    from daem.data.collate import entity_pad_collate
    from torch.utils.data import DataLoader
    import numpy as np

    def DeepMatcherDataset(base_dir):
        from daem.data.dataset import EntityPairDataset
        from pathlib import Path
        import fastText as fasttext
        base_dir = Path(base_dir)
        train = EntityPairDataset(base_dir / 'train.csv')
        valid = EntityPairDataset(base_dir / 'validation.csv')
        test = EntityPairDataset(base_dir / 'test.csv')
        tokens = [c.get_tokens() for c in [train, valid, test]]
        tokens = np.unique(np.concatenate(tokens))
        print('Token set size:', tokens.shape[0])
        from daem.data.vocabulary_table import VocabularyTable
        word_embedding_model = fasttext.load_model('wiki.en.bin')
        vocab = VocabularyTable(tokens, word_embedding_model)
        train.inject_vocab(vocab)
        valid.inject_vocab(vocab)
        test.inject_vocab(vocab)
        return train, valid, test, vocab
    # data
    train, valid, test, vocab = DeepMatcherDataset('datasets/Structured/' + task_name)
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(train, batch_size=128, collate_fn=entity_pad_collate, num_workers=8)
    val_loader = DataLoader(valid, batch_size=256, collate_fn=entity_pad_collate, num_workers=8)
    test_loader  = DataLoader(test, batch_size=256, collate_fn=entity_pad_collate, num_workers=8)

    # model
    # columns = ['name', 'addr', 'city']
    model = PurePlusMatcher(vocab.vectors, train.columns)

    # training
    trainer = pl.Trainer(gpus=1, max_epochs=50, progress_bar_refresh_rate=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


def main():
    task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'iTunes-Amazon', 
                  'Walmart-Amazon', 'Fodors-Zagats', 'Beer']
    for task_name in task_names:
        test_model(task_name)

if __name__ == '__main__':
    main()
