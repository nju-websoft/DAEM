import torch
import torch.nn as nn
from better_lstm import LSTM
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pad_sequence
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from daem.helpers import performance_batch


class InterffeDatasetContainer():
    def __init__(self, word_index, char_index):
        self.word_index = word_index
        self.char_index = char_index

    def split_and_pack(self, src_tokens, dst_tokens, batch_size=120, device='cpu'):
        output = []

        labels = [[tok in dst_tokens[i] for tok in seq]
                  for i, seq in enumerate(src_tokens)]
        for i in range(0, len(src_tokens), batch_size):
            begin = i
            end = min(len(src_tokens), i + batch_size)
            packed = pack_sequence(
                [torch.LongTensor([self.word_index[tok] for tok in seq])
                 for seq in src_tokens[begin:end]],
                enforce_sorted=False).to(device)
            seq_label = pack_sequence([torch.LongTensor(labels[begin:end][i]) for i in packed.sorted_indices]).data.to(
                device)
            output.append((packed, seq_label))
        return output


class Interffe(nn.Module):
    def __init__(self, we):
        super(Interffe, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(we))
        self.lstm_size = int(self.embeddings.embedding_dim * 0.5)
        self.lstm = LSTM(self.embeddings.embedding_dim,
                         self.lstm_size, dropoutw=0.05, bidirectional=True)
        self.lstm.flatten_parameters()
        self.linear1 = nn.Linear(self.lstm_size * 2, 100)
        self.linear2 = nn.Linear(100, 30)
        self.linear3 = nn.Linear(30, 2)

    def forward(self, input):
        input, _ = input
        out, (_, _) = self.lstm(PackedSequence(
            self.embeddings(input.data), input.batch_sizes))
        out = torch.relu(self.linear1(out.data + self.embeddings(input.data)))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out


class InterffeLSTMCNNResidual(nn.Module):
    def __init__(self, we, char_length):
        super(InterffeLSTMCNNResidual, self).__init__()
        char_embedding_dim = 50
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(we))
        self.token_embedding_dim = 50
        # self.char_embeddings = nn.Embedding(len(iffe.i2c), char_embedding_dim)
        self.char_embeddings = nn.Embedding(char_length, char_embedding_dim)
        self.lstm_size = int(self.token_embedding_dim * 0.5)
        self.lstm = LSTM(self.embeddings.embedding_dim + self.token_embedding_dim,
                         self.lstm_size, dropoutw=0.05, bidirectional=True)
        self.lstm.flatten_parameters()
        self.linear1 = nn.Linear(self.lstm_size * 2, 100)
        self.linear2 = nn.Linear(100, 30)
        self.linear3 = nn.Linear(30, 2)
        self.char_cnn = nn.Conv2d(in_channels=1, out_channels=self.token_embedding_dim, kernel_size=(3, char_embedding_dim), padding=(2, 0))

    def forward(self, input):
        token_seq, char_seq = input
        cnn_out = self.char_cnn(self.char_embeddings(char_seq.data).unsqueeze(1))
        char_embed = nn.functional.max_pool2d(cnn_out, kernel_size=(cnn_out.size(2), 1)).view(cnn_out.size(0), self.token_embedding_dim)
        out, (_, _) = self.lstm(PackedSequence(
            torch.cat([self.embeddings(token_seq.data), char_embed], 1)
            , token_seq.batch_sizes))
        out = torch.relu(self.linear1(out.data + char_embed))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out


class InterffeLSTM(nn.Module):
    def __init__(self, we):
        super(InterffeLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(we))
        self.lstm_size = int(self.embeddings.embedding_dim * 0.5)
        self.lstm = LSTM(self.embeddings.embedding_dim,
                         self.lstm_size, dropoutw=0.05, bidirectional=True)
        self.lstm.flatten_parameters()
        self.linear1 = nn.Linear(self.lstm_size * 2, 100)
        self.linear2 = nn.Linear(100, 30)
        self.linear3 = nn.Linear(30, 2)

    def forward(self, input):
        input, _ = input
        out, (_, _) = self.lstm(PackedSequence(
            self.embeddings(input.data), input.batch_sizes))
        out = torch.relu(self.linear1(out.data))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out


class InterffeLSTMCNN(nn.Module):
    def __init__(self, we, char_length):
        super(InterffeLSTMCNN, self).__init__()
        char_embedding_dim = 50
        token_embedding_dim = 25
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(we))
        # self.char_embeddings = nn.Embedding(len(iffe.i2c), char_embedding_dim)
        self.char_embeddings = nn.Embedding(char_length, char_embedding_dim)
        self.lstm_size = int(self.embeddings.embedding_dim * 0.5)
        self.lstm = LSTM(self.embeddings.embedding_dim + token_embedding_dim,
                         self.lstm_size, dropoutw=0.05, bidirectional=True)
        self.lstm.flatten_parameters()
        self.linear1 = nn.Linear(self.lstm_size * 2, 100)
        self.linear2 = nn.Linear(100, 30)
        self.linear3 = nn.Linear(30, 2)
        self.char_cnn = nn.Conv2d(in_channels=1, out_channels=token_embedding_dim, kernel_size=(3, char_embedding_dim), padding=(2, 0))

    def forward(self, input):
        token_seq, char_seq = input
        cnn_out = self.char_cnn(self.char_embeddings(char_seq.data).unsqueeze(1))
        char_embed = nn.functional.max_pool2d(cnn_out, kernel_size=(cnn_out.size(2), 1)).view(cnn_out.size(0), 25)
        out, (_, _) = self.lstm(PackedSequence(
            torch.cat([self.embeddings(token_seq.data), char_embed], 1)
            , token_seq.batch_sizes))
        out = torch.relu(self.linear1(out.data))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out

class InterFieldFeatureExtraction(object):
    def __init__(self, model_target):
        self.word_embedding_model = None
        self.model_target = model_target

    def run_ffe_model(self, entities, src, dst, loop_cnt=40, device='cuda', model=Interffe):
        data = entities[[src, dst]]
        data = data[data[src].notna()]
        test = data[data[dst].isna()]
        train = data[data[dst].notna()]
        train_src = train[src].values
        train_dst = train[dst].values

        train_src_tokens = [
            nltk.tokenize.word_tokenize(seq) for seq in train_src]
        train_dst_tokens = [
            nltk.tokenize.word_tokenize(seq) for seq in train_dst]
        test_src_tokens = [nltk.tokenize.word_tokenize(
            seq) for seq in test[src].values]

        self._prepare_word_embedding(
            train_src_tokens + test_src_tokens, test_src_tokens)
        
        self.train(train_src_tokens, train_dst_tokens, loop_cnt, device=device, model=model)
        
        return self.predict(test, test_src_tokens, src, dst, device)
    
    
    def predict(self, test, test_src_tokens, src, dst, device):
        test_batches = self._split_and_pack(test_src_tokens, device=device)
        self.net.load_state_dict(torch.load(self.model_target))
        self.net.eval()
        begin = 0
        end = 0
        targets = []
        for batch in test_batches:
            begin = end
            end = begin + int(batch[0].batch_sizes[0].numpy())
            self.net.eval()
            out = self.net(batch)
            out = (torch.softmax(out, 1)[:, 1] >= 0.5).detach().to('cpu')
            labels = pad_packed_sequence(PackedSequence(out, batch[0].batch_sizes))[0].T[batch[0].unsorted_indices]
            for i, seq in enumerate(test_src_tokens[begin:end]):
                target = [t for j, t in enumerate(seq) if labels[i, j]]
                targets.append(target)
        src2dst = {}
        src2dst[src] = test[src].values
        src2dst[dst] = targets
        return pd.DataFrame(src2dst)
    
    def train(self, train_src_tokens, train_dst_tokens, loop_cnt, device, model):
        batches = self._split_and_pack(
            train_src_tokens, train_dst_tokens, 16, device=device)
        if model == InterffeLSTMCNN or model == InterffeLSTMCNNResidual:
            self.net = model(self.we, len(self.i2c)).to(device)
        else:
            self.net = model(self.we).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1])).to(device)
        optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        train_batch_size = int(len(batches) / 4 * 3)

        best_f1 = -1.0

        with tqdm(range(0, loop_cnt * len(batches) * 2)) as pbar:
            for epoch in range(0, loop_cnt):
                self.net.train()
                for batch, label in batches[:train_batch_size]:
                    optimizer.zero_grad()
                    output = self.net(batch)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    pbar.update()
                with torch.no_grad():
                    self.net.eval()
                    p0 = performance_batch(self.net, batches[:train_batch_size])
                    p1 = performance_batch(self.net, batches[train_batch_size:])
                    pbar.update(len(batches))
                    template = 'epoch=%d\ttrain=%.3f,%.3f,%.3f\tvalid=%.3f,%.3f,%.3f'
                    pbar.set_postfix_str(template % tuple([epoch] + list(p0) + list(p1)))
                    if p1[2] > best_f1:
                        torch.save(self.net.state_dict(), self.model_target)

    def _prepare_word_embedding(self, src_tokens, dst_tokens):
        if self.word_embedding_model is None:
            import fastText
            import os
            self.word_embedding_model = fastText.load_model(os.path.realpath(os.path.join(__file__, '../../../wiki.en.bin')))
        self.i2t = list(set(sum(src_tokens, [])) | set(sum(dst_tokens, [])))
        self.t2i = dict((t, i) for i, t in enumerate(self.i2t))
        self.we = np.zeros(
            (len(self.i2t), self.word_embedding_model.get_dimension()))
        for i, t in enumerate(self.i2t):
            self.we[i, :] = self.word_embedding_model.get_word_vector(t)

        charset = set(''.join(self.i2t))
        charset = ['<PAD>'] + list(charset)
        self.i2c = charset
        self.c2i = dict((w, i) for i, w in enumerate(self.i2c))

    def _split_and_pack(self, src_tokens, dst_tokens=None, batch_size=120, device='cpu'):
        output = []

        if dst_tokens is not None:
            labels = [[tok in dst_tokens[i] for tok in seq]
                        for i, seq in enumerate(src_tokens)]
        for i in range(0, len(src_tokens), batch_size):
            begin = i
            end = min(len(src_tokens), i + batch_size)
            packed = pack_sequence(
                [torch.LongTensor([self.t2i[tok] for tok in seq])
                    for seq in src_tokens[begin:end]],
                enforce_sorted=False).to(device)
            batch_tokens = [self.i2t[t] for t in packed.data]
            char_seq = pad_sequence([torch.LongTensor([self.c2i.get(b, 0) for b in token]) for token in batch_tokens]).T.to(device)
            if dst_tokens is not None:
                seq_label = pack_sequence([torch.LongTensor(labels[begin:end][i]) for i in packed.sorted_indices]).data.to(
                    device)
                output.append(((packed, char_seq), seq_label))
            else:
                output.append((packed, char_seq))
        return output
