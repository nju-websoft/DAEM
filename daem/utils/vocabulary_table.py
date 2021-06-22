import numpy as np

class VocabularyTable():
    def __init__(self, df, columns, word_embedding_model):
        token_set = set()
        for col in columns:
            for tokens in df[col]:
                token_set |= set(tokens)
        self.itos = sorted(list(token_set))
        if '<unk>' in self.itos:
            self.itos.remove('<unk>')
        if '<pad>' in self.itos:
            self.itos.remove('<pad>')
        self.itos = ['<unk>', '<pad>'] + self.itos
        self.stoi = dict((t, i) for i, t in enumerate(self.itos))
        self.vectors = np.zeros(
            (len(self.itos), word_embedding_model.get_dimension()))
        for i, t in enumerate(self.itos):
            self.vectors[i, :] = word_embedding_model.get_word_vector(t)
