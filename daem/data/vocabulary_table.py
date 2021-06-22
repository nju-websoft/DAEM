import numpy as np

class VocabularyTable():
    def __init__(self, tokens, word_embedding_model):
        self.itos = np.sort(tokens)
        if '<unk>' in self.itos:
            self.itos.remove('<unk>')
        if '<pad>' in self.itos:
            self.itos.remove('<pad>')
        self.itos = np.concatenate([['<unk>', '<pad>'], self.itos])
        self.stoi = dict((t, i) for i, t in enumerate(self.itos))
        self.vectors = np.zeros(
            (len(self.itos), word_embedding_model.get_dimension()))
        for i, t in enumerate(self.itos):
            self.vectors[i, :] = word_embedding_model.get_word_vector(t)
