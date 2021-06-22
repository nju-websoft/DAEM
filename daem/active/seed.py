import random

import deepmatcher as dm
import pandas as pd
import numpy as np


def numpy_topk(arr, k):
    threshold = np.partition(arr, k)[k]
    return np.arange(0, arr.shape[0])[arr <= threshold][:k]


def extract_seeds_jaccard(examples, fields, cnt=10):
    import numpy as np
    from tqdm import tqdm
    vectors = []
    for field in fields:
        sim = np.zeros(len(examples))
        for i in tqdm(range(0, len(examples))):
            left_value = getattr(examples[i], 'left_' + field)
            right_value = getattr(examples[i], 'right_' + field)
            if len(left_value) == 0 or len(right_value) == 0:
                sim[i] = 0
            else:
                sim[i] = len(set(left_value) & set(right_value)) / \
                    len(set(left_value) | set(right_value))
        vectors.append(sim)
    vectors = np.vstack(vectors).T
    import numpy as np
    npred = np.array([(vectors >= vectors[i]).all(1).sum()
                      for i in range(0, vectors.shape[0])])
    nsucc = np.array([(vectors <= vectors[i]).all(1).sum()
                      for i in range(0, vectors.shape[0])])
    return list(numpy_topk(npred, cnt)) + list(numpy_topk(nsucc, cnt))


def extract_seeds_bert_fast(examples, fields, cnt=10):
    from bert_serving.client import BertClient
    import numpy as np
    bc = BertClient()
    vectors = []
    for field in fields:
        sim = -np.ones(len(examples))
        left_tokens = [getattr(e, 'left_' + field) for e in train.examples]
        right_tokens = [getattr(e, 'right_' + field) for e in train.examples]
        indexes = [i for i in range(0, len(left_tokens)) if len(
            left_tokens[i]) > 0 and len(right_tokens[i]) > 0]
        x = bc.encode([left_tokens[i] for i in indexes], is_tokenized=True)
        x = x / np.sqrt((x * x).sum(1)).reshape(-1, 1)
        y = bc.encode([right_tokens[i] for i in indexes], is_tokenized=True)
        y = y / np.sqrt((y * y).sum(1)).reshape(-1, 1)
        sim[indexes] = (x * y).sum(1)
        vectors.append(sim)
    vectors = np.vstack(vectors).T
    import numpy as np
    npred = np.array([(vectors >= vectors[i]).all(1).sum()
                      for i in range(0, vectors.shape[0])])
    nsucc = np.array([(vectors <= vectors[i]).all(1).sum()
                      for i in range(0, vectors.shape[0])])
    return list(numpy_topk(npred, cnt)) + list(numpy_topk(nsucc, cnt))


def generate_train_examples(left, right, columns, n_positive=20, n_negative=4):
    examples = []
    for x, x_prefix, y, y_prefix in [(left, 'left_', right, 'right_'), (right, 'right_', left, 'left_')]:
        x = x[columns].sample(n_positive).rename(columns=dict((c, x_prefix + c) for c in columns))
        x = pd.concat([x] * n_negative, ignore_index=True)
        y = y[columns].sample(n_positive * n_negative).rename(columns=dict((c, y_prefix + c) for c in columns))
        y.index = x.index
        examples.append(pd.merge(x, y, left_index=True, right_index=True, how='inner').assign(label=0))
        for col in columns:
            x[y_prefix + col] = x[x_prefix + col].apply(lambda l: [x for x in l if random.random() > 0.1])
        examples.append(x.assign(label = 1))
    examples = pd.concat(examples, sort=False)
    examples = examples.loc[np.random.permutation(examples.index)]
    return list(examples.itertuples())