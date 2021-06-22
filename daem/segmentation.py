base_dir = '../datasets/Structured-raw/'
tasks = 'Amazon-Google, DBLP-ACM, DBLP-GoogleScholar, iTunes-Amazon, Walmart-Amazon'.split(', ')
# tasks = 'Amazon-Google, DBLP-ACM, iTunes-Amazon, Walmart-Amazon'.split(', ')
print(tasks)

iter_feature_extraction_fields = {
    'Amazon-Google': [('title', 'manufacturer')],
    'DBLP-ACM': [('title', 'venue')],
    'DBLP-GoogleScholar': [('title', 'venue')],
    'Walmart-Amazon': [('title', 'brand'), ('title', 'modelno'), ('title', 'category')]
}


import pandas as pd
import numpy as np
# import py_stringmatching as sm


def split_entity_columns(df, index_column='id'):
    columns = list(df.columns)
    columns.remove(index_column)
    return index_column, columns


def load_dataset(base_dir, task_name):
    task_dir = base_dir + task_name + '/'
    left = pd.read_csv(task_dir + 'tableA.csv')
    right = pd.read_csv(task_dir + 'tableB.csv')
    train = pd.read_csv(task_dir + 'train.csv')
    valid = pd.read_csv(task_dir + 'valid.csv')
    test = pd.read_csv(task_dir + 'test.csv')
    idx, columns = split_entity_columns(left)
    assert idx == split_entity_columns(right)[0]
    assert columns == split_entity_columns(right)[1]
    return left, right, train, valid, test, idx, columns


def feature_extraction(samples, columns, model, label_column='label'):
    X = []
    y = []
    for _, row in samples.iterrows():
        vector = []
        for col in columns:
            l = model.get_word_vector(str(row['ltable_' + col]))
            r = model.get_word_vector(str(row['rtable_' + col]))
            f = (l * r).sum(0) / np.sqrt((l * l).sum(0) * (r * r).sum(0))
            vector.append(f)
        X.append(np.array(vector))
        y.append(row[label_column])
    return np.array(X), np.array(y)


def feature_extraction_v2(samples, columns, model, label_column='label'):
    simfunc = sm.Jaccard()
    tokenizer = sm.AlphanumericTokenizer(return_set=True)
    tokenizer1 = sm.QgramTokenizer(qval=2)
    X = []
    y = []
    for _, row in samples.iterrows():
        vector = []
        for col in columns:
            if col != 'price':
                l = tokenizer.tokenize(str(row['ltable_' + col]))
                r = tokenizer.tokenize(str(row['rtable_' + col]))
                f = sm.OverlapCoefficient().get_sim_score(l, r)
                vector.append(f)
                l = tokenizer1.tokenize(str(row['ltable_' + col]))
                r = tokenizer1.tokenize(str(row['rtable_' + col]))
                f = simfunc.get_sim_score(l, r)
                vector.append(f)
            else:
                import math
                l = float(row['ltable_' + col])
                r = float(row['rtable_' + col])
                f = math.fabs(l - r) / max(math.fabs(l), math.fabs(r), 1)
                if f >= -1 and f <= 1:
                    f = f
                else:
                    f = 0.0
                vector.append(f)
        X.append(np.array(vector))
        y.append(row[label_column])
    return np.array(X), np.array(y)


def overlap_summary(df, columns, tokenizer=None):
    if tokenizer is None:
        tokenizer = sm.AlphanumericTokenizer(return_set=True)
    measure = sm.OverlapCoefficient()
    overlap_rates = np.zeros((len(columns), len(columns)), dtype=np.float)
    for _, row in df.iterrows():
        for i in range(0, len(columns)):
            for j in range(0, len(columns)):
                if i != j:
                    overlap_rates[i, j] += measure.get_sim_score(
                        tokenizer.tokenize(str(row[columns[i]])),
                        tokenizer.tokenize(str(row[columns[j]]))
                    ) / len(df)
    overlap_rates = pd.DataFrame(overlap_rates, columns=columns, index=columns)
    return overlap_rates
import fasttext
import py_stringmatching as sm

for task_name in tasks[0:5]:
    print(task_name)
    left, right, train, valid, test, idx, columns = load_dataset(base_dir, task_name)
    left_freq = (left[columns].isna().sum() / len(left))
    right_freq = right[columns].isna().sum() / len(right)
    # left_complement = left_freq[left_freq >= 0.05].index
    # right_complement = right_freq[right_freq >= 0.05].index
    # left_overlap = overlap_summary(left, columns)
    # right_overlap = overlap_summary(right, columns)
    print(left_freq)
    print(right_freq)
    # print(left_overlap)
    # print(right_overlap)
    if True:
        manufacturers = set(left['manufacturer'].fillna('')) | set(right['manufacturer'].fillna(''))
        # - {'', 'standard', 'publisher', 'professional'}
        # manufacturers = {'adobe', 'microsoft', 'avanquest', 'apple', 'emc'}
        tokenizer = sm.WhitespaceTokenizer(return_set=True)
        for _, row in right.iterrows():
            if type(row['manufacturer']) != str:
                final_tok = ''
                for tok in tokenizer.tokenize(row['title']):
                    if tok in manufacturers and len(tok) > len(final_tok):
                        final_tok = tok
                if len(final_tok) > 0:
                    right.loc[_, 'manufacturer'] = final_tok

    # left, right, train, valid, test, idx, columns = load_dataset(base_dir, task_name)
    # samples = pd.concat([train, valid, test])
    # samples.index = np.arange(0, len(samples))
    # print(train.columns)
    # print(left.columns)

    left_prefixed = left.rename(columns=dict((k, 'ltable_' + k) for k in ['id'] + columns))
    right_prefixed = right.rename(columns=dict((k, 'rtable_' + k) for k in ['id'] + columns))

    train = pd.merge(pd.merge(train, left_prefixed), right_prefixed)
    valid = pd.merge(pd.merge(valid, left_prefixed), right_prefixed)
    test = pd.merge(pd.merge(test, left_prefixed), right_prefixed)
    train.to_csv('train.csv')
    valid.to_csv('valid.csv')
    test.to_csv('test.csv')
    # break
    # print(left.isna().sum())
    # print(right.isna().sum())
    print(train.columns)

    # X = []
    # y = []
    # for _, row in train.iterrows():
    #     vector = []
    #     for col in columns:
    #         l = model.get_word_vector(str(row['ltable_' + col]))
    #         r = model.get_word_vector(str(row['rtable_' + col]))
    #         f = (l * r).sum(0) / np.sqrt((l * l).sum(0) * (r * r).sum(0))
    #         vector.append(f)
    #     X.append(np.array(vector))
    #     y.append(row['label'])
    # np.save('X.bin', np.array(X))
    # np.save('y.bin', np.array(y))
    if False:
        model = fasttext.load_model("../wiki.en.bin")
        X, y = feature_extraction(train, columns, model)
        np.save('X', X)
        np.save('y', y)
    else:
        X = np.load('X.npy')
        y = np.load('y.npy')
        X, y = feature_extraction_v2(train, columns, None)

    if False:
        model = fasttext.load_model("../wiki.en.bin")
        tX, ty = feature_extraction(test, columns, model)
        np.save('tX', tX)
        np.save('ty', ty)
    else:
        tX = np.load('tX.npy')
        ty = np.load('ty.npy')
        tX, ty = feature_extraction_v2(test, columns, None)
    # break
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, recall_score, precision_score
    for k in range(2, 20):
        print(k)
        clf = SVC(class_weight={0: 1, 1: k})
        clf.fit(X, y)
        p = clf.predict(X)
        score = clf.score(X, y)
        # print("Score: %s, %s, %s" % (precision_score(y, p), recall_score(y, p), f1_score(y, p)))
        # print("Score: %s" % score)
        tp = clf.predict(tX)
        print("Score: %s, %s, %s" % (precision_score(ty, tp), recall_score(ty, tp), f1_score(ty, tp)))

    break