import torch
import pandas as pd
import numpy as np

def performance_raw(outputs, label):
    predicted = (outputs[:, 1] >= 0.5).to('cpu').detach().numpy()
    tp = label[predicted].sum()
    p = predicted.sum()
    r = label.sum()
    return tp / max(p, 1), tp / r, 2 * tp / (p + r)


def performance(net, data, label):
    outputs = net(data)
    predicted = (outputs[:, 1] >= 0.5).to('cpu').detach().numpy()
    tp = label[predicted].sum()
    p = predicted.sum()
    r = label.sum()
    return tp / max(p, 1), tp / r, 2 * tp / (p + r)


def performance_batch(net, batches):
    tp, p, r = 0, 0, 0
    for batch, label in batches:
        predicted = (torch.softmax(net(batch), 1)[:, 1] >= 0.5).to('cpu').detach().numpy()
        label = label.to('cpu').detach().numpy()
        tp += label[predicted].sum()
        p += predicted.sum()
        r += label.sum()
    return tp / max(p, 1), tp / r, 2 * tp / (p + r)


def predict_batch(net, batches):
    with torch.no_grad():
        output = []
        for batch, label in batches:
            predicted = (torch.softmax(net(batch), 1)[:, 1]).to('cpu').detach().numpy()
            output.append(predicted)
    return np.concatenate(output)


def pretty_print_example(dataset, idx):
    e = dataset.examples[idx].__dict__ 
    pair = dict((col, [' '.join(e['left_' + col]), ' '.join(e['right_' + col])]) for col in dataset.canonical_text_fields)
    return pd.DataFrame(pair)


def pretty_print_examples(dataset, indexes):
    cols = dataset.canonical_text_fields
    pairs = []
    header = ['id'] + ['left_' + col for col in cols] + ['right_' + col for col in cols] + ['label']
    for idx in indexes:
        e = dataset.examples[idx].__dict__
        pair = [idx] + [' '.join(e['left_' + col]) for col in cols] + [' '.join(e['right_' + col]) for col in cols] + [e['label']]
        pairs.append(pair)
    return pd.DataFrame(pairs, columns=header)

def copy_example(example, fields):
    from torchtext.data.example import Example
    new_example = Example()
    new_example.id = example.id
    new_example.label = example.label
    for field in fields:
        setattr(new_example, 'left_' + field, getattr(example, 'left_' + field))
        setattr(new_example, 'right_' + field, getattr(example, 'right_' + field))
    return new_example

def suffix(df, column_suffix):
    columns = df.columns
    return df.rename(columns=dict((column, column + column_suffix)
                     for column in columns))


def prefix(df, column_prefix):
    columns = df.columns
    return df.rename(columns=dict((column, column_prefix + column)
                     for column in columns))