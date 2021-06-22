import pandas as pd
import torch
from daem.models import ffe
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

basedir = './'


task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'iTunes-Amazon', 'Walmart-Amazon']

mappings = {
    'Walmart-Amazon': [('title', 'modelno')],
    'Amazon-Google': [('title', 'manufacturer')]
}

def raw_column_name(col, prefix):
    if col[:len(prefix)] == prefix:
        return col[len(prefix):]

def extract_columns(df):
    columns = set(df.columns) - {'id', 'label'}
    left_columns = [col for col in columns if col.startswith('left_')]
    right_columns = [col for col in columns if col.startswith('right_')]
    left_columns = sorted([raw_column_name(c, 'left_') for c in left_columns])
    right_columns = sorted([raw_column_name(c, 'right_') for c in right_columns])
    assert left_columns == right_columns
    return left_columns


def extract_entities(df):
    columns = extract_columns(df)
    df1 = df[['left_' + col for col in columns]].rename(columns=dict(('left_' + col, col) for col in columns))
    df2 = df[['right_' + col for col in columns]].rename(columns=dict(('right_' + col, col) for col in columns))
    return pd.concat([df1, df2]).drop_duplicates()

iffe = ffe.InterFieldFeatureExtraction(model_target='')

experiment_results = []
results = {}

for exp_rep in range(0, 5):
    for task_name in mappings.keys():
        print(task_name)
        df = pd.read_csv(basedir + '/datasets/Structured/%s/train.csv' % task_name)
        entities = extract_entities(df)
        for src, dst in mappings[task_name]:
            print('%s ==> %s' % (src, dst))
            data = entities[[src, dst]]
            data = data[data[src].notna()]
            test = data[data[dst].isna()]
            train = data[data[dst].notna()]
            train_src = train[src].values
            train_dst = train[dst].values
        # continue
        for model_name, model in [('ours', ffe.InterffeLSTMCNNResidual), ('lstm', ffe.InterffeLSTM), ('lstm-res', ffe.Interffe), ('lstm-cnn', ffe.InterffeLSTMCNN)]:
            iffe.model_target = basedir + './data/cache/models/iffe/%s-%d-%s-%s-%s.pkl' % (model_name, exp_rep, task_name, src, dst)
            output = iffe.run_ffe_model(data, src, dst, model=model)
            result = output[[src, dst]]
            results[model_name] = result
            gold = pd.read_csv(basedir + 'datasets/ground_truth/%s-%s-%s.csv' % (task_name, src, dst))[[src, dst]]
            out = pd.merge(gold, result, on=src)
            out.loc[out[dst + '_x'].isna(), dst + '_x'] = ''
            out[dst + '_y'] = out[dst + '_y'].apply(' '.join)
            acc = (out[dst + '_y'] == out[dst + '_x']).sum() / len(out)
            print(acc)
            experiment_results.append((task_name, src, dst, model_name, exp_rep, acc))
        torch.save(iffe.net.state_dict(), basedir + '/data/results/iffe-%s-%s-%s.pth' % (task_name, src, dst))
        output.to_pickle(basedir + '/data/results/iffe-%s-%s-%s.pkl' % (task_name, src, dst))
df = pd.DataFrame(experiment_results, columns=['dataset', 'src', 'dst', 'model', 'rep', 'f1']).groupby(by=['dataset', 'src', 'dst', 'model'])['f1'].agg(['mean', 'std']).to_csv(basedir + './data/results/iffe.csv')
print(df)

results = {}
iffe = ffe.InterFieldFeatureExtraction(model_target='')
for task_name in mappings.keys():
    print(task_name)
    df = pd.read_csv(basedir + '/datasets/Structured/%s/train.csv' % task_name)
    entities = extract_entities(df)
    for src, dst in mappings[task_name]:
        print('%s ==> %s' % (src, dst))
        data = entities[[src, dst]]
        data = data[data[src].notna()]
        test = data[data[dst].isna()]
        train = data[data[dst].notna()]
        train_src = train[src].values
        train_dst = train[dst].values
    # continue
    for model_name, model in [('ours', ffe.InterffeLSTMCNNResidual)]:
        iffe.model_target = '/tmp/iffe-%s-%d.pkl' % (model_name, 0)
        output = iffe.run_ffe_model(data, src, dst, model=model)
        result = output[[src, dst]]
        results[model_name] = result
        gold = pd.read_csv(basedir + '/ground_truth/%s-%s-%s.csv' % (task_name, src, dst))[[src, dst]]
        out = pd.merge(gold, result, on=src)
        out.loc[out[dst + '_x'].isna(), dst + '_x'] = ''
        out[dst + '_y'] = out[dst + '_y'].apply(' '.join)
        acc = (out[dst + '_y'] == out[dst + '_x']).sum() / len(out)
        print(acc)
        # experiment_results.append((task_name, src, dst, model_name, exp_rep, acc))
    torch.save(iffe.net.state_dict(), basedir + '/results/iffe-%s-%s-%s.pth' % (task_name, src, dst))
    output.to_pickle(basedir + '/results/iffe-%s-%s-%s.pkl' % (task_name, src, dst))
    output.to_csv(basedir + '/results/iffe-%s-%s-%s.csv' % (task_name, src, dst))