BASE_DIR = 'datasets/Structured-raw/'
import pandas as pd

class Dataset(object):
    def __init__(self, name):
        left = pd.read_csv(BASE_DIR + name + '/tableA.csv')
        self.left = left.rename(columns=dict((column, 'left_' + column) for column in left.columns))
        right = pd.read_csv(BASE_DIR + name + '/tableB.csv')
        self.right = right.rename(columns=dict((column, 'right_' + column) for column in right.columns))
        self.train = self.merge_attributes(pd.read_csv(BASE_DIR + name + '/train.csv'))
        self.valid = self.merge_attributes(pd.read_csv(BASE_DIR + name + '/valid.csv'))
        self.test = self.merge_attributes(pd.read_csv(BASE_DIR + name + '/test.csv'))
    def merge_attributes(self, df):
        df = df.rename(columns={'ltable_id': 'left_id', 'rtable_id': 'right_id'})
        df = pd.merge(df, self.left, on=['left_id'])
        df = pd.merge(df, self.right, on=['right_id'])
        df = df.drop(columns=['left_id', 'right_id'])
        df['id'] = df.index
        return df
    def save(self, name=None):
        import os
        if os.path.exists(name):
            raise RuntimeError("Destination directory exists")
        os.mkdir(name)
        self.train.to_csv(name + '/train.csv', index=None)
        self.test.to_csv(name + '/test.csv', index=None)
        self.valid.to_csv(name + '/validation.csv', index=None)
        pd.concat([self.train, self.test, self.valid]).to_csv(name + '/samples.csv', index=None)

import os
os.mkdir('datasets/Structured')
for name in os.listdir(BASE_DIR):
    path = 'datasets/Structured/' + name
    if not os.path.exists(path):
        ds = Dataset(name)
        ds.save(path)
