import pandas as pd


def inter_ffe_injection(task_name, train, valid, test):
    mappings = {
        'Amazon-Google': [('title', 'manufacturer')],
        'Walmart-Amazon': [('title', 'modelno')]
    }
    if task_name in mappings:
        for src, dst in mappings[task_name]:
            print(src, dst)
            interffe = pd.read_pickle('../results/iffe-%s-%s-%s.pkl' % (task_name, src, dst))
            interffe = dict((tu[src], ' '.join(tu[dst])) for _, tu in interffe.iterrows())
            for part in [train, valid, test]:
                for i in range(0, len(part.examples)):
                    for where in ['left_', 'right_']:
                        src_value = ' '.join(getattr(part.examples[i], where + src))
                        if src_value in interffe and len(getattr(part.examples[i], where + dst)) == 0 and len(interffe[src_value]) > 0:
                            setattr(part.examples[i], where + dst, interffe[src_value].split(' '))


def inter_ffe_injection_entity(task_name, parts):
    mappings = {
        'Amazon-Google': [('title', 'manufacturer')],
        'Walmart-Amazon': [('title', 'modelno')]
    }
    hit = 0
    if task_name in mappings:
        for src, dst in mappings[task_name]:
            print(src, dst)
            interffe = pd.read_pickle('../results/iffe-%s-%s-%s.pkl' % (task_name, src, dst))
            interffe = dict((tu[src], ' '.join(tu[dst])) for _, tu in interffe.iterrows())
            for part in parts:
                for i in range(0, len(part)):
                    src_value = ' '.join(part.loc[i, src])
                    if src_value in interffe and len(part.loc[i, dst]) == 0 and len(interffe[src_value]) > 0:
                        hit += 1
                        part.loc[i, dst] = interffe[src_value]
                part[dst] = part[dst].apply(lambda x: x if type(x) == list else x.split(' '))
    return hit
   