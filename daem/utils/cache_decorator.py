from pathlib import Path
import pandas as pd

class CacheDecoreator(object):
    def __init__(self, base_dir, task_name):
        self.base_dir = Path(base_dir)
        self.task_name = task_name

    def category(self, cate, force=False):
        def real_func(func):
            def wrapper(*args, **kwargs):
                import os
                cache_name = self.base_dir / cate / (self.task_name + '.pkl')
                if not force and cache_name.exists():
                    return pd.read_pickle(str(cache_name))
                else:
                    import time
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    pd.to_pickle(result, str(cache_name))
                    pd.to_pickle(
                        {
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time
                        },
                        str(self.base_dir / cate / (self.task_name + '.time.pkl')))
                return result
            return wrapper
        return real_func

    def read_cache(self, cate):
        return pd.read_pickle(str(self.base_dir / cate / (self.task_name + '.pkl')))

    def read_time(self, cate):
        return pd.read_pickle(str(self.base_dir / cate / (self.task_name + '.time.pkl')))