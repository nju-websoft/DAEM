import numpy as np


class OnlineMatcher(object):
    def __init__(self, features):
        self.features = features
        self.predicted = np.zeros(len(features), dtype=np.int32)
        self.index = np.arange(0, len(features))
        self.question_cnt = 0

    def acquisition_function(self):
        unresolved = np.arange(0, len(self.predicted))[self.predicted == 0]
        return None if len(unresolved) == 0 else (unresolved.min(), 1.0)

    def get_quality(self, y_labels):
        positives = np.sum(self.predicted == 1)
        true_positives = np.sum((self.predicted == 1) & (self.predicted == y_labels))
        return np.array([self.question_cnt, true_positives, positives], dtype=np.int)

    def update_model(self, positions, labels):
        self.question_cnt += len(positions)
        self.predicted[positions] = labels
