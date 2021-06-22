from daem.online_matcher.online_matcher import OnlineMatcher

from itertools import zip_longest

import numpy as np
import cupy as cp
import time


class PartialOrderClassifier(OnlineMatcher):
    def __init__(self, features):
        super().__init__(features)

    def acquisition_function(self):
        return super().acquisition_function()

    def update_model(self, positions, labels):
        super().update_model(positions, labels)
        for pos, label in zip_longest(positions, labels):
            if label == 1:
                self.predicted[self.index[(self.features >= self.features[pos]).all(1) & (self.predicted == 0)]] = 1
            elif label == -1:
                self.predicted[self.index[(self.features <= self.features[pos]).all(1) & (self.predicted == 0)]] = -1


class RandomMonotoneClassifier(PartialOrderClassifier):
    def __init__(self, features, samples=None):
        super().__init__(features)
        if samples is None:
            self.samples = np.random.permutation(np.arange(0, len(features)))
        else:
            self.samples = samples
        self.next_sample_id = 0

    def acquisition_function(self):
        while self.next_sample_id < len(self.samples) and self.predicted[self.samples[self.next_sample_id]] != 0:
            self.next_sample_id += 1
        if self.next_sample_id == len(self.predicted):
            return None
        else:
            return (self.samples[self.next_sample_id], 0.5)

    def update_model(self, positions, labels):
        super().update_model(positions, labels)


class MaximumInferenceClassifier(PartialOrderClassifier):
    def __init__(self, features):
        super().__init__(cp.asarray(features))
        width = self.features.shape[1]
        self.predicted = cp.asarray(self.predicted)
        self.index = cp.asarray(self.index)
        self.out = cp.ndarray(self.features.shape[0], dtype=cp.int32)
        self.n_samples = self.features.shape[0]
        self.kernel = cp.RawKernel(r'''
        extern "C" __global__
        void power_benefits_%d(const float * X, const int * state, const long long N, const long long begin, const long long end, int * out) {
            int tid = begin + blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < end) {
                out[tid] = 0;
                if (state[tid] == 0) {
                    for (int i = 0; i < N; i++) {
                        if (state[i] == 0) {
                            out[tid] += (%s);
                            out[tid] += (%s);
                        }
                    }
                }
            }
        }
        ''' % (width,
               ' && '.join('(X[i + %d * N] >= X[tid + %d * N])' % (k, k) for k in range(0, width)),
               ' && '.join('(X[i + %d * N] <= X[tid + %d * N])' % (k, k) for k in range(0, width))
               ), 'power_benefits_%d' % width)

    def acquisition_function(self):
        threads_per_block = 64
        # blocks_per_grid = (self.out.size + (threads_per_block - 1)) // threads_per_block
        # self.kernel((blocks_per_grid,), (threads_per_block,), (self.features, self.predicted, self.n_samples, self.out))
        # cp.cuda.Stream.null.synchronize()


        sep = np.arange(0, self.out.size, 64 * 64 * 4)
        if self.out.size != sep[-1]:
            sep = np.append(sep, self.out.size)

        for i in range(0, len(sep) - 1):
            threads_per_block = 64
            blocks_per_grid = (sep[i + 1] - sep[i] + (threads_per_block - 1)) // threads_per_block
            self.kernel((blocks_per_grid,), (threads_per_block,), (self.features, self.predicted, self.n_samples, sep[i], sep[i + 1], self.out))
            cp.cuda.Stream.null.synchronize()
            #time.sleep(1)
        pos = cp.argmax(self.out * (self.predicted == 0))
        return None if self.out[pos] == 0 else (pos, self.out[pos])

    def update_model(self, positions, labels):
        super().update_model(positions, labels)


class MaximumExpectedMatchClassifier(PartialOrderClassifier):
    def __init__(self, features, prob=None):
        super().__init__(cp.asarray(features))
        width = self.features.shape[1]
        self.predicted = cp.asarray(self.predicted)
        self.index = cp.asarray(self.index)
        self.pred = cp.ndarray(self.features.shape[0], dtype=cp.int32)
        self.succ = cp.ndarray(self.features.shape[0], dtype=cp.int32)
        self.n_samples = self.features.shape[0]
        if prob is not None:
            self.prob = cp.asarray(prob, dtype=cp.float32)
        else:
            self.prob = cp.ones(self.n_samples, dtype=cp.float32)
        self.kernel = cp.RawKernel(r'''
        extern "C" __global__
        void power_benefits_%d(const float * X, const int * state, const long long N, int * pred, int * succ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < N) {
                pred[tid] = 0;
                succ[tid] = 0;
                if (state[tid] == 0) {
                    for (int i = 0; i < N; i++) {
                        if (state[i] == 0) {
                            pred[tid] += (%s);
                            succ[tid] += (%s);
                        }
                    }
                }
            }
        }
        ''' % (width,
               ' && '.join('(X[i + %d * N] >= X[tid + %d * N])' % (k, k) for k in range(0, width)),
               ' && '.join('(X[i + %d * N] <= X[tid + %d * N])' % (k, k) for k in range(0, width))
               ), 'power_benefits_%d' % width)

    def acquisition_function(self):
        threads_per_block = 64
        blocks_per_grid = (self.pred.size + (threads_per_block - 1)) // threads_per_block
        self.kernel((blocks_per_grid,), (threads_per_block,), (self.features, self.predicted, self.n_samples, self.pred, self.succ))
        cp.cuda.Stream.null.synchronize()
        pos = cp.argmax(self.pred * (self.predicted == 0) * self.prob + self.succ * (self.predicted == 0) * (1 - self.prob))
        return None if self.predicted[pos] != 0 else (pos, self.pred[pos] * self.prob[pos] + self.succ[pos] * (1 - self.prob[pos]))

    def update_model(self, positions, labels):
        super().update_model(positions, labels)
