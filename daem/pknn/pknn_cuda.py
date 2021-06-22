import cupy as cp

KNN_CODE_TEMPLATE = r"""
#define SWAP(x, y, h) h = x; x = y; y = h;
typedef unsigned int  ui;

extern "C" __global__
#if {k} > 0
void topk_{dimension}_{k}(const float * vectors, ui n, ui k, ui d, float * pdists, ui * pidxes) {
#else
void topk_{dimension}    (const float * vectors, ui n, ui k, ui d, float * dists,  ui * idxes ) {
#endif
    ui ThrPerBlk = blockDim.x;
    ui Mybid = blockIdx.x;
    ui MYtid = threadIdx.x;
    ui MYgtid = ThrPerBlk * Mybid + MYtid;
    float dist1, dist2;
    ui idx1, idx2;
    ui i, j, l, r;
    float vec1[{dimension}];
    if (MYgtid < n) {
#if {k} > 0
        float dists[{k}];
        ui idxes[{k}];
#else
        dists += MYgtid * k;
        idxes += MYgtid * k;
#endif
        for (j = 0; j < k; j++) {
            dists[j] = -1.0e5;
            idxes[j] = 0;
        }
        // vec1 = (float *)malloc(d * sizeof(float));
        memcpy(vec1, vectors + MYgtid * d, d * sizeof(float));
        for (i = 0; i < n; i++) {
            dist1 = 0.0f;
            for (j = 0; j < d; j++) {
                dist1 += vec1[j] * vectors[i * d + j];
            }

            if (dist1 > dists[0]) {
                dists[0] = dist1;
                idxes[0] = i;
                for (j = 0; j < k; ) {
                    l = j + j + 1;
                    r = l + 1;
                    if (r < k) {
                        if (dists[l] < dists[r]) {
                            // smallest is l
                            if (dists[j] > dists[l]) {
                                SWAP(dists[j], dists[l], dist1)
                                SWAP(idxes[j], idxes[l], idx1)
                                j = l;
                            } else {
                                break;
                            }
                        } else {
                            // smallest is r
                            if (dists[j] > dists[r]) {
                                SWAP(dists[j], dists[r], dist1)
                                SWAP(idxes[j], idxes[r], idx1)
                                j = r;
                            } else {
                                break;
                            }
                        }
                    } else if (l < k) {
                        if (dists[j] > dists[l]) {
                            SWAP(dists[j], dists[l], dist1)
                            SWAP(idxes[j], idxes[l], idx1)
                            j = l;
                        } else {
                            break;
                        }
                    } else {
                        j = k;
                    }
                }
            }
        }
#if {k} > 0
        for (j = 0; j < k; j++) {
            pdists[MYgtid * k + j] = dists[j];
            pidxes[MYgtid * k + j] = idxes[j];
        }
#endif
    }
}
"""

KNN_LEFT_TO_RIGHT_CODE_TEMPLATE = r"""
#define SWAP(x, y, h) h = x; x = y; y = h;
typedef unsigned int  ui;

extern "C" __global__
#if {k} > 0
void topk_l2r_{dimension}_{k}(const float * vectors, ui n, const float * vectors1, ui n1, ui k, ui d, float * pdists, ui * pidxes) {
#else
void topk_l2r_{dimension}    (const float * vectors, ui n, const float * vectors1, ui n1, ui k, ui d, float * dists,  ui * idxes ) {
#endif
    ui ThrPerBlk = blockDim.x;
    ui Mybid = blockIdx.x;
    ui MYtid = threadIdx.x;
    ui MYgtid = ThrPerBlk * Mybid + MYtid;
    float dist1, dist2;
    ui idx1, idx2;
    ui i, j, l, r;
    float vec1[{dimension}];
    if (MYgtid < n) {
#if {k} > 0
        float dists[{k}];
        ui idxes[{k}];
#else
        dists += MYgtid * k;
        idxes += MYgtid * k;
#endif
        for (j = 0; j < k; j++) {
            dists[j] = -1.0e5;
            idxes[j] = 0;
        }
        // vec1 = (float *)malloc(d * sizeof(float));
        memcpy(vec1, vectors + MYgtid * d, d * sizeof(float));
        for (i = 0; i < n1; i++) {
            dist1 = 0.0f;
            for (j = 0; j < d; j++) {
                dist1 += vec1[j] * vectors1[i * d + j];
            }

            if (dist1 > dists[0]) {
                dists[0] = dist1;
                idxes[0] = i;
                for (j = 0; j < k; ) {
                    l = j + j + 1;
                    r = l + 1;
                    if (r < k) {
                        if (dists[l] < dists[r]) {
                            // smallest is l
                            if (dists[j] > dists[l]) {
                                SWAP(dists[j], dists[l], dist1)
                                SWAP(idxes[j], idxes[l], idx1)
                                j = l;
                            } else {
                                break;
                            }
                        } else {
                            // smallest is r
                            if (dists[j] > dists[r]) {
                                SWAP(dists[j], dists[r], dist1)
                                SWAP(idxes[j], idxes[r], idx1)
                                j = r;
                            } else {
                                break;
                            }
                        }
                    } else if (l < k) {
                        if (dists[j] > dists[l]) {
                            SWAP(dists[j], dists[l], dist1)
                            SWAP(idxes[j], idxes[l], idx1)
                            j = l;
                        } else {
                            break;
                        }
                    } else {
                        j = k;
                    }
                }
            }
        }
#if {k} > 0
        for (j = 0; j < k; j++) {
            pdists[MYgtid * k + j] = dists[j];
            pidxes[MYgtid * k + j] = idxes[j];
        }
#endif
    }
}
"""

RANKING_LEFT_TO_RIGHT_CODE_TEMPLATE = r"""
typedef unsigned int  ui;

extern "C" __global__
void ranking_{dimension}(const float * vectors, const ui * answers, ui n, const float * vectors1, ui n1, ui d, ui * ranks) {
    ui ThrPerBlk = blockDim.x;
    ui Mybid = blockIdx.x;
    ui MYtid = threadIdx.x;
    ui MYgtid = ThrPerBlk * Mybid + MYtid;
    float dist1, dist2;
    ui idx1, idx2;
    ui i, j, l, r;
    float vec1[{dimension}];
    if (MYgtid < n) {
        memcpy(vec1, vectors + MYgtid * d, d * sizeof(float));
        
        dist2 = 0.0f;
        for (j = 0; j < d; j++) {
            dist2 += vec1[j] * vectors1[answers[MYgtid] * d + j];
        }

        for (i = 0; i < n1; i++) {
            if (i == answers[MYgtid]) {
                continue;
            }
            dist1 = 0.0f;
            for (j = 0; j < d; j++) {
                dist1 += vec1[j] * vectors1[i * d + j];
            }
            
            if (dist1 > dist2) {
                ranks[MYgtid] += 1;
            }
        }
    }
}
"""

pairwise_k_nearest_neighbors_unordered_kernels = {}


def pairwise_k_nearest_neighbors_unordered(vectors, k, threads_per_block = 64):
    count = vectors.shape[0]
    dimension = vectors.shape[1]
    if (dimension, k) not in pairwise_k_nearest_neighbors_unordered_kernels:
        if dimension + k + k < 1000: # in turing gpu, each thread can use up to 4k register file in average
            pairwise_k_nearest_neighbors_unordered_kernels[(dimension, k)] = cp.RawKernel(KNN_CODE_TEMPLATE.replace('{dimension}', str(dimension)).replace('{k}', str(k)), "topk_%d_%d" % (dimension, k))
        else:
            if (dimension, 0) not in pairwise_k_nearest_neighbors_unordered_kernels:
                pairwise_k_nearest_neighbors_unordered_kernels[(dimension, 0)] = cp.RawKernel(KNN_CODE_TEMPLATE.replace('{dimension}', str(dimension)).replace('{k}', str(0)), "topk_%d" % (dimension))
            pairwise_k_nearest_neighbors_unordered_kernels[(dimension, k)] = pairwise_k_nearest_neighbors_unordered_kernels[(dimension, 0)]
    kernel = pairwise_k_nearest_neighbors_unordered_kernels[(dimension, k)]
    blocks_per_grid = (count + threads_per_block - 1) // threads_per_block
    indexes = cp.zeros((count, k), dtype=cp.uint32)
    products = cp.zeros((count, k), dtype=cp.float32)
    kernel((blocks_per_grid,), (threads_per_block,), (cp.asarray(vectors, dtype=cp.float32), count, k, dimension, products, indexes))
    return cp.asnumpy(indexes), cp.asnumpy(products)

pairwise_k_nearest_neighbors_l2r_unordered_kernels = {}


def pairwise_k_nearest_neighbors_l2r_unordered(vectors1, vectors2, k, threads_per_block = 64):
    assert vectors1.shape[1] == vectors2.shape[1]
    dimension = vectors1.shape[1]
    count = vectors1.shape[0]
    if (dimension, k) not in pairwise_k_nearest_neighbors_l2r_unordered_kernels:
        if dimension + k + k < 1000: # in turing gpu, each thread can use up to 4k register file in average
            pairwise_k_nearest_neighbors_l2r_unordered_kernels[(dimension, k)] = cp.RawKernel(KNN_LEFT_TO_RIGHT_CODE_TEMPLATE.replace('{dimension}', str(dimension)).replace('{k}', str(k)), "topk_l2r_%d_%d" % (dimension, k))
        else:
            if (dimension, 0) not in pairwise_k_nearest_neighbors_l2r_unordered_kernels:
                pairwise_k_nearest_neighbors_l2r_unordered_kernels[(dimension, 0)] = cp.RawKernel(KNN_LEFT_TO_RIGHT_CODE_TEMPLATE.replace('{dimension}', str(dimension)).replace('{k}', str(0)), "topk_l2r_%d" % (dimension))
            pairwise_k_nearest_neighbors_l2r_unordered_kernels[(dimension, k)] = pairwise_k_nearest_neighbors_l2r_unordered_kernels[(dimension, 0)]
    kernel = pairwise_k_nearest_neighbors_l2r_unordered_kernels[(dimension, k)]
    blocks_per_grid = (count + threads_per_block - 1) // threads_per_block
    indexes = cp.zeros((count, k), dtype=cp.uint32)
    products = cp.zeros((count, k), dtype=cp.float32)
    kernel((blocks_per_grid,), (threads_per_block,), (cp.asarray(vectors1, dtype=cp.float32), vectors1.shape[0], cp.asarray(vectors2, dtype=cp.float32), vectors2.shape[0], k, dimension, products, indexes))
    return cp.asnumpy(indexes), cp.asnumpy(products)

ranking_l2r_kernels = {}


def ranking_l2r(vectors1, vectors2, answers, threads_per_block = 64):
    assert vectors1.shape[1] == vectors2.shape[1]
    assert vectors1.shape[0] == answers.shape[0]
    dimension = vectors1.shape[1]
    count = vectors1.shape[0]
    if dimension not in ranking_l2r_kernels:
        ranking_l2r_kernels[dimension] = cp.RawKernel(RANKING_LEFT_TO_RIGHT_CODE_TEMPLATE.replace('{dimension}', str(dimension)), "ranking_%d" % (dimension))
    kernel = ranking_l2r_kernels[dimension]
    blocks_per_grid = (count + threads_per_block - 1) // threads_per_block
    ranks = cp.zeros(count, dtype=cp.uint32)
    kernel((blocks_per_grid,), (threads_per_block,), (cp.asarray(vectors1, dtype=cp.float32), cp.asarray(answers, dtype=cp.uint32), vectors1.shape[0], cp.asarray(vectors2, dtype=cp.float32), vectors2.shape[0], dimension, ranks))
    return cp.asnumpy(ranks)


def sort_by_product(idx, prod):
    import numpy as np
    order = np.argsort(-prod, 1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(prod, order, axis=1)

if __name__ == "__main__":
    import numpy as np
    import cupy as cp
    # import pknn
    vectors1 = np.random.normal(0, 1, (1000, 200))
    vectors1 /= np.sqrt((vectors1 * vectors1).sum(1)).reshape(vectors1.shape[0], 1)
    assert (np.abs(((vectors1 * vectors1).sum(1) - 1.)) < 1.e-5).all()
    vectors2 = np.random.normal(0, 1, (1500, 200))
    vectors2 /= np.sqrt((vectors2 * vectors2).sum(1)).reshape(vectors2.shape[0], 1)

    idx, prod = pairwise_k_nearest_neighbors_l2r_unordered(vectors1, vectors2, 100)
    cp.cuda.runtime.deviceSynchronize()
    dist = np.matmul(vectors1, vectors2.T)
    print((np.sort(np.partition(-dist, 100, 1)[:, :100], 1) - prod < 1e-5).mean())

    idx, prod = pairwise_k_nearest_neighbors_unordered(vectors1, 100)
    cp.cuda.runtime.deviceSynchronize()
    dist = np.matmul(vectors1, vectors1.T)
    indexes = np.arange(0, vectors1.shape[0] * 100).reshape(vectors1.shape[0], 100)
    print((np.sort(np.partition(-dist, 100, 1)[:, :100], 1) - prod < 1e-5).mean())

    rank = ranking_l2r(vectors1, vectors2, np.arange(0, vectors1.shape[0]))
    cp.cuda.runtime.deviceSynchronize()
    dist = np.matmul(vectors1, vectors2.T)
    print(np.power(((dist > (vectors1 * vectors2[:len(vectors1), :]).sum(1).reshape(vectors1.shape[0], 1))  &
    (1 - np.eye(vectors1.shape[0], vectors2.shape[0], dtype=np.bool))
    ).sum(1) - rank, 2).mean())
