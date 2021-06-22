#define SWAP(x, y, h) h = x; x = y; y = h;
#define GLUE_HELPER(x, y) x##y
#define GLUE(x, y) GLUE_HELPER(x, y)
#define GLUE4_HELPER(x, y, z, w) x##y##z##w
#define GLUE4(x, y, z, w) GLUE4_HELPER(x, y, z, w)
typedef unsigned int  ui;

extern "C" 
__global__
#ifdef PARAMETER_K
void GLUE4_HELPER(topk_, DIMENSION, _, PARAMETER_K)(const float * vectors, ui n, float * pdists, ui * pidxes) {
    const ui k = PARAMETER_K;
#else
void GLUE(topk_, DIMENSION)(const float * vectors, ui n, ui k, float * dists,  ui * idxes) {
#endif
    const ui d = DIMENSION;
    ui ThrPerBlk = blockDim.x;
    ui Mybid = blockIdx.x;
    ui MYtid = threadIdx.x;
    ui MYgtid = ThrPerBlk * Mybid + MYtid;
    float dist1, dist2;
    ui idx1, idx2;
    ui i, j, l, r;
    float vec1[DIMENSION];
    if (MYgtid < n) {
#ifdef PARAMETER_K
        float dists[k];
        ui idxes[k];
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
                for (j = 0; j < k; ) {
                    l = j + j + 1;
                    r = l + 1;
                    if (r < k) {
                        if (dists[l] < dists[r]) {
                            // smallest is l
                            if (dists[j] > dists[l]) {
                                dists[j] = dists[l]; idxes[j] = idxes[l]; j = l;
                            } else {
                                break;
                            }
                        } else {
                            // smallest is r
                            if (dist1 > dists[r]) {
                                dists[j] = dists[r]; idxes[j] = idxes[r]; j = r;
                            } else {
                                break;
                            }
                        }
                    } else if (l < k) {
                        if (dist1 > dists[l]) {
                            dists[j] = dists[l]; idxes[j] = idxes[l]; j = l;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                dists[j] = dist1;
                idxes[j] = i;
            }
        }
#ifdef PARAMETER_K
        for (j = 0; j < k; j++) {
            pdists[MYgtid * k + j] = dists[j];
            pidxes[MYgtid * k + j] = idxes[j];
        }
#endif
    }
}