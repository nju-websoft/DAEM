#define SWAP(x, y, h) h = x; x = y; y = h;
typedef unsigned int  ui;

extern "C" __global__
#ifdef PARAMTER_k
void topk_##DIMENSION##_##PARAMTER_k(const float * vectors, ui n, ui k, ui d, float * pdists, ui * pidxes) {
#else
#define PARAMTER_k k
void topk_##DIMENSION    (const float * vectors, ui n, ui k, ui d, float * dists,  ui * idxes ) {
#endif
    ui ThrPerBlk = blockDim.x;
    ui Mybid = blockIdx.x;
    ui MYtid = threadIdx.x;
    ui MYgtid = ThrPerBlk * Mybid + MYtid;
    float dist1, dist2;
    ui idx1, idx2;
    ui i, j, l, r;
    float vec1[DIMENSION];
    if (MYgtid < n) {
#ifdef PARAMTER_k
        float dists[PARAMTER_k];
        ui idxes[PARAMTER_k];
#else
        dists += MYgtid * PARAMTER_k;
        idxes += MYgtid * PARAMTER_k;
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
                for (j = 0; j < PARAMTER_k; ) {
                    l = j + j + 1;
                    r = l + 1;
                    if (r < PARAMTER_k) {
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
                    } else if (l < PARAMTER_k) {
                        if (dists[j] > dists[l]) {
                            SWAP(dists[j], dists[l], dist1)
                            SWAP(idxes[j], idxes[l], idx1)
                            j = l;
                        } else {
                            break;
                        }
                    } else {
                        j = PARAMTER_k;
                    }
                }
            }
        }
#ifdef PARAMTER_k
        for (j = 0; j < PARAMTER_k; j++) {
            pdists[MYgtid * PARAMTER_k + j] = dists[j];
            pidxes[MYgtid * PARAMTER_k + j] = idxes[j];
        }
#endif
    }
}