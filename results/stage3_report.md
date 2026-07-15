# Stage 3 report: token-level semantic binding (Fix 3)

- docs 9987, queries 500 (held out), d_s 16384, d_t 4096
- 7946 tokens with semantic centroids (min_docs=3)
- base config: 256 clusters, cluster_weight=0.25

## Main results

| encoder | recall@10 raw | recall@10 rerank-100 |
|---|---|---|
| D: semantic bundle (token clustering, no readout) | 0.245 | 0.609 |
| C: SimHash of teacher (ceiling) | 0.707 | 0.977 |

## Cluster count sweep

| n_clusters | cluster_weight | raw | rerank |
|---|---|---|---|
| 64 | 0.1 | 0.245 | 0.607 |
| 64 | 0.25 | 0.240 | 0.578 |
| 64 | 0.5 | 0.198 | 0.483 |
| 128 | 0.1 | 0.246 | 0.603 |
| 128 | 0.25 | 0.242 | 0.605 |
| 128 | 0.5 | 0.214 | 0.552 |
| 256 | 0.1 | 0.248 | 0.601 |
| 256 | 0.25 | 0.245 | 0.609 |
| 256 | 0.5 | 0.217 | 0.565 |
| 512 | 0.1 | 0.244 | 0.601 |
| 512 | 0.25 | 0.246 | 0.613 |
| 512 | 0.5 | 0.233 | 0.583 |
| 1024 | 0.1 | 0.245 | 0.602 |
| 1024 | 0.25 | 0.249 | 0.614 |
| 1024 | 0.5 | 0.240 | 0.603 |

## Notes
- Fix 3 injects semantic similarity at the TOKEN level by binding each token's
  SHA hypervector to its cluster's SHA hypervector (majority vote).
- Cluster assignment is learned from teacher embedding centroids (K-means).
- Artifact: cluster_ids (int16 array, ~60 KB for 30k tokens) + 32-byte seed.
  No floats in the inference artifact (tenet #12).
- The teacher embeddings, centroids, and K-means model are training-time-only.
- Root cause addressed: SHA tokens are decorrelated -> bundle had only lexical
  overlap (ridge upper bound = 0.602 rerank). Semantic binding gives tokens in
  the same cluster correlated vectors -> bundle now carries semantic similarity.
