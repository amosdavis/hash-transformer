# Binary Token Memory: pure bitops report

- docs 9997, queries 500, d=16384, mode=all
- 30512 tokens, 5 co-occur epochs, 3 feedback iters

| encoder | recall@10 raw | recall@10 rerank-100 |
|---|---|---|
| A | 0.245 | 0.590 |
| M2(5ep) | 0.099 | 0.249 |
| M2+A | --- | 0.624 |
| M3(3it) | 0.121 | 0.269 |
| M3+A | --- | 0.618 |
| C: SimHash of teacher (ref ceiling) | ~0.70 | ~0.95 |

## Operations (NO matmul, NO floats in training/inference):
- SHA-256: code init, flip-mask generation
- XOR: difference detection, bit-flip application
- AND: mask selection
- popcount: Hamming distance (byte lookup table)
- integer tally + threshold: majority-vote bundling (IDF-weighted)

## Notes
- A: SHA-derived token codes, no learning
- M2: co-occurrence Hebbian bit-flip (self-supervised, no teacher)
- M3: retrieval feedback (perceptron ranking on recall@k)
- Teacher embeddings used ONLY for ground-truth neighbors (offline eval)
- Byte-level flip granularity (faster than bit-level, slightly coarser)
