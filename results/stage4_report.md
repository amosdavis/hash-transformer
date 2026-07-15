# Stage 4 report: semantic bundle + readout (stage3 (semantic bundles))

- docs 9987, queries 500, d_s=16384, d_t=4096

| encoder | recall@10 raw | recall@10 rerank-100 |
|---|---|---|
| A: bundle (no learning) | 0.245 | 0.609 |
| R: ridge readout (float upper bound) | 0.302 | 0.656 |
| R union A | — | 0.780 |
| C: SimHash of teacher (ceiling) | 0.711 | 0.977 |

## Decision

- Ridge readout IMPROVES over bundle: 0.609 -> 0.656
  The semantic features DO carry extractable signal (unlike plain bundles).
  Gap to ceiling: 0.656 -> 0.977
