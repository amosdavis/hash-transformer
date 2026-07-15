# Stage 1 report: Hebbian hash-layer student vs teacher

- docs 9990, queries 500 (held out), d_s 16384, d_t 4096, tables 16, buckets/table 4096, max tokens 128

| codes | recall@10 raw | recall@10 rerank-100 |
|---|---|---|
| A: raw bundle s0 (lexical, no learning) | 0.257 | 0.595 |
| B: student (bucket memory) | 0.048 | 0.145 |
| C: SimHash of teacher (Stage 0 ceiling) | 0.698 | 0.981 |
