# Stage 0 report: SimHash fidelity on drake-memory embeddings

- docs: 19990, queries: 500, k: 10, teacher dim: 512 (nomic Matryoshka-truncated)
- cosine brute-force baseline: 0.12s total (0.2 ms/query), float32 footprint 41 MB

| d (bits) | recall@10 raw | recall@10 rerank-100 | bytes/vec | total codes | Hamming scan ms/query |
|---|---|---|---|---|---|
| 4096 | 0.635 | 0.932 | 512 | 10 MB | 48.9 |
| 8192 | 0.706 | 0.940 | 1024 | 20 MB | 96.1 |
| 16384 | 0.783 | 0.946 | 2048 | 41 MB | 195.1 |

Notes:
- Hamming scan here is numpy uint8-lookup popcount (interpreter-bound); a native
  VPOPCNT/AVX2 scan is typically 20-50x faster, while the cosine baseline already
  enjoys optimized BLAS. Fidelity numbers are the point of Stage 0; speed parity
  is a Rust/pgvector-bit follow-up.
- Codes saved to results/codes_d*.npy are the Stage 1 training targets.
- Projection matrix G is derived from seed 42 at training time only (tenet #12).
