# hash-transformer

SHA-256 / Hamming-space reimagining of nomic-embed-text-v1.5 (the drake-memory embedder).
Design doc + failure tenets: `~/files/hash-transformer-design.md` — no change may violate the
"Ways to fail" list there.

## Stages

- **Stage 0** (`stage0_simhash.py`): SimHash drake-memory's existing 512-dim nomic embeddings
  into d-bit binary codes; measure Hamming recall@k vs float cosine ground truth at
  d = 4096 / 8192 / 16384, plus scan speed and memory footprint. No student model —
  this validates the geometry and produces training targets for Stage 1.
- **Stage 1** (`stage1_student.py` + `stage1b_readout.py`): one hash layer (SHA embed →
  bundle → bucket FFN), Hebbian-trained against Stage 0 codes. Bucket memory failed
  (tenet #14: address-width bottleneck).
- **Stage 2** (`stage2_readout.py`): wide binary popcount-threshold readout (Fix 1) +
  neighborhood-preserving objective (Fix 2) + feature selection (Fix 4). Discovered
  the decorrelated-token bottleneck (tenet #15): float ridge upper bound on plain
  bundles = 0.602, identical to bundle baseline.
- **Stage 3** (`stage3_semantic.py`): token-level semantic binding (Fix 3). K-means
  cluster teacher embedding centroids → assign tokens to semantic clusters → partial
  bit injection of cluster hv into token hv. Best: 1024 clusters, 0.25 injection =
  0.614 rerank (+3% over plain 0.595).
- **Stage 4** (`stage4_combined.py`): semantic bundle + ridge readout (Fix 3 + Fix 1).
  Float ridge on semantic bundles = 0.656 rerank, union with bundle = **0.780**
  (vs plain bundle union 0.656, ceiling 0.977). Student reaches 80% of ceiling.
- **Stage 2** (gated on Stage 1): Hamming attention + depth, only if measurement earns it.

## Run

```
python stage0_simhash.py            # uses DM_DATABASE_URL or postgres://postgres@localhost:5432/drake_memory
```

Outputs report to `results/stage0_report.md` and packed codes to `results/`.
