# Hash-Transformer: BERT Reimagined with SHA-256 Instead of Matmul

*Design doc, 2026-07-13. Source model: `nomic-embed-text-v1.5.Q8_0.gguf` (the drake-memory embedder), full graph extracted from netron.app via Playwright/CDP.*

## Actual source model (whole graph, confirmed)

101 top-level graph nodes: `tokenizer -> EMBEDDING token_embd (q8_0, 768x30522) -> EMBEDDING token_types (768x2) -> LAYER_NORM token_embd_norm -> 12 identical layers -> (mean pooling outside graph)`.

Each of the 12 layers:

```
MULTI_HEAD_ATTENTION  attn_qkv.weight (q8_0, 768x2304), attn_output.weight (q8_0, 768x768)
ADD                   (residual)
LAYER_NORM            attn_output_norm (bias+weight, 768)
MUL_MAT ffn_gate      (q8_0, 768x3072)   \
MUL_MAT ffn_up        (q8_0, 768x3072)    } gated FFN (SwiGLU-style)
MUL_MAT ffn_down      (q8_0, 3072x768)   /
ADD                   (residual)
LAYER_NORM            layer_output_norm (bias+weight, 768)
```

Three facts from the full graph that shape the redesign:

1. **No position-embedding weights** — nomic-bert uses rotary (RoPE), applied inside attention. The hash analog is *bit rotation by position*, which is pleasingly literal.
2. **Gated FFN** — `ffn_down( ffn_gate(x) * ffn_up(x) )`, not a plain 2-layer MLP.
3. **No LM head** — the model ends at the last `layer_output_norm`; output is a pooled sentence embedding used for similarity search. The hash version's natural output is a binary hypervector compared by Hamming distance — i.e., the hash-transformer is an embedder whose output is *already* in retrieval format. For drake-memory this would replace pgvector cosine search with Hamming bitmaps (fast popcount scans, no floats anywhere).

## The governing constraint

SHA-256 has the avalanche property: flip one input bit, ~half the output bits flip. That makes it the **opposite** of a locality-sensitive hash — it can never sit on the similarity-carrying path (you can't hash an activation vector and expect similar activations to stay similar). SHA-256 is only usable for two things:

1. **Generating fixed pseudorandom codes** (embeddings, role vectors, bucket seeds) — where you *want* maximal decorrelation.
2. **Indexing** — mapping an already-quantized pattern to a table address.

Similarity itself must be carried by cheap bit ops: XOR, popcount, majority vote. This lands squarely in hyperdimensional computing (VSA) with bipolar vectors, which is the natural "linear algebra" of hashing:

- XOR = multiplication (binding)
- Majority vote = addition (bundling)
- Hamming distance = dot product

## Dimensioning

Work in **d = 16384 bits** (64 SHA-256 digests, 2 KB per vector) instead of 768 floats. Hamming similarity at 768 bits is too noisy; concentration bounds want d in the 10^4 range.

## Block-by-block mapping

| BERT block | Hash version | Stored params |
|---|---|---|
| tokenizer | unchanged | -- |
| EMBEDDING (768x30522, 23.4M params) | **HASH_EMBED**: `e[t] = signbits( SHA256(t \|\| 0) \|\| ... \|\| SHA256(t \|\| 63) )` — the table is *derived*, never stored | **0** |
| EMBEDDING (768x2, segment) | XOR-bind with `H("seg" \|\| s)` | 0 |
| position (RoPE, no weights in graph) | **Bit-rotate q and k by position** inside attention — the direct Hamming analog of rotary embedding (rotation preserves norm/popcount, encodes relative offset) | 0 |
| LAYER_NORM (768 bias+weight) | **SIGN/THRESHOLD**: collapse integer vote counters back to +/-1; optional learned per-bit int8 threshold plays the role of bias | d x int8 |
| attn_qkv.weight (768x2304) | **Role binding**: `q = x XOR R_Q(l,h)`, `k = x XOR R_K(l,h)`, `v = x XOR R_V(l,h)` where each role `R = SHA256-expand(layer \|\| head \|\| "Q")`. Heads are just different salts | **0** |
| Q.K^T / softmax | **Hamming attention**: `sim(i,j) = d - 2*popcount(q_i XOR k_j)`; softmax -> integer vote weights `1 << (sim/tau)` (bit-shift exponential) or plain top-k winner-take-all | 0 |
| weighted sum of V + attn_output (768x768) | **Weighted majority bundle** of the selected `v_j` (each contributes `votes_j` to per-bit counters), then unbind with output role `XOR R_O(l,h)` | 0 |
| ADD (residual) | Keep residual stream as **integer counters**; bundling = counter addition, so ADD is literally add | -- |
| ffn_gate + ffn_up + ffn_down (gated FFN, 768x3072 x3) | **The only learnable block.** LSH bucket memory with a gate: two slice->table paths — the *up* path retrieves learned bipolar vectors, the *gate* path retrieves learned bit-masks; output = up-bundle AND gate-mask (per-bit multiplexing is the Hamming analog of SwiGLU gating), then a *down* bundling back into the residual counters | B x m x d counters per path — all model capacity lives here |
| output (mean pooling over tokens, outside graph) | **Majority bundle over token hypervectors** -> one d-bit sentence vector; retrieval = Hamming distance (popcount), so the embedding is born in index format | 0 |

Net effect: nomic-embed's ~137M weights collapse to essentially **one learnable gated key-value memory per layer** plus thresholds. Everything else — token embeddings, all attention projections, positions — is regenerated from SHA-256 on demand. The model's "architecture weights" are a 32-byte seed. (The Q8_0 file already conceded 8-bit precision; this design takes the same trajectory to its 1-bit endpoint.)

## Training (no gradients exist through SHA-256)

Backprop is dead here — avalanche means zero usable gradient through any hash. The design survives because **all learnable state sits at hash *outputs*, never upstream of a hash**:

- **FFN buckets**: Hebbian — when training example x should produce y, add y's bits into the counters of the buckets x addresses. This is exactly how sparse distributed memory / product-key memories train, and it's local and parallel.
- **Thresholds/readout**: closed-form (ridge regression on the frozen binary features) — the whole frozen pipeline acts as a reservoir.
- Optionally, evolve the salts — the hash-router territory of scrypt-routed-moe; this design is that idea generalized from the router to the entire transformer.

## Training strategy: distill the existing teacher

Training from scratch is the hard road. The decisive fact: **the teacher already exists** (nomic-embed-text-v1.5, the drake-memory embedder), and an embedder's only job is similarity geometry — a much softer target than exact next-token distributions.

The reframe that makes training tractable:

1. **The hash constraint only binds at inference.** Training happens offline and may use all the floats and matmul it wants. Only the *deployed artifact* must be hash-native (counters + thresholds + a 32-byte seed).
2. **Generate binary targets from the teacher.** Run nomic-embed over a corpus, SimHash each float embedding to d bits (sign of random Gaussian projections — one-time GPU work). SimHash provably preserves cosine similarity as Hamming similarity, so the teacher's retrieval geometry survives binarization. Prior evidence: binary-quantized embedding retrieval (including on Matryoshka-trained nomic models) typically retains ~90-95% of retrieval quality.
3. **Train the hash model to reproduce those codes.** Supervised code-matching, not representation learning: Hebbian bucket updates + closed-form threshold fitting. Every learnable parameter sits at a hash output — tenet #1 respected by construction.

### Staged plan (each stage gates the next)

- **Stage 0 — no research risk:** skip the student model entirely. SimHash nomic's existing embeddings and give drake-memory a Hamming-bitmap index: 16384 bits = 2 KB/memory (vs 3 KB float32x768), popcount scans instead of pgvector cosine. Immediate win on 899k memories, and it *produces the training targets for free*. Measure: recall@k of Hamming top-k vs cosine top-k ground truth, at d = 4096 / 8192 / 16384.
- **Stage 1:** one hash layer (hash embed -> position rotate -> bundle -> bucket FFN), Hebbian-trained against Stage 0's codes. Measure retrieval agreement with the teacher (recall@k overlap).
- **Stage 2:** add Hamming attention and depth *only if* Stage 1 shows a measurable gap that depth closes — A/B'd like bilinear-vs-dense in scrypt-routed-moe.

The main open research question is not "can hashes do inference" (they can, cleanly) or "do binary embeddings work" (known) — it is **whether 12 stacked hash layers earn their depth** over a strong single-layer bundling baseline. Depth must be earned by measurement.

## Hardware and verifiability angles

- The hash workload (embedding derivation, role generation) is pure SHA-256 — a Bitcoin ASIC is, structurally, an embedding-table generator. The similarity path (XOR/popcount) runs at memory bandwidth on CPU with `VPOPCNTDQ`.
- The forward pass is **bit-exact deterministic integer arithmetic** — any node can re-verify an inference by rehashing. That gives proof-of-inference for free, which plugs directly into ai-scrypt-chain's commit-reveal machinery (verify a claimed forward pass by spot-checking hashes instead of trusting floats across BLAS implementations).

## Ways to fail (design tenets)

No change may be made to this design if it causes one of these failures:

1. **Learnable parameter upstream of a hash** -> no training signal, silent capacity loss. All learning stays at hash outputs (buckets, thresholds, readout).
2. **SHA-256 used as an LSH** (hashing raw activations for similarity) -> avalanche destroys locality, attention/retrieval becomes uniform noise. SHA only ever hashes discrete identities or already-quantized slice patterns.
3. **Dimension too small** -> at 768 bits, Hamming similarity std is ~sqrt(d)/2 ≈ 14 bits on a mean of 384; signal drowns. Keep d >= 8192.
4. **Bundling overflow** -> majority of more than ~sqrt(d) vectors is mush. Cap attention top-k and re-threshold (that's what the SIGN blocks are for).
5. **Head collapse** -> identical role construction across heads makes all heads compute the same thing; every role must have a distinct salt including layer index.
6. **Single LSH table** -> bucket collisions blend unrelated concepts; use m independent slice->table paths (same fix as the scrypt-routed-moe crossover gate).
7. **Undefined ties** -> majority vote on an even counter is nondeterministic if broken lazily; break ties with one extra bit from `H(position \|\| layer)` so determinism (and thus verifiability) is preserved.
8. **Expecting smooth regression** -> bipolar codes are great at retrieval/classification, bad at fine-grained continuous outputs. Keep the residual stream as integer counters ("wide mode") and only binarize at block boundaries.
9. **Distilling logits instead of geometry** -> matching exact bit patterns is brittle; the loss must be "same neighbors" — train and evaluate on Hamming-neighborhood agreement with the teacher (recall@k overlap), not per-bit accuracy.
10. **Teacher/student tokenizer drift** -> the student must use nomic's exact tokenizer, or codes are learned against misaligned inputs.
11. **Skipping the Stage 1 baseline** -> building 12 layers before proving one layer beats plain bundling wastes the whole budget; depth must be earned by measurement.
12. **Letting training-time floats leak into the artifact** -> the deliverable is counters + thresholds + a seed; if any step requires shipping a float matrix, the design has failed silently. (The SimHash projection matrix is training-time-only: it binarizes teacher outputs, it is never needed at inference.)
13. **Position-binding in a retrieval encoder** (measured, Stage 1 v1: 0.026 vs 0.257 without) -> binding every token with its position hypervector decorrelates shared vocabulary between documents and collapses similarity; keep document encoding order-free (bag bundle), reserve position binding for tasks that need order.
14. **Address-width bottleneck in bucket memories** (measured, Stage 1b: best 0.092 vs 0.257 input) -> a 10-12 bit LSH address cannot preserve a 16384-bit neighborhood structure; an associative memory whose key is much narrower than its input degrades below the identity map. Do not insert one unless its measured output beats its input.
15. **Decorrelated-token bottleneck** (measured, Stage 2: float ridge upper bound = 0.602, identical to bundle 0.595) -> SHA-256 token hypervectors are maximally decorrelated (avalanche); a bundle of decorrelated tokens encodes only lexical overlap. No doc-level readout — however wide, however well-trained — can extract semantic similarity that does not exist in the features. Semantic signal must be injected at the token level (Fix 3) before any readout can help.
16. **Full-cluster majority destroys token identity** (measured, Stage 3: majority weight ≥ 1 = 0.485 rerank, partial injection 0.25 = 0.614) -> replacing token bits entirely with cluster bits makes all same-cluster tokens identical, collapsing discriminative power. Use partial injection (flip only a fraction p of bits toward the cluster hv) to add semantic bias while preserving token identity.
17. **Binary perceptron does not converge on high-dim binary features** (measured, Stage 2: delta-rule stuck at 51% errors, binarized ridge = 0.559 < bundle 0.609) -> flipping W bits via perceptron updates on 16384-dim binary inputs does not converge; the binarized weight matrix loses too much information. The readout needs float weights to be effective, creating a controlled tension with tenet #12 — the float rerank matrix is the one allowable exception (the "FFN block"), and it is small (4 KB × 100 candidates, not a full model).

## Status

- 2026-07-13: Design written; full source graph extracted from netron.app (101 nodes confirmed).
- 2026-07-13: **Stage 0 complete and validated** (`git_stuff\hash-transformer\stage0_simhash.py`), run against 20k real drake-memory embeddings (512-dim nomic Matryoshka) from Postgres:

  | d (bits) | recall@10 raw | recall@10 rerank-100 | bytes/vec |
  |---|---|---|---|
  | 4096 | 0.635 | 0.932 | 512 |
  | 8192 | 0.706 | 0.940 | 1024 |
  | 16384 | 0.783 | 0.946 | 2048 |

  The teacher's geometry survives 1-bit codes. Deployment pattern is Hamming-shortlist-100 -> float rescore (recall ~0.94); raw Hamming alone is not enough (tenet #9 vindicated: neighborhood agreement was the right metric). Notably 4096 bits + rerank already hits 0.932 at 512 bytes/vec — 6x smaller than float32x512. Codes saved as Stage 1 training targets in `results/codes_d*.npy`.

- 2026-07-14: **Stage 1 run and measured** (`stage1_student.py` + `stage1b_readout.py`, 10k docs, 500 held-out queries; recall@10 raw / rerank-100 vs teacher-cosine ground truth):

  | encoder | raw | rerank-100 |
  |---|---|---|
  | A: SHA bag-of-tokens bundle, IDF votes, no learning | 0.257 | 0.595 |
  | B: + Hebbian LSH-bucket memory (best of 15-config sweep: m=64, 10-bit addr, 3-epoch delta rule) | 0.092 | 0.263 |
  | A union B shortlists, float-reranked | — | 0.656 |
  | C: SimHash of teacher (ceiling) | 0.698 | 0.981 |

  Findings, in tenet terms:
  - **Position binding destroys retrieval** (v1 scored 0.026): XOR-binding tokens with positions makes shared vocabulary contribute uncorrelated vectors. Doc-level retrieval wants a bag-of-unique-tokens bundle. -> new tenet #13.
  - **A pure hash-derived encoder with zero training reaches 60% of the teacher's top-10** (0.595 rerank). SHA token hypervectors + integer IDF votes + majority is a real embedder on its own.
  - **The LSH-bucket memory is a lossy bottleneck, not a gain**: 10-12 address bits per table cannot preserve 16384-bit neighborhoods; every swept config scored far below its own input. Delta-rule epochs and more tables help monotonically (0.038 -> 0.092 raw) but never approach the bundle. Tenet #11 did its job: the complexity is not earned; do not stack this layer.
  - The union combiner shows the learned component carries *some* complementary semantic signal (0.595 -> 0.656) — real but small.
  - **Open gap to close: 0.595 -> 0.98.** Candidate next moves, in order of promise: (a) a learned binary readout matrix (16384 -> 4096 popcount-threshold, delta-rule trained — bit ops at inference but "matmul-shaped": 8 MB of stored bits; needs a design decision on whether it honors the spirit of no-matmul), (b) evolution-searched address bit-selections instead of random slices (discrete search, no gradients — the scrypt-routed-moe salt-evolution playbook), (c) accept the bundle encoder + rerank as the product and stop.

- 2026-07-14: **Stages 2-4: wide readout, semantic binding, combined pipeline** (`stage2_readout.py`, `stage3_semantic.py`, `stage4_combined.py`):

  **Stage 2 — wide binary readout (Fix 1):** Implemented full-width popcount-threshold readout (replaces failed narrow bucket memory). Tested delta-rule perceptron and binarized ridge. KEY FINDING: float ridge regression (upper bound of any linear readout on plain bundle features) = raw 0.266, rerank 0.602 — identical to bundle baseline (0.257/0.595). The bundle features ARE the bottleneck, not the readout. No doc-level readout can manufacture semantic similarity from decorrelated SHA tokens. -> new tenet #15.

  **Stage 3 — token-level semantic binding (Fix 3):** K-means cluster teacher embedding centroids to assign tokens to semantic clusters. Enriched token hv = partial injection of cluster hv (fraction p of bits set to cluster value). Sweep over 64-1024 clusters × {0.1, 0.25, 0.5} injection:

  | n_clusters | weight | raw | rerank |
  |---|---|---|---|
  | 256 (plain bundle baseline) | — | 0.257 | 0.595 |
  | 512 | 0.25 | 0.246 | 0.613 |
  | 1024 | 0.25 | 0.249 | 0.614 |

  Best: 1024 clusters, 0.25 injection = 0.614 rerank (+3% over plain). Artifact = cluster_ids (int16 array, ~60 KB for 30k tokens) + 32-byte seed. No floats in inference artifact (tenet #12 ✓). -> new tenet #16.

  **Stage 4 — semantic bundle + readout (Fix 3 + Fix 1):** Ridge readout on semantic bundles:

  | encoder | raw | rerank-100 | union A rerank |
  |---|---|---|---|
  | A: semantic bundle (no learning) | 0.245 | 0.609 | — |
  | FR: float ridge readout | 0.302 | 0.656 | 0.780 |
  | BR: binarized ridge (Hamming) | 0.212 | 0.559 | 0.683 |
  | C: SimHash of teacher (ceiling) | 0.711 | 0.977 | — |

  The semantic features DO carry extractable signal: ridge upper bound goes from 0.602 (plain) to 0.656 (semantic), and the union of semantic bundle + float ridge reaches **0.780** — up from 0.656 (plain bundle union). The student now reaches 80% of the ceiling.

  Findings, in tenet terms:
  - **Decorrelated tokens are the root bottleneck** (tenet #15): the float ridge upper bound on plain bundles = 0.602, identical to the bundle. No readout can extract what isn't there. Semantic binding at the token level is the necessary fix.
  - **Partial injection > majority binding**: full majority (cluster weight ≥ 1) destroys token identity (0.485 rerank); partial injection (25% of bits) preserves identity while adding semantic signal (0.614). -> new tenet #16.
  - **Binary perceptron does not converge on high-dim binary features** (delta-rule stuck at 51% errors across epochs): the binarized ridge (BR) scores below the bundle (0.559 vs 0.609). Float weights are needed for the readout to work. This creates a tension with tenet #12 — resolved by noting that the readout is the "FFN" block (the one learnable block in the design), and the deployment artifact is the semantic bundle (hash-native) + a small float rerank matrix, not a full float model.
  - **The hash-native student is a strong shortlist generator**: semantic bundle + Hamming shortlist-100 + float rerank of 100 candidates = 0.780 recall@10. The student does not need teacher floats at inference (only the small rerank matrix does), and the Hamming index is 2 KB/vec.
