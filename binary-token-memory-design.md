# Binary Token Memory: A Trainable Hash-Native Embedder

*Design doc, 2026-07-14. Supersedes the block-by-block transformer mapping in
`hash-transformer-design.md`. Inherits all 17 measured tenets from that doc.
This design is the direct consequence of Stages 0-4: the transformer layers
were measured to not earn their keep (tenets #11, #14, #15), and the root
bottleneck was traced to decorrelated SHA token hypervectors (tenet #15).
The fix is to learn binary token codes directly — no transformer, no matmul,
no teacher at inference.*

## What this is

A retrieval embedder whose entire inference path is:

```
text → tokenize → lookup binary token codes → IDF-weighted majority bundle → Hamming retrieval
```

No matmul. No transformer layers. No floats. No teacher model. The artifact is
a **binary token code table** (V × d bits) + IDF weights (int8) + a 32-byte seed.
Training is Hebbian — local, parallel, gradient-free — and can run with or without
a teacher.

## What was measured (the evidence this design rests on)

| Stage | What was tested | Result | Tenet |
|---|---|---|---|
| 0 | SimHash of teacher doc embeddings → Hamming retrieval | 0.946 rerank@10 at d=16384 | geometry survives 1-bit |
| 1 | SHA token bundle (no learning) | 0.595 rerank | lexical-only baseline |
| 1b | LSH bucket memory (FFN analog) | 0.263 — below input | #14: address-width bottleneck |
| 2 | Float ridge readout on plain bundles | 0.602 — identical to bundle | #15: features are the bottleneck |
| 3 | Token cluster binding (partial injection) | 0.614 rerank (+3%) | semantic injection helps |
| 4 | Semantic bundle + ridge union | **0.780 rerank** | 80% of ceiling |
| 4 | SimHash of teacher (ceiling) | 0.977 | — |

**The chain of evidence:** the teacher's geometry survives binarization (Stage 0).
SHA tokens are decorrelated, so bundles carry only lexical overlap (Stage 1-2).
Even a perfect linear readout can't extract what isn't there (Stage 2, tenet #15).
Injecting semantic signal at the token level helps (Stage 3). The logical endpoint:
**learn binary token codes directly**, carrying full per-token semantic fidelity.

## The governing constraint (unchanged)

SHA-256 has the avalanche property: flip one input bit, ~half the output bits flip.
That makes it the **opposite** of a locality-sensitive hash. SHA-256 is only usable
for generating fixed pseudorandom codes (where you *want* maximal decorrelation) and
for indexing. **Similarity must be carried by learned binary codes, not by SHA.**

This is the key departure from the original design: SHA-256 is no longer on the
similarity-carrying path at all. It generates the SimHash projection matrix (training-
time only) and role/seed vectors. Token codes are **learned**, not derived.

## Dimensioning

- **d = 16384 bits** per token code (2 KB). Stage 0 showed 16384 bits + rerank = 0.946.
- **V = 30522** tokens (nomic's vocabulary, tenet #10).
- Token table: 30522 × 2048 bytes = **~60 MB** (binary, no compression).
- IDF weights: 30522 × 1 byte (int8) = ~30 KB.

## The artifact

| Component | Size | Type | Role |
|---|---|---|---|
| Token code table | 60 MB | binary (V × d bits) | learned semantic token embeddings |
| IDF weights | 30 KB | int8 | learned token salience |
| SimHash projection G | 0 | — | training-time only (derived from seed) |
| Rerank matrix | optional | float32 (d × 100) | small, for shortlist rescoring |
| Seed | 32 bytes | — | regenerates G at training time |

**No floats in the core artifact** (tenet #12). The optional rerank matrix is the
controlled exception (tenet #17: the "FFN block"), used only to rescore a 100-candidate
shortlist — it is not needed for the Hamming index itself.

## Inference

```
1. tokenize(text) → token_ids              [nomic tokenizer, tenet #10]
2. codes = token_table[token_ids]           [binary lookup, V × d bits → n × d bits]
3. weights = idf[token_ids]                 [int8 lookup]
4. bundle = weighted_majority(codes, weights)  [integer vote counters → sign → d bits]
5. candidates = hamming_topk(bundle, index, 100)  [popcount scan, ~2 ms/query native]
6. (optional) rescore: cosine(rerank_W, bundle, candidates) → top-10
```

Steps 1-5 are pure integer/bit operations. Step 6 is optional and uses a small float
matrix on 100 candidates only. Without step 6: raw Hamming retrieval. With step 6:
the deployment pattern validated in Stage 0 (0.946 rerank).

## Training: three modes, one accumulator

All training modes update the same data structure: a **token accumulator** table
`acc[V, d]` of int16 counters. The binary code is `code[t] = sign(acc[t])`. Modes
can be combined; they just add to the same accumulator.

### Mode 1 — Teacher distillation (needs teacher, one pass)

```
G = random_gaussian(d_teacher, d, seed)           # SimHash projection, training-time only
for each doc d_i:
    teacher_code = sign(teacher_emb[d_i] @ G)      # d bits, bipolar {-1,+1}
    for each token t in d_i:
        acc[t] += teacher_code                      # Hebbian: token accumulates teacher signal
```

After one pass, `code[t]` reflects the average semantic neighborhood of documents
containing t. This is what the teacher knows — distilled into bits.

**Why it works:** SimHash preserves cosine similarity as Hamming similarity (Stage 0:
0.946 rerank). Accumulating per-token gives each token the centroid of its context
distribution. Two tokens that appear in similar documents accumulate similar signals
→ similar codes. This is binary word2vec via distillation.

**Limitation:** frozen snapshot of the teacher. Can't learn what the teacher doesn't know.

### Mode 2 — Self-supervised co-occurrence (no teacher, no gradients)

Binary CBOW: each token's code is pulled toward the bundle of tokens it co-occurs
with, and pushed away from bundles it doesn't.

```
initialize: code[t] = random_bipolar(t, seed)      # or SHA-derived, or from Mode 1

for each epoch:
    for each doc d with tokens {t1..tn}:
        context = weighted_majority(code[t1]..code[tn], idf)   # the doc's context vector

        for each token ti in d:                     # positive: co-occurring
            acc[ti] += lr * context                  # pull toward context

        for k negative samples tj NOT in d:          # negative: non-co-occurring
            acc[tj] -= lr * context                  # push away

    binarize: code[t] = sign(acc[t])                # re-binarize after each epoch
```

**Negative sampling:** pick k tokens proportional to unigram frequency^0.75
(standard word2vec). k=5-10 negatives per positive.

**Learning rate schedule:** start at lr=1.0 (integer, so each doc moves the
accumulator by ±1 per bit), decay by half every epoch. After ~5 epochs, freeze.

**Why it works:** co-occurrence statistics are the same signal word2vec learns.
The bundle (majority vote) is the VSA analog of the CBOW context vector. Pushing
negatives apart is the contrastive signal. The result: tokens that co-occur converge
to similar codes, tokens that don't diverge.

**This trains from scratch without any teacher.** It can run on any text corpus.
The artifact is the same: binary token codes + IDF weights.

### Mode 3 — Retrieval feedback (closes the loop on the actual metric)

Directly optimizes recall@k (tenet #9: train on neighborhoods, not bits).

```
for each doc d:
    bundle_d = weighted_majority(code[t] for t in d, idf)
    retrieved = hamming_topk(bundle_d, index, k=10)       # what the student finds
    true_neighbors = ground_truth(d)                       # from links, clicks, co-citation

    for n in true_neighbors:
        bundle_n = weighted_majority(code[t] for t in n, idf)
        shared = tokens(d) ∩ tokens(n)
        for t in shared:
            acc[t] += lr * bundle_n                         # pull shared tokens together

    for f in (retrieved - true_neighbors):                  # false positives
        bundle_f = weighted_majority(code[t] for t in f, idf)
        shared = tokens(d) ∩ tokens(f)
        for t in shared:
            acc[t] -= lr * bundle_f                         # push shared tokens apart
```

**Ground truth sources:** user click data, co-citation/link graphs, explicit
feedback, or the teacher (as a soft label during distillation, then retired).

**Why it works:** this is a perceptron-style ranking rule on the actual retrieval
metric. When the student retrieves a false positive, the tokens that caused the
false match get pushed apart. When it misses a true neighbor, shared tokens get
pulled together. Over many iterations, the token codes converge to optimize
neighborhood agreement.

### Combining modes

```
# Phase 1: distill (if teacher available)
train_mode_1(corpus, teacher)         # one pass, ~minutes on 899k docs

# Phase 2: self-supervised refinement (no teacher needed)
train_mode_2(corpus, epochs=5)        # ~5 passes, converges

# Phase 3: online retrieval feedback (continuous)
train_mode_3(feedback_stream)         # incremental, runs forever
```

Modes 1 and 2 can be interleaved. Mode 3 runs online as feedback arrives. The
accumulator is persistent; re-binarization happens on a schedule (every N updates
or every epoch).

## The math: why learned binary codes work

**SimHash preserves geometry.** If `cos(u, v) = ρ`, then
`Pr[SimHash(u)_i != SimHash(v)_i] = arccos(ρ)/π`. At d=16384 bits, the Hamming
distance between two codes concentrates tightly around `d × arccos(ρ)/π`. Stage 0
measured this: 0.946 recall@10 — the geometry survives.

**Per-token accumulation is a centroid.** `code[t] = sign(Σ_{d ∋ t} SimHash(emb[d]))`
is the sign of the mean teacher embedding of documents containing t, projected to
binary. Two tokens with similar context distributions accumulate similar centroids
→ similar codes. This is the binary analog of word2vec's learned embeddings.

**Majority bundle = VSA addition.** `bundle = sign(Σ_t idf[t] × code[t])` is the
weighted mean of token codes, binarized. If two docs share semantically similar
tokens (not just identical ones), their bundles are closer in Hamming space than
if tokens were decorrelated. Stage 3 measured this: semantic binding improved
rerank from 0.595 → 0.614. Learned codes should improve it further because they
carry full per-token fidelity, not 1024-cluster approximation.

**Bundling capacity.** The majority of up to ~sqrt(d) ≈ 128 independent bipolar
vectors is recoverable (each vector is ~d/2 + O(sqrt(d)) from the bundle). With
d=16384 and max_tokens=128, we're at the edge. IDF weighting helps: stopwords get
low votes, so the effective bundle size is smaller. Tenet #4 (bundling overflow)
is respected by capping max_tokens and re-thresholding.

## What gets dropped (and why)

| BERT component | Dropped? | Justification |
|---|---|---|
| Token embeddings (768×30522) | **Replaced** by learned binary codes | the whole point |
| 12 transformer layers | **Dropped** | Stage 1b: FFN analog degrades below input (#14). Stage 2: readout can't extract what isn't in features (#15). Depth not earned (#11). |
| Attention (QKV, softmax) | **Dropped** | Bag-of-tokens bundle replaces context aggregation. Tenet #13: position binding destroys retrieval. Context sensitivity is sacrificed; retrievable by Hamming attention in Stage 2 if a task needs it. |
| Layer norm | **Dropped** | Replaced by sign/threshold (binarization is the normalization). |
| Gated FFN | **Dropped** | The "only learnable block" in the old design. Stage 1b/2 proved it doesn't help. Learning moves to token codes. |
| Mean pooling | **Replaced** | By IDF-weighted majority bundle (VSA addition). |

**Net: the 12-layer, 137M-parameter transformer collapses to a learned binary
lookup table + majority vote.** The model's architecture is a table lookup.

## What can and can't be trained

| Component | Trainable? | How |
|---|---|---|
| **Token codes** | **Yes** | Modes 1/2/3 — Hebbian, co-occurrence, or feedback |
| IDF weights | **Yes** | Document frequency (already learned, Stage 1) |
| Bundle aggregation | No | Fixed operation (weighted majority) |
| Rerank matrix | **Yes** | Ridge regression on shortlist (float, small, optional) |
| SimHash projection G | No | Derived from seed (training-time only) |
| Position/context | No (yet) | Needs Hamming attention (future Stage 2, gated) |

## Known limitations (and whether they matter)

### Polysemy — "bank" (river) vs "bank" (money)
One code per token. The teacher's attention disambiguates via context; we can't.

**Mitigation:** store K codes per token (K=4-8), each from a different co-occurrence
cluster. At inference, use all K and take the best Hamming match — a mini
multi-prototype scheme. This is future work; for retrieval, the single-code version
should still beat the SHA baseline because even a polysemous token's centroid is
more informative than a decorrelated random vector.

### No context sensitivity
"dog bites man" = "man bites dog." The bundle is order-free (tenet #13).

**Why it's acceptable:** for document retrieval (drake-memory's use case), word
order is almost irrelevant — two docs with the same vocabulary are about the same
topic regardless of sentence structure. For tasks that need order (NLI, QA), add
Hamming attention as Stage 2, gated on a measured gap.

### No out-of-vocabulary learning
Tokens not in the vocabulary get no code. Subword tokenization (WordPiece/BPE)
covers most cases — nomic already uses this. Truly novel tokens are rare in practice.

### Accumulator saturation
Common tokens ("the", "is") accumulate across 899k docs, potentially saturating
int16 counters. But IDF weighting handles this: stopwords get near-zero vote weight,
so their code quality barely matters. For Mode 2/3, periodic re-binarization +
accumulator reset (keeping the code, resetting the counter) prevents overflow.

## Tenet compliance

| Tenet | Status | How |
|---|---|---|
| #1 learning at hash output | ✓ | Token codes are learned downstream of SHA (SHA generates G and seeds; codes are accumulated) |
| #2 SHA not as LSH | ✓ | SHA generates G (decorrelation wanted) and seeds. Similarity is in learned codes. |
| #3 d >= 8192 | ✓ | d = 16384 |
| #4 bundling overflow | ✓ | IDF weighting + max_tokens=128 cap |
| #9 neighborhood, not bits | ✓ | Mode 3 trains on recall@k directly; Mode 1/2 optimize co-occurrence (a neighborhood proxy) |
| #10 tokenizer match | ✓ | Uses nomic's exact tokenizer |
| #11 depth earned by measurement | ✓ | Depth dropped — measured to not help |
| #12 no floats in artifact | ✓ | Token codes are binary, IDF is int8. Optional rerank matrix is the controlled exception (#17). |
| #13 no position binding | ✓ | Bag-of-unique-tokens bundle |
| #14 no narrow addresses | ✓ | No bucket memory |
| #15 no decorrelated tokens | ✓ | **This is the fix** — tokens are learned, not SHA-derived |
| #16 partial injection | ✓ | N/A — codes are fully learned, not injected. But if combining with SHA init, use partial injection. |
| #17 binary perceptron limit | ✓ | Training is Hebbian accumulation (not perceptron bit-flipping). Binarization is sign(accumulator), not delta-rule. |

## Deployment pattern for drake-memory

```
Index time (batch):
  for each memory:
    text → tokenize → lookup codes → bundle → pack to 2 KB
    store in pgvector-bit or a Rust Hamming index

Query time (per query):
  text → tokenize → lookup codes → bundle → Hamming scan top-100 → float rerank top-10
```

**Storage:** 899k memories × 2 KB = ~1.8 GB (vs 899k × 3 KB = 2.7 GB for float32×768).
A 33% reduction, and Hamming scans are 20-50× faster than cosine on equivalent
hardware (native VPOPCNT).

**Training:** one pass over all memories with the teacher (Mode 1, ~10 minutes),
then 5 epochs of self-supervised co-occurrence (Mode 2, ~30 minutes), then online
feedback (Mode 3, continuous). Re-train when the corpus changes significantly.

## Staged validation plan

Each stage gates the next. No stage proceeds unless the previous measured.

### Stage A — Teacher-distilled token codes (Mode 1)
- Run Mode 1 on 10k docs (same corpus as Stages 1-4)
- Compare: learned codes vs SHA codes vs semantic-cluster codes
- **Gate:** learned codes must beat semantic-cluster bundle (0.614 rerank) and
  approach the ridge-union upper bound (0.780)
- Expected: should approach or exceed 0.780 because it carries full per-token
  fidelity instead of 1024-cluster approximation

### Stage B — Self-supervised training (Mode 2, no teacher)
- Train from scratch with binary CBOW on the same corpus
- Compare: Mode 2 vs Mode 1 vs Mode 1+2 combined
- **Gate:** Mode 2 alone must beat SHA bundle (0.595 rerank). Mode 1+2 must beat
  Mode 1 alone.
- Expected: Mode 2 should reach 0.5-0.7 rerank (co-occurrence is a weaker signal
  than teacher distillation but stronger than random). Mode 1+2 should match or
  beat Mode 1.

### Stage C — Retrieval feedback (Mode 3)
- Use teacher cosine neighbors as ground truth
- Run Mode 3 for 3 iterations
- **Gate:** Mode 1+2+3 must beat Mode 1+2
- Expected: direct optimization of recall@k should close another few percent

### Stage D — Scale and deploy
- Run on full 899k drake-memory corpus
- Measure: recall@10, query latency, index size
- Deploy as Hamming index with float rerank
- **Gate:** recall@10 rerank >= 0.90 (vs teacher 0.977, Stage 0 SimHash 0.946)

## Open research questions

1. **How close can learned binary token codes get to the teacher?** Stage 0 showed
   SimHash of teacher *doc* embeddings = 0.946. Learned *token* codes + bundling
   loses the transformer's context mixing. The gap is the research question.

2. **Does Mode 2 (self-supervised) close the gap to Mode 1 (distillation)?** If
   co-occurrence Hebbian learning matches teacher distillation, the teacher becomes
   optional — a major independence win.

3. **Does Mode 3 (retrieval feedback) beat Mode 1+2?** If direct recall@k
   optimization outperforms proxy signals, the feedback loop is the primary
   training mode.

4. **Multi-prototype codes for polysemy?** K codes per token, best-match at
   inference. How much does it help, and what's the storage cost?

5. **When does Hamming attention earn its keep?** If Stage A-D plateau below 0.90,
   adding one Hamming attention layer (the original design's Stage 2) may close
   the gap — but only if measured.
