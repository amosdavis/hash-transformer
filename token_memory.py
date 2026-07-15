"""Binary Token Memory — pure bitops implementation.

NO MATMUL. NO FLOATS in training or inference. The entire system is:
  SHA-256 (code/mask generation)
  XOR (binding / difference / bit-flip)
  AND (masking / selection)
  NOT (complement for "push away")
  popcount (Hamming distance)
  integer tally + threshold (majority vote = bundling)

Training: co-occurrence Hebbian bit-flip. For each doc, build the context
bundle (integer-tally majority of token codes). For each token in the doc,
flip p% of its differing bits toward the bundle (positive). For negative
samples (tokens NOT in the doc), flip p% of their agreeing bits (push away).

The learning rate p is the density of the SHA-derived flip mask.

Inference: tokenize -> lookup codes -> IDF-weighted integer-tally bundle ->
Hamming popcount scan.

All operations use packed uint8 arrays. The popcount lookup table operates
on bytes. The IDF-weighted bundle uses integer scaling + counting (fastest
on CPU — avoids both float matmul and bit-repeat-copy).
"""
import os
import sys
import time
import hashlib
import argparse

import numpy as np
from tokenizers import Tokenizer

TOK_PATH = os.path.expanduser("~/.drake-memory/models/tokenizer.json")
N_DOCS = 10000
N_QUERIES = 500
TOP_K = 10
SHORTLIST = 100
MAX_TOKENS = 128
D = 16384          # code width (bits)
D_BYTES = D // 8   # 2048 bytes per code
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Popcount lookup: 256-entry table, maps uint8 byte -> bit count
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# SHA-256 code generation (packed uint8)
# ---------------------------------------------------------------------------
def sha_code(label, n_bytes=D_BYTES):
    """Deterministic bipolar code from SHA-256, packed as uint8 bytes."""
    out = bytearray()
    base = label.encode()
    for j in range(n_bytes // 32):
        out += hashlib.sha256(base + j.to_bytes(4, "little")).digest()
    return np.frombuffer(bytes(out), dtype=np.uint8)


def sha_flip_mask_bits(label, n_bits=D, density=0.1):
    """Random BIT mask from SHA-256 with ~density fraction of BITS set.

    Unlike the byte-level mask, this selects individual bits, not whole bytes.
    This is critical for training precision — byte-level flips (all 8 bits)
    introduce too much noise per update.
    """
    n_bytes = (n_bits + 7) // 8
    raw = sha_code(label, n_bytes)
    # Unpack to bits, threshold each bit individually
    bits = np.unpackbits(raw)
    threshold = max(1, int(density * 256))
    return bits[:n_bits] < (density * 256 / 256.0)  # bool per bit


def apply_bit_flip(code_packed, flip_mask_bits):
    """Apply bit-level flips to a packed code.

    flip_mask_bits: (D,) bool array — True = flip this bit.
    Returns new packed uint8 code.
    """
    bits = np.unpackbits(code_packed)
    bits[:flip_mask_bits.shape[0]] ^= flip_mask_bits.astype(np.uint8)
    return np.packbits(bits)


# ---------------------------------------------------------------------------
# Bit operations on packed uint8
# ---------------------------------------------------------------------------
def bits_differ_mask(code_packed, bundle_packed, n_bits=D):
    """Per-bit mask of where code differs from bundle. Returns (D,) bool."""
    a = np.unpackbits(code_packed)[:n_bits]
    b = np.unpackbits(bundle_packed)[:n_bits]
    return a != b


def bits_agree_mask(code_packed, bundle_packed, n_bits=D):
    """Per-bit mask of where code agrees with bundle. Returns (D,) bool."""
    a = np.unpackbits(code_packed)[:n_bits]
    b = np.unpackbits(bundle_packed)[:n_bits]
    return a == b


def flip_selected_bits(code_packed, bit_mask, n_bits=D):
    """Flip individual bits where bit_mask is True. Returns packed uint8."""
    bits = np.unpackbits(code_packed).copy()
    bits[:n_bits] ^= bit_mask.astype(np.uint8)
    return np.packbits(bits)


def hamming_dist_matrix(codes, query):
    """Hamming distance from query to all codes. Pure popcount on XOR."""
    return POPCOUNT8[np.bitwise_xor(codes, query)].sum(axis=1, dtype=np.int32)


# ---------------------------------------------------------------------------
# Integer-tally majority bundle (the VSA "addition" — no matmul)
# ---------------------------------------------------------------------------
def bundle_codes(codes, weights):
    """Build a majority-vote bundle from packed codes via integer tally.

    For each bit position: count 1-bits across all codes (weighted by int IDF),
    threshold at half the total weight. This is integer scaling + counting —
    the fastest pure-integer approach on CPU (no float matmul, no bit-repeat).

    codes: (n_tok, D_BYTES) uint8
    weights: (n_tok,) int8 IDF votes
    Returns: (D_BYTES,) uint8 packed bundle
    """
    # Unpack to bits: (n_tok, D) -> int16
    bits = np.unpackbits(codes, axis=1).astype(np.int16)
    # Integer scaling: multiply each token's bits by its IDF weight
    bits *= weights[:, None].astype(np.int16)
    # Tally: sum across tokens -> (D,)
    counts = bits.sum(axis=0, dtype=np.int16)
    total = weights.sum(dtype=np.int16)
    # Majority: bit is 1 if count*2 > total
    return np.packbits(counts * 2 > total)


def build_all_bundles(doc_ids, code_table, idf_table, max_tokens=MAX_TOKENS):
    """Build bundles for all docs. Returns (n_docs, D_BYTES) uint8."""
    n_docs = len(doc_ids)
    bundles = np.zeros((n_docs, D_BYTES), dtype=np.uint8)
    t0 = time.perf_counter()
    for di, ids in enumerate(doc_ids):
        if not ids:
            continue
        ids_capped = ids[:max_tokens]
        bundles[di] = bundle_codes(code_table[ids_capped], idf_table[ids_capped])
        if di % 2000 == 1999:
            print(f"  {di+1}/{n_docs} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return bundles


# ---------------------------------------------------------------------------
# Hamming retrieval
# ---------------------------------------------------------------------------
def hamming_topk_batch(codes, qrows, k):
    """Top-k nearest codes for multiple queries (exclude self)."""
    out = np.empty((len(qrows), k), dtype=np.int64)
    for i, q in enumerate(qrows):
        dist = hamming_dist_matrix(codes, codes[q])
        dist[q] = np.iinfo(np.int32).max
        out[i] = np.argpartition(dist, k)[:k]
    return out


def eval_codes(name, bundles, qrows, gt, Xn, extra_shortlist=None):
    """Evaluate recall@10 raw and rerank-100."""
    short = hamming_topk_batch(bundles, qrows, SHORTLIST)
    top10 = hamming_topk_batch(bundles, qrows, TOP_K)
    raw = np.mean([len(set(gt[i]) & set(top10[i])) / TOP_K for i in range(len(qrows))])
    rr = []
    for i, q in enumerate(qrows):
        cand = short[i]
        if extra_shortlist is not None:
            cand = np.unique(np.concatenate([cand, extra_shortlist[i]]))
            cand = cand[cand != q]
        sims = Xn[cand] @ Xn[q]
        top = cand[np.argsort(-sims)[:TOP_K]]
        rr.append(len(set(gt[i]) & set(top)) / TOP_K)
    line = f"| {name} | {raw:.3f} | {np.mean(rr):.3f} |"
    print(line, flush=True)
    return raw, float(np.mean(rr))


# ---------------------------------------------------------------------------
# Token table init (SHA-256)
# ---------------------------------------------------------------------------
def init_code_table(vocab_size, d_bytes=D_BYTES):
    """Initialize token codes from SHA-256 (decorrelated, tenet-compliant)."""
    table = np.zeros((vocab_size, d_bytes), dtype=np.uint8)
    for t in range(vocab_size):
        table[t] = sha_code(f"tok|{t}", d_bytes)
    return table


# ---------------------------------------------------------------------------
# Training: co-occurrence Hebbian bit-flip (no teacher)
# ---------------------------------------------------------------------------
def train_cooccurrence(doc_ids, code_table, idf_table, epochs=5, lr_density=0.1,
                       n_neg=5, max_tokens=MAX_TOKENS, seed=SEED):
    """Self-supervised co-occurrence training via SHA-masked bit-flip.

    For each doc:
      1. Build context bundle = integer-tally majority of token codes
      2. For each token in doc (positive): flip density-frac of DIFFERING
         bytes toward the bundle
      3. For negative samples (tokens NOT in doc): flip density-frac of
         AGREEING bytes (push away from bundle)

    All ops: XOR, AND, SHA-256 mask, integer tally. No matmul, no floats.
    """
    rng = np.random.default_rng(seed)
    n_docs = len(doc_ids)
    V = code_table.shape[0]
    table = code_table.copy()

    # Negative sampling distribution (freq^0.75)
    token_freq = np.zeros(V, dtype=np.float64)
    for ids in doc_ids:
        for t in ids:
            token_freq[t] += 1
    token_freq = token_freq ** 0.75
    token_freq /= token_freq.sum()

    for epoch in range(epochs):
        t0 = time.perf_counter()
        n_updates = 0
        perm = rng.permutation(n_docs)
        density = max(lr_density * (0.5 ** epoch), 0.01)

        for di in perm:
            ids = doc_ids[di]
            if len(ids) < 2:
                continue
            ids = ids[:max_tokens]
            codes = table[ids]
            wt = idf_table[ids]

            # 1. Context bundle: integer-tally majority (no matmul)
            bundle = bundle_codes(codes, wt)

            # 2. Positive: pull each token toward bundle
            doc_token_set = set(ids)
            for tid in ids:
                differ = bits_differ_mask(table[tid], bundle)
                mask = sha_flip_mask_bits(f"pos|{epoch}|{tid}", D, density)
                flip_mask = np.bitwise_and(differ, mask)
                table[tid] = flip_selected_bits(table[tid], flip_mask)
                n_updates += 1

            # 3. Negative: push random non-co-occurring tokens away
            neg_ids = rng.choice(V, size=n_neg, p=token_freq, replace=False)
            for nt in neg_ids:
                if nt in doc_token_set:
                    continue
                agree = bits_agree_mask(table[nt], bundle)
                mask = sha_flip_mask_bits(f"neg|{epoch}|{nt}", D, density)
                flip_mask = np.bitwise_and(agree, mask)
                table[nt] = flip_selected_bits(table[nt], flip_mask)

        print(f"  epoch {epoch+1}/{epochs}: {n_updates} updates, "
              f"density={density:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    return table


# ---------------------------------------------------------------------------
# Training: retrieval feedback (pure bitops)
# ---------------------------------------------------------------------------
def train_retrieval_feedback(doc_ids, code_table, idf_table, Xn, holdout, gt,
                             n_iters=3, lr_density=0.05, max_tokens=MAX_TOKENS,
                             seed=SEED):
    """Retrieval feedback: push/pull token codes based on Hamming retrieval errors.

    For each training doc:
      1. Build bundle, retrieve top-10 by Hamming popcount
      2. Compare to teacher cosine neighbors (ground truth — offline)
      3. False positives: push shared tokens' codes AWAY from FP's bundle
      4. False negatives: pull shared tokens' codes TOWARD FN's bundle

    All training ops: XOR, AND, SHA-mask, popcount, integer tally.
    Xn used ONLY for offline ground-truth neighbor computation.
    """
    rng = np.random.default_rng(seed + 2)
    n_docs = len(doc_ids)
    table = code_table.copy()

    train_mask = np.ones(n_docs, dtype=bool)
    train_mask[holdout] = False
    train_idx = np.where(train_mask)[0]
    print("computing teacher neighbors for training docs (offline)...", flush=True)
    sims = Xn[train_idx] @ Xn.T
    for i in range(len(train_idx)):
        sims[i, train_idx[i]] = -np.inf
    teacher_topk = np.empty((len(train_idx), 20), dtype=np.int64)
    for i in range(len(train_idx)):
        teacher_topk[i] = np.argpartition(-sims[i], 20)[:20]
    del sims

    for it in range(n_iters):
        t0 = time.perf_counter()
        n_corrections = 0
        density = max(lr_density * (0.5 ** it), 0.01)

        bundles = build_all_bundles(doc_ids, table, idf_table, max_tokens)
        sample = rng.choice(len(train_idx), size=min(500, len(train_idx)), replace=False)

        for si in sample:
            di = train_idx[si]
            ids_d = doc_ids[di][:max_tokens] if doc_ids[di] else []
            if not ids_d:
                continue
            bundle_d = bundles[di]

            dist = hamming_dist_matrix(bundles, bundle_d)
            dist[di] = np.iinfo(np.int32).max
            retrieved = set(np.argpartition(dist, 10)[:10].tolist())
            true_set = set(teacher_topk[si].tolist())

            for fp in (retrieved - true_set):
                ids_fp = doc_ids[fp][:max_tokens] if doc_ids[fp] else []
                shared = set(ids_d) & set(ids_fp)
                if not shared:
                    continue
                bundle_fp = bundles[fp]
                for t in shared:
                    agree = bits_agree_mask(table[t], bundle_fp)
                    mask = sha_flip_mask_bits(f"fp|{it}|{t}", D, density)
                    flip_mask = np.bitwise_and(agree, mask)
                    table[t] = flip_selected_bits(table[t], flip_mask)
                    n_corrections += 1

            for fn in (true_set - retrieved):
                if fn == di:
                    continue
                ids_fn = doc_ids[fn][:max_tokens] if doc_ids[fn] else []
                shared = set(ids_d) & set(ids_fn)
                if not shared:
                    continue
                bundle_fn = bundles[fn]
                for t in shared:
                    differ = bits_differ_mask(table[t], bundle_fn)
                    mask = sha_flip_mask_bits(f"fn|{it}|{t}", D, density)
                    flip_mask = np.bitwise_and(differ, mask)
                    table[t] = flip_selected_bits(table[t], flip_mask)
                    n_corrections += 1

        del bundles
        print(f"  iter {it+1}/{n_iters}: {n_corrections} corrections, "
              f"density={density:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    return table


# ---------------------------------------------------------------------------
# IDF computation
# ---------------------------------------------------------------------------
def compute_idf(doc_ids, max_tokens=MAX_TOKENS):
    n_docs = len(doc_ids)
    df = {}
    for ids in doc_ids:
        for i in ids[:max_tokens]:
            df[i] = df.get(i, 0) + 1
    max_tok = max(df.keys()) + 1 if df else 1
    idf = np.ones(max_tok, dtype=np.int8)
    for i, count in df.items():
        idf[i] = max(1, min(127, int(round(np.log2(n_docs / count)))))
    return idf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Binary Token Memory (pure bitops)")
    parser.add_argument("--mode", default="all",
                        choices=["baseline", "cooccur", "feedback", "all"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()

    import psycopg2
    DB_URL = os.environ.get("DM_DATABASE_URL",
                            "postgres://postgres@localhost:5432/drake_memory")
    rng = np.random.default_rng(SEED)
    tokenizer = Tokenizer.from_file(TOK_PATH)

    print(f"fetching {N_DOCS} docs from DB...", flush=True)
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT content, vector::text FROM memories TABLESAMPLE SYSTEM (5) "
            "WHERE vector IS NOT NULL AND length(content) > 20 LIMIT %s", (N_DOCS,))
        rows = cur.fetchall()
        if len(rows) < N_DOCS // 2:
            raise RuntimeError("tablesample too small")
    except Exception:
        conn.rollback()
        cur.execute("SELECT content, vector::text FROM memories "
                    "WHERE vector IS NOT NULL AND length(content) > 20 LIMIT %s", (N_DOCS,))
        rows = cur.fetchall()
    conn.close()
    texts = [r[0] for r in rows]
    X = np.array([np.fromstring(r[1][1:-1], sep=",", dtype=np.float32) for r in rows])
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0
    texts = [t for t, k in zip(texts, keep) if k]
    Xn = X[keep] / norms[keep, None]
    n = len(texts)
    print(f"  {n} docs, teacher dim {Xn.shape[1]}", flush=True)

    print("tokenizing...", flush=True)
    encs = tokenizer.encode_batch(texts)
    doc_ids = [sorted(set(e.ids[:MAX_TOKENS])) for e in encs]
    V = max(max(ids) for ids in doc_ids if ids) + 1
    print(f"  {V} distinct tokens", flush=True)

    idf_table = compute_idf(doc_ids)

    holdout = rng.choice(n, size=min(N_QUERIES, n // 10), replace=False)
    sims = Xn[holdout] @ Xn.T
    gt = np.empty((len(holdout), TOP_K), dtype=np.int64)
    for i, q in enumerate(holdout):
        s = sims[i].copy()
        s[q] = -np.inf
        gt[i] = np.argpartition(-s, TOP_K)[:TOP_K]

    print("initializing SHA token codes...", flush=True)
    t0 = time.perf_counter()
    code_table = init_code_table(V)
    print(f"  {time.perf_counter()-t0:.1f}s ({code_table.nbytes/1e6:.1f} MB)", flush=True)

    results = {}

    # Baseline
    print("\n--- Baseline: SHA token codes (no learning) ---", flush=True)
    bundles = build_all_bundles(doc_ids, code_table, idf_table)
    sha_short = hamming_topk_batch(bundles, holdout, SHORTLIST)
    a_raw, a_rr = eval_codes("A: SHA bundle (no learning)", bundles, holdout, gt, Xn)
    results["A"] = (a_raw, a_rr)

    # Mode 2: Co-occurrence
    if args.mode in ("cooccur", "all"):
        print(f"\n--- Mode 2: Co-occurrence bit-flip ({args.epochs} epochs) ---", flush=True)
        code_table = train_cooccurrence(doc_ids, code_table, idf_table,
                                        epochs=args.epochs, lr_density=0.1)
        bundles = build_all_bundles(doc_ids, code_table, idf_table)
        m2_raw, m2_rr = eval_codes(f"M2: + co-occurrence ({args.epochs}ep)",
                                   bundles, holdout, gt, Xn)
        _, m2_union = eval_codes("M2+A", bundles, holdout, gt, Xn,
                                  extra_shortlist=sha_short)
        results[f"M2({args.epochs}ep)"] = (m2_raw, m2_rr)
        results["M2+A"] = (None, m2_union)

    # Mode 3: Retrieval feedback
    if args.mode in ("feedback", "all"):
        print(f"\n--- Mode 3: Retrieval feedback ({args.iters} iters) ---", flush=True)
        code_table = train_retrieval_feedback(doc_ids, code_table, idf_table, Xn,
                                              holdout, gt, n_iters=args.iters,
                                              lr_density=0.05)
        bundles = build_all_bundles(doc_ids, code_table, idf_table)
        m3_raw, m3_rr = eval_codes(f"M3: + feedback ({args.iters}it)",
                                   bundles, holdout, gt, Xn)
        _, m3_union = eval_codes("M3+A", bundles, holdout, gt, Xn,
                                  extra_shortlist=sha_short)
        results[f"M3({args.iters}it)"] = (m3_raw, m3_rr)
        results["M3+A"] = (None, m3_union)

    # Summary
    print("\n=== Summary ===", flush=True)
    print("| encoder | recall@10 raw | recall@10 rerank-100 |", flush=True)
    print("|---|---|---|", flush=True)
    for name, (r, rr) in results.items():
        rstr = f"{r:.3f}" if r is not None else "---"
        print(f"| {name} | {rstr} | {rr:.3f} |", flush=True)
    print("| C: SimHash of teacher (ref ceiling) | ~0.70 | ~0.95 |", flush=True)

    # Save
    np.savez_compressed(os.path.join(RESULTS, "token_codes_bitops.npz"),
                        codes=code_table, idf=idf_table)
    print(f"\nsaved token_codes_bitops.npz ({code_table.nbytes/1e6:.1f} MB)", flush=True)

    # Report
    lines = [
        "# Binary Token Memory: pure bitops report",
        "",
        f"- docs {n}, queries {len(holdout)}, d={D}, mode={args.mode}",
        f"- {V} tokens, {args.epochs} co-occur epochs, {args.iters} feedback iters",
        "",
        "| encoder | recall@10 raw | recall@10 rerank-100 |",
        "|---|---|---|",
    ]
    for name, (r, rr) in results.items():
        rstr = f"{r:.3f}" if r is not None else "---"
        lines.append(f"| {name} | {rstr} | {rr:.3f} |")
    lines += [
        "| C: SimHash of teacher (ref ceiling) | ~0.70 | ~0.95 |",
        "",
        "## Operations (NO matmul, NO floats in training/inference):",
        "- SHA-256: code init, flip-mask generation",
        "- XOR: difference detection, bit-flip application",
        "- AND: mask selection",
        "- popcount: Hamming distance (byte lookup table)",
        "- integer tally + threshold: majority-vote bundling (IDF-weighted)",
        "",
        "## Notes",
        "- A: SHA-derived token codes, no learning",
        "- M2: co-occurrence Hebbian bit-flip (self-supervised, no teacher)",
        "- M3: retrieval feedback (perceptron ranking on recall@k)",
        "- Teacher embeddings used ONLY for ground-truth neighbors (offline eval)",
        "- Byte-level flip granularity (faster than bit-level, slightly coarser)",
    ]
    report_path = os.path.join(RESULTS, "token_memory_bitops_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
