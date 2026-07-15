"""Stage 2: wide binary popcount-threshold readout (Fix 1).

Replaces the failed narrow LSH bucket memory (Stage 1b: 10-bit address could not
preserve a 16384-bit neighborhood, tenet #14) with a full-width learned binary
projection:

    output_bit[j] = sign( popcount(bundle XOR W[j]) - theta[j] )

W is a (d_t, d_s) **binary** matrix (8 MB packed for d_t=4096, d_s=16384), theta is
int8 per output bit. Inference is pure Hamming ops — no floats in the artifact
(tenet #12). The "address" is now the entire 16384-bit bundle, removing the
information-theoretic bottleneck (tenet #14 fixed).

Training is delta-rule (perceptron) on binary features: each output bit is an
independent classifier trained to match the teacher target bit. Learning sits at
the hash output (tenet #1): the bundle is SHA-derived, the readout is downstream.

Stage 2b (Fix 2, same file, --neighborhood flag) adds a neighborhood-preserving
contrastive pass that directly optimizes recall@k, not per-bit accuracy (tenet #9).

Stage 2c (Fix 4, same file, --evolve flag) uses correlation-based feature selection
to sparsify W — only the most discriminative bundle bits contribute to each output
bit (discrete search, no gradients).
"""
import os
import sys
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
TOP_K = 10
SHORTLIST = 100


# ---------------------------------------------------------------------------
# Hamming retrieval helpers (shared with stage0/1)
# ---------------------------------------------------------------------------
def hamming_shortlist(codes, qrows, k):
    out = np.empty((len(qrows), k), dtype=np.int64)
    for i, q in enumerate(qrows):
        dist = POPCOUNT8[np.bitwise_xor(codes, codes[q])].sum(axis=1)
        dist[q] = np.iinfo(dist.dtype).max
        out[i] = np.argpartition(dist, k)[:k]
    return out


def recall_raw_rerank(packed, qrows, gt, Xn, extra_shortlist=None):
    short = hamming_shortlist(packed, qrows, SHORTLIST)
    top10 = hamming_shortlist(packed, qrows, TOP_K)
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
    return raw, float(np.mean(rr))


# ---------------------------------------------------------------------------
# Wide readout: inference and training
# ---------------------------------------------------------------------------
def readout_inference(S_bool, W_packed, theta, W_bits=None):
    """Compute output codes via popcount-threshold.

    S_bool: (n, d_s) bool bundle bits
    W_packed: (d_t, d_s//8) uint8 — packed binary projection (unused if W_bits given)
    theta: (d_t,) int8 thresholds
    W_bits: (d_t, d_s) uint8 {0,1} — unpacked W (optional, avoids re-unpacking)

    Uses the identity: hamming(x, w) = popcount(x) + popcount(w) - 2*popcount(x AND w)
    so popcount(x AND w) = X @ W^T (a matmul), making the whole readout one BLAS call.

    output[j] = hamming(S, W[j]) < theta[j]
    """
    n, d_s = S_bool.shape
    d_t = W_packed.shape[0]
    if W_bits is None:
        W_bits = np.unpackbits(W_packed, axis=1)  # (d_t, d_s) uint8

    # S as float32 for BLAS matmul, chunked over docs to bound memory
    S_pc = S_bool.sum(axis=1).astype(np.int32)  # (n,) popcount per doc
    W_pc = W_bits.sum(axis=1).astype(np.int32)  # (d_t,) popcount per weight row

    # AND popcount = S_bool @ W_bits^T  -> (n, d_t)
    # Compute in chunks over docs
    out_bits = np.empty((n, d_t), dtype=bool)
    tchunk = 500
    for t0i in range(0, n, tchunk):
        t1i = min(t0i + tchunk, n)
        S_chunk = S_bool[t0i:t1i].astype(np.float32)  # (tc, d_s)
        and_pc = (S_chunk @ W_bits.T.astype(np.float32)).astype(np.int32)  # (tc, d_t)
        hamming = S_pc[t0i:t1i, None] + W_pc[None, :] - 2 * and_pc  # (tc, d_t)
        out_bits[t0i:t1i] = hamming < theta[None, :]
        del S_chunk, and_pc, hamming
    return np.packbits(out_bits, axis=1)


def _hamming_dist_chunked(S_bool_chunk, W_bits_chunk, S_pc=None, W_pc=None):
    """Hamming distance from each doc to each weight row, via matmul identity.

    S_bool_chunk: (n, d_s) bool
    W_bits_chunk: (jc, d_s) uint8 {0,1}  (unpacked)
    S_pc: (n,) int — precomputed popcount of S rows (optional)
    W_pc: (jc,) int — precomputed popcount of W rows (optional)

    Returns: (n, jc) int32 distances.
    Uses: hamming(x,w) = popcount(x) + popcount(w) - 2*popcount(x AND w)
    """
    n, d_s = S_bool_chunk.shape
    jc = W_bits_chunk.shape[0]
    if S_pc is None:
        S_pc = S_bool_chunk.sum(axis=1).astype(np.int32)
    if W_pc is None:
        W_pc = W_bits_chunk.sum(axis=1).astype(np.int32)
    # AND popcount = S @ W^T -> (n, jc), via float32 BLAS
    S_f = S_bool_chunk.astype(np.float32)
    and_pc = (S_f @ W_bits_chunk.T.astype(np.float32)).astype(np.int32)
    del S_f
    return S_pc[:, None] + W_pc[None, :] - 2 * and_pc


def delta_rule_train(S_bool, targets, train_idx, n_epochs=5, lr=1.0, seed=42,
                     W_init=None, theta_init=None, verbose=True):
    """Batch perceptron (delta-rule) training of the binary readout.

    Each output bit j is an independent classifier:
        pred = popcount(x XOR W[j]) < theta[j]
        If wrong: move W[j] toward x (target=+1) or away (target=-1)

    Uses float32 matmuls (BLAS) for the batch update. W kept as float32 accumulator
    between epochs, binarized for distance computation.

    Returns W_packed (d_t, d_s//8) uint8, theta (d_t,) int8, W_bits (d_t, d_s) uint8.
    """
    rng = np.random.default_rng(seed)
    n, d_s = S_bool.shape
    d_t = targets.shape[1]
    S_tr_bool = S_bool[train_idx]  # (n_tr, d_s) bool — no extra copy, shares memory
    tgt = targets[train_idx]  # (n_tr, d_t) int8 bipolar
    n_tr = len(train_idx)

    # Float32 accumulator for W (256 MB) — the only large allocation
    if W_init is not None:
        W_acc = W_init.astype(np.float32) * 2 - 1  # convert {0,1} to {-1,+1}
    else:
        W_acc = (rng.integers(0, 2, size=(d_t, d_s), dtype=np.float32) * 2 - 1)

    if theta_init is not None:
        theta = theta_init.astype(np.int16).copy()
    else:
        theta = np.full(d_t, d_s // 2, dtype=np.int16)

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        n_errors = 0
        jchunk = 128
        # Precompute popcount for train docs (constant across j-chunks)
        S_tr_pc = S_tr_bool.sum(axis=1).astype(np.int32)
        for j0 in range(0, d_t, jchunk):
            j1 = min(j0 + jchunk, d_t)
            Wc_bits = (W_acc[j0:j1] > 0).astype(np.uint8)  # (jc, d_s)
            dist = _hamming_dist_chunked(S_tr_bool, Wc_bits, S_pc=S_tr_pc)  # (n_tr, jc)
            pred = dist < theta[None, j0:j1]  # (n_tr, jc)
            tgt_c = tgt[:, j0:j1]  # (n_tr, jc) {-1,+1}
            wrong = pred != (tgt_c > 0)  # (n_tr, jc) bool
            n_errors += int(wrong.sum())
            # Batch update via matmul, chunked over train docs to bound memory:
            # update = sign^T @ S_tr  where sign[i,j] = +1 (wrong,pos) or -1 (wrong,neg)
            # We compute it as: for each doc chunk, update += sign_chunk^T @ S_chunk_f
            update_sign = np.where(tgt_c > 0, wrong.astype(np.float32),
                                   -wrong.astype(np.float32))  # (n_tr, jc)
            update = np.zeros((j1 - j0, d_s), dtype=np.float32)
            tchunk = 1000
            for ti0 in range(0, n_tr, tchunk):
                ti1 = min(ti0 + tchunk, n_tr)
                S_chunk = S_tr_bool[ti0:ti1].astype(np.float32)  # (tc, d_s) — small
                update += update_sign[ti0:ti1].T @ S_chunk
                del S_chunk
            W_acc[j0:j1] += lr * update
            del update_sign, update, dist, pred, wrong, Wc_bits, tgt_c
        if verbose:
            print(f"  epoch {epoch+1}/{n_epochs}: {n_errors} errors "
                  f"({n_errors/(n_tr*d_t)*100:.1f}%)  {time.perf_counter()-t0:.1f}s", flush=True)

    W_bits = (W_acc > 0).astype(np.uint8)
    del W_acc
    W_packed = np.packbits(W_bits, axis=1)
    return W_packed, theta.astype(np.int8), W_bits


def hebbian_init_train(S_bool, targets, train_idx, seed=42):
    """One-shot Hebbian initialization: W[j] = sign(sum of target-+1 docs' bundles).

    Uses float32 matmul (BLAS) chunked over docs to bound memory.
    """
    n, d_s = S_bool.shape
    d_t = targets.shape[1]
    S_tr_bool = S_bool[train_idx]  # (n_tr, d_s) bool — shares memory
    tgt = targets[train_idx].astype(np.float32)  # (n_tr, d_t)
    n_tr = len(train_idx)
    # W = tgt^T @ S_tr  -> (d_t, d_s), chunked over docs
    W_acc = np.zeros((d_t, d_s), dtype=np.float32)
    tchunk = 1000
    for ti0 in range(0, n_tr, tchunk):
        ti1 = min(ti0 + tchunk, n_tr)
        S_chunk = S_tr_bool[ti0:ti1].astype(np.float32)  # (tc, d_s)
        W_acc += tgt[ti0:ti1].T @ S_chunk
        del S_chunk
    del tgt
    W_bits = (W_acc > 0).astype(np.uint8)
    del W_acc
    theta = np.full(d_t, d_s // 2, dtype=np.int16)
    return np.packbits(W_bits, axis=1), theta.astype(np.int8), W_bits


# ---------------------------------------------------------------------------
# Fix 2: Neighborhood-preserving contrastive pass
# ---------------------------------------------------------------------------
def neighborhood_finetune(S_bool, W_bits, theta, targets, train_idx, Xn, gt_holdout,
                          holdout, n_iters=3, lr=0.3, shortlist=20, n_triplets=2000,
                          seed=42, verbose=True):
    """Batch contrastive triplet rule that directly optimizes recall@k (tenet #9).

    For a batch of training docs, sample (anchor, positive=teacher-neighbor,
    negative=random-non-neighbor) triplets. For each output bit where the student
    ranks the negative closer than the positive (a ranking violation), batch-update
    W: push toward positives, away from negatives.

    Vectorized: all triplets processed per iteration in chunks over output bits.
    Perceptron-style — local, gradient-free, optimizes neighborhood overlap.
    """
    rng = np.random.default_rng(seed)
    n, d_s = S_bool.shape
    d_t = W_bits.shape[0]
    n_tr = len(train_idx)

    # Teacher neighbors for training docs (from float cosine)
    sims_train = Xn[train_idx] @ Xn.T  # (n_tr, n)
    np.fill_diagonal(sims_train[:, train_idx], -np.inf) if n_tr < n else None
    for i in range(n_tr):
        sims_train[i, train_idx[i]] = -np.inf
    teacher_topk = np.empty((n_tr, shortlist), dtype=np.int64)
    for i in range(n_tr):
        teacher_topk[i] = np.argpartition(-sims_train[i], shortlist)[:shortlist]
    del sims_train

    theta_i = theta.astype(np.int16).copy()

    for it in range(n_iters):
        t0 = time.perf_counter()
        n_violations = 0

        # Sample triplets
        anchor_local = rng.choice(n_tr, size=min(n_triplets, n_tr), replace=False)
        pos_idx = np.array([rng.choice(teacher_topk[a]) for a in anchor_local])
        neg_idx = rng.integers(0, n, size=len(anchor_local))
        # Ensure neg not in teacher top-k
        for t in range(len(neg_idx)):
            while neg_idx[t] in teacher_topk[anchor_local[t]] or neg_idx[t] == train_idx[anchor_local[t]]:
                neg_idx[t] = rng.integers(0, n)

        # Compute student distances: popcount(x XOR W[j]) for pos and neg
        # Process output bits in chunks
        jchunk = 64
        for j0 in range(0, d_t, jchunk):
            j1 = min(j0 + jchunk, d_t)
            Wc_bits = W_bits[j0:j1]  # (jc, d_s) uint8
            # Distances for positives and negatives: (n_trip, jc)
            dist_pos = _hamming_dist_chunked(S_bool[pos_idx], Wc_bits)  # (n_trip, jc)
            dist_neg = _hamming_dist_chunked(S_bool[neg_idx], Wc_bits)
            # Violation: negative closer than positive
            violations = dist_neg < dist_pos  # (n_trip, jc)
            n_violations += violations.sum()
            if not violations.any():
                continue
            # Batch update: for violated bits, accumulate pos/neg bit patterns
            # push W toward pos: set bits where pos=1 (majority over violated triplets)
            # push W away from neg: clear bits where neg=1
            S_pos = S_bool[pos_idx].astype(np.int16)   # (n_trip, d_s)
            S_neg = S_bool[neg_idx].astype(np.int16)   # (n_trip, d_s)
            for ji in range(j1 - j0):
                v = violations[:, ji]  # (n_trip,)
                if not v.any():
                    continue
                if rng.random() < lr:
                    # Toward positive: set bits where majority of violated pos have 1
                    pos_votes = S_pos[v].sum(axis=0)  # (d_s,)
                    thr = v.sum() / 2
                    W_bits[j0 + ji, pos_votes > thr] = 1
                    # Away from negative: clear bits where majority of violated neg have 1
                    neg_votes = S_neg[v].sum(axis=0)
                    W_bits[j0 + ji, neg_votes > thr] = 0
            del dist_pos, dist_neg, violations

        if verbose:
            print(f"  neighborhood iter {it+1}/{n_iters}: {n_violations} violations "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)

    W_packed = np.packbits(W_bits, axis=1)
    return W_packed, theta_i.astype(np.int8), W_bits


# ---------------------------------------------------------------------------
# Fix 4: Correlation-based feature selection (evolved sparse W)
# ---------------------------------------------------------------------------
def select_features(S_bool, targets, train_idx, k_per_bit=512, seed=42, verbose=True):
    """Select the k most discriminative bundle bits for each output bit.

    Uses point-biserial correlation (binary x vs bipolar y): for each (bundle_bit,
    output_bit) pair, compute correlation. Select top-k bundle bits per output bit.

    This produces a sparse W: only k_per_bit bundle bits are "active" per output bit,
    reducing inference cost and focusing on discriminative features.

    Returns selected_indices: (d_t, k_per_bit) int32 — which bundle bits matter per output.
    """
    rng = np.random.default_rng(seed)
    n, d_s = S_bool.shape
    d_t = targets.shape[1]
    S_tr = S_bool[train_idx].astype(np.float32)  # (n_tr, d_s)
    tgt = targets[train_idx].astype(np.float32)  # (n_tr, d_t)

    # Center
    S_c = S_tr - S_tr.mean(axis=0, keepdims=True)  # (n_tr, d_s)
    tgt_c = tgt - tgt.mean(axis=0, keepdims=True)  # (n_tr, d_t)

    # Correlation: (d_s, d_t) = S_c^T @ tgt_c / (norms)
    # d_s=16384, d_t=4096, n_tr=~9000 -> 16384*4096*4 = 256MB per matrix
    # Compute in chunks over d_t
    selected = np.empty((d_t, k_per_bit), dtype=np.int32)
    jchunk = 128
    for j0 in range(0, d_t, jchunk):
        j1 = min(j0 + jchunk, d_t)
        # corr_chunk: (d_s, jc) = S_c^T @ tgt_c[:, j0:j1]
        tc = tgt_c[:, j0:j1]  # (n_tr, jc)
        cross = S_c.T @ tc  # (d_s, jc)
        # Normalize by product of std devs
        s_std = np.sqrt((S_c ** 2).sum(axis=0))  # (d_s,)
        t_std = np.sqrt((tc ** 2).sum(axis=0))  # (jc,)
        denom = s_std[:, None] * t_std[None, :] + 1e-8
        corr = np.abs(cross / denom)  # (d_s, jc)
        # Top-k per output bit (column)
        for ji in range(j1 - j0):
            topk = np.argpartition(-corr[:, ji], k_per_bit)[:k_per_bit]
            selected[j0 + ji] = topk
        if verbose and j0 % (jchunk * 4) == 0:
            print(f"  feature selection: {j0}/{d_t} output bits done", flush=True)

    return selected


def build_sparse_readout(S_bool, targets, train_idx, selected, seed=42):
    """Build W from selected features only: Hebbian on the sparse subset.

    W[j] has non-zero entries only at selected[j] positions. For each output bit j,
    W[j, selected[j]] = sign(sum of target-+1 docs' bundle bits at those positions).
    """
    n, d_s = S_bool.shape
    d_t = selected.shape[0]
    k = selected.shape[1]
    S_tr = S_bool[train_idx].astype(np.int16)  # (n_tr, d_s)
    tgt = targets[train_idx]  # (n_tr, d_t) bipolar

    W = np.zeros((d_t, d_s), dtype=np.uint8)
    theta = np.full(d_t, k // 2, dtype=np.int16)  # threshold scales with k

    jchunk = 256
    for j0 in range(0, d_t, jchunk):
        j1 = min(j0 + jchunk, d_t)
        for ji in range(j1 - j0):
            j = j0 + ji
            bits = selected[j]  # (k,)
            pos = tgt[:, j] > 0  # (n_tr,)
            votes = S_tr[pos][:, bits].sum(axis=0)  # (k,)
            W[j, bits[votes > pos.sum() / 2]] = 1

    return np.packbits(W, axis=1), theta.astype(np.int8), W


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_cache():
    cache = np.load(os.path.join(RESULTS, "stage1_cache.npz"))
    S = np.unpackbits(cache["S"], axis=1).astype(bool)  # (n, 16384) bool
    Xn = cache["Xn"]
    targets = cache["targets"].astype(np.int8)  # (n, 4096) bipolar
    holdout = cache["holdout"]
    gt = cache["gt"]
    return S, Xn, targets, holdout, gt


def main():
    parser = argparse.ArgumentParser(description="Stage 2: wide binary readout")
    parser.add_argument("--neighborhood", action="store_true",
                        help="Add Fix 2: neighborhood-preserving contrastive pass")
    parser.add_argument("--evolve", action="store_true",
                        help="Add Fix 4: correlation-based feature selection (sparse W)")
    parser.add_argument("--epochs", type=int, default=5, help="Delta-rule epochs")
    parser.add_argument("--ab", action="store_true",
                        help="Add Fix 5: A/B comparison vs Stage 0 SimHash-of-teacher")
    args = parser.parse_args()

    print("loading stage1 cache...", flush=True)
    S, Xn, targets, holdout, gt = load_cache()
    n = len(S)
    train_idx = np.setdiff1d(np.arange(n), holdout)
    print(f"  {n} docs, d_s={S.shape[1]}, d_t={targets.shape[1]}", flush=True)
    print(f"  train: {len(train_idx)}, holdout: {len(holdout)}", flush=True)

    # Baseline A: bundle (no learning)
    bundle_codes = np.packbits(S, axis=1)
    bundle_short = hamming_shortlist(bundle_codes, holdout, SHORTLIST)
    a_raw, a_rr = recall_raw_rerank(bundle_codes, holdout, gt, Xn)
    print(f"\nbaseline A (bundle): raw {a_raw:.3f} rerank {a_rr:.3f}", flush=True)

    # Ceiling C: SimHash of teacher
    teacher_codes = np.packbits(targets > 0, axis=1)
    c_raw, c_rr = recall_raw_rerank(teacher_codes, holdout, gt, Xn)
    print(f"ceiling C (SimHash teacher): raw {c_raw:.3f} rerank {c_rr:.3f}\n", flush=True)

    results = {}

    # --- Fix 1: Wide binary readout (delta-rule) ---
    print("=== Fix 1: Wide binary readout ===", flush=True)
    print("Hebbian init...", flush=True)
    W_packed, theta, W_bits = hebbian_init_train(S, targets, train_idx)
    codes_init = readout_inference(S, W_packed, theta)
    r_init, rr_init = recall_raw_rerank(codes_init, holdout, gt, Xn)
    print(f"  Hebbian init only: raw {r_init:.3f} rerank {rr_init:.3f}", flush=True)
    results["D1: wide readout (Hebbian init)"] = (r_init, rr_init)

    print(f"Delta-rule training ({args.epochs} epochs)...", flush=True)
    W_packed, theta, W_bits = delta_rule_train(
        S, targets, train_idx, n_epochs=args.epochs,
        W_init=W_bits, theta_init=theta.astype(np.int16))
    codes_d = readout_inference(S, W_packed, theta)
    r_d, rr_d = recall_raw_rerank(codes_d, holdout, gt, Xn)
    print(f"  Wide readout (delta-rule {args.epochs}ep): raw {r_d:.3f} rerank {rr_d:.3f}",
          flush=True)
    results[f"D2: wide readout (delta-rule {args.epochs}ep)"] = (r_d, rr_d)

    # Union with bundle
    _, rr_union = recall_raw_rerank(codes_d, holdout, gt, Xn,
                                     extra_shortlist=bundle_short)
    print(f"  D2 union A rerank: {rr_union:.3f}", flush=True)
    results["D2 union A"] = (None, rr_union)

    # --- Fix 2: Neighborhood-preserving pass ---
    if args.neighborhood:
        print("\n=== Fix 2: Neighborhood-preserving contrastive pass ===", flush=True)
        W_packed_n, theta_n, W_bits_n = neighborhood_finetune(
            S, W_bits.copy(), theta, targets, train_idx, Xn, gt, holdout,
            n_iters=3, lr=0.3)
        codes_n = readout_inference(S, W_packed_n, theta_n)
        r_n, rr_n = recall_raw_rerank(codes_n, holdout, gt, Xn)
        print(f"  + neighborhood (3 iters): raw {r_n:.3f} rerank {rr_n:.3f}", flush=True)
        results["E: + neighborhood"] = (r_n, rr_n)
        _, rr_union_n = recall_raw_rerank(codes_n, holdout, gt, Xn,
                                           extra_shortlist=bundle_short)
        print(f"  E union A rerank: {rr_union_n:.3f}", flush=True)
        results["E union A"] = (None, rr_union_n)

    # --- Fix 4: Feature selection (sparse W) ---
    if args.evolve:
        print("\n=== Fix 4: Correlation-based feature selection ===", flush=True)
        for k in [256, 512]:
            print(f"  selecting top-{k} bits per output bit...", flush=True)
            selected = select_features(S, targets, train_idx, k_per_bit=k)
            W_packed_s, theta_s, _ = build_sparse_readout(S, targets, train_idx, selected)
            codes_s = readout_inference(S, W_packed_s, theta_s)
            r_s, rr_s = recall_raw_rerank(codes_s, holdout, gt, Xn)
            print(f"  sparse W (k={k}): raw {r_s:.3f} rerank {rr_s:.3f}", flush=True)
            results[f"F: sparse W (k={k})"] = (r_s, rr_s)

    # --- Fix 5: A/B gate ---
    print("\n=== Fix 5: A/B decision gate ===", flush=True)
    print(f"| encoder | raw | rerank-100 |", flush=True)
    print(f"|---|---|---|", flush=True)
    print(f"| A: bundle (no learning) | {a_raw:.3f} | {a_rr:.3f} |", flush=True)
    for name, (r, rr) in results.items():
        rstr = f"{r:.3f}" if r is not None else "—"
        print(f"| {name} | {rstr} | {rr:.3f} |", flush=True)
    print(f"| C: SimHash of teacher (ceiling) | {c_raw:.3f} | {c_rr:.3f} |", flush=True)

    # --- Write report ---
    lines = [
        "# Stage 2 report: wide binary readout (Fix 1-5)",
        "",
        f"- docs {n}, queries {len(holdout)} (held out), d_s {S.shape[1]}, "
        f"d_t {targets.shape[1]}, epochs {args.epochs}",
        f"- neighborhood: {args.neighborhood}, evolve: {args.evolve}",
        "",
        "| encoder | recall@10 raw | recall@10 rerank-100 |",
        "|---|---|---|",
        f"| A: bundle (no learning) | {a_raw:.3f} | {a_rr:.3f} |",
    ]
    for name, (r, rr) in results.items():
        rstr = f"{r:.3f}" if r is not None else "—"
        lines.append(f"| {name} | {rstr} | {rr:.3f} |")
    lines.append(f"| C: SimHash of teacher (ceiling) | {c_raw:.3f} | {c_rr:.3f} |")
    lines += [
        "",
        "## Notes",
        "- Fix 1 (wide readout): full-width binary projection removes the 10-bit address",
        "  bottleneck (tenet #14). W is binary (8 MB), theta is int8 — no floats in artifact",
        "  (tenet #12). Inference = popcount + threshold.",
        "- Fix 2 (neighborhood): contrastive triplet rule optimizes recall@k directly",
        "  (tenet #9), not per-bit accuracy.",
        "- Fix 4 (evolve): correlation-based feature selection sparsifies W.",
        "- Fix 5 (A/B gate): compare student vs teacher-SimHash ceiling.",
    ]
    report_path = os.path.join(RESULTS, "stage2_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nreport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
