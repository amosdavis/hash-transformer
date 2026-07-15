"""Stage 0: SimHash drake-memory's nomic embeddings -> binary codes; measure retrieval fidelity.

Validates that the teacher's similarity geometry survives 1-bit quantization
(design doc tenet #9: evaluate neighborhood agreement, not per-bit anything).

The Gaussian projection matrix G is TRAINING-TIME ONLY (tenet #12): it binarizes
teacher outputs to make targets/indexes. It is never part of the inference artifact.
"""
import os
import sys
import time

import numpy as np
import psycopg2

DB_URL = os.environ.get("DM_DATABASE_URL", "postgres://postgres@localhost:5432/drake_memory")
N_DOCS = 20000
N_QUERIES = 500
TOP_K = 10
DIMS = [4096, 8192, 16384]
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def fetch_embeddings(n):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    # TABLESAMPLE gives a cheap spread across the 899k-row table; fall back to LIMIT
    try:
        cur.execute(
            "SELECT vector::text FROM memories TABLESAMPLE SYSTEM (5) "
            "WHERE vector IS NOT NULL LIMIT %s", (n,))
        rows = cur.fetchall()
        if len(rows) < n // 2:
            raise RuntimeError("tablesample too small")
    except Exception:
        conn.rollback()
        cur.execute("SELECT vector::text FROM memories WHERE vector IS NOT NULL LIMIT %s", (n,))
        rows = cur.fetchall()
    conn.close()
    if not rows:
        sys.exit("no embeddings found in drake_memory.memories")
    X = np.array([np.fromstring(r[0][1:-1], sep=",", dtype=np.float32) for r in rows])
    return X


def simhash(X, d, rng):
    G = rng.standard_normal((X.shape[1], d)).astype(np.float32)
    bits = (X @ G) > 0
    return np.packbits(bits, axis=1)  # (n, d//8) uint8


def hamming_topk(codes, qidx, k):
    """Brute-force Hamming top-k for each query row against all codes (excluding self)."""
    out = np.empty((len(qidx), k), dtype=np.int64)
    t0 = time.perf_counter()
    for i, q in enumerate(qidx):
        dist = POPCOUNT8[np.bitwise_xor(codes, codes[q])].sum(axis=1)
        dist[q] = np.iinfo(dist.dtype).max  # exclude self
        out[i] = np.argpartition(dist, k)[:k]
    return out, time.perf_counter() - t0


def rerank_recall(Xn, qidx, gt, candidates, k):
    """Deployment pattern: Hamming shortlist -> float cosine rescore of shortlist only."""
    recalls = []
    for i, q in enumerate(qidx):
        cand = candidates[i]
        sims = Xn[cand] @ Xn[q]
        top = cand[np.argsort(-sims)[:k]]
        recalls.append(len(set(gt[i]) & set(top)) / k)
    return float(np.mean(recalls))


def cosine_topk(Xn, qidx, k):
    out = np.empty((len(qidx), k), dtype=np.int64)
    t0 = time.perf_counter()
    sims = Xn[qidx] @ Xn.T  # (q, n)
    for i, q in enumerate(qidx):
        s = sims[i]
        s[q] = -np.inf
        out[i] = np.argpartition(-s, k)[:k]
    return out, time.perf_counter() - t0


def main():
    rng = np.random.default_rng(SEED)
    print(f"fetching up to {N_DOCS} embeddings from {DB_URL.split('@')[-1]} ...", flush=True)
    X = fetch_embeddings(N_DOCS)
    n, dim = X.shape
    print(f"got {n} embeddings, dim={dim}", flush=True)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    keep = norms[:, 0] > 0
    X = X[keep]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(Xn)

    qidx = rng.choice(n, size=min(N_QUERIES, n), replace=False)
    gt, t_cos = cosine_topk(Xn, qidx, TOP_K)
    print(f"cosine ground truth: {t_cos:.2f}s for {len(qidx)} queries x {n} docs", flush=True)

    lines = [
        "# Stage 0 report: SimHash fidelity on drake-memory embeddings",
        "",
        f"- docs: {n}, queries: {len(qidx)}, k: {TOP_K}, teacher dim: {dim} (nomic Matryoshka-truncated)",
        f"- cosine brute-force baseline: {t_cos:.2f}s total ({1e3*t_cos/len(qidx):.1f} ms/query), "
        f"float32 footprint {n*dim*4/1e6:.0f} MB",
        "",
        "| d (bits) | recall@10 raw | recall@10 rerank-100 | bytes/vec | total codes | Hamming scan ms/query |",
        "|---|---|---|---|---|---|",
    ]
    SHORTLIST = 100
    for d in DIMS:
        codes = simhash(Xn, d, rng)
        shortlist, t_ham = hamming_topk(codes, qidx, SHORTLIST)
        raw = np.mean([len(set(gt[i]) & set(shortlist[i][:TOP_K])) / TOP_K
                       for i in range(len(qidx))])
        # NB: hamming_topk's argpartition k-set is unordered; order shortlist by distance
        # doesn't matter for rerank (cosine reorders) but raw@10 needs true top-10:
        top10, _ = hamming_topk(codes, qidx, TOP_K)
        raw = np.mean([len(set(gt[i]) & set(top10[i])) / TOP_K for i in range(len(qidx))])
        rr = rerank_recall(Xn, qidx, gt, shortlist, TOP_K)
        np.save(os.path.join(RESULTS, f"codes_d{d}.npy"), codes)
        line = (f"| {d} | {raw:.3f} | {rr:.3f} | {codes.shape[1]} | {codes.nbytes/1e6:.0f} MB "
                f"| {1e3*t_ham/len(qidx):.1f} |")
        lines.append(line)
        print(line, flush=True)

    lines += [
        "",
        "Notes:",
        "- Hamming scan here is numpy uint8-lookup popcount (interpreter-bound); a native",
        "  VPOPCNT/AVX2 scan is typically 20-50x faster, while the cosine baseline already",
        "  enjoys optimized BLAS. Fidelity numbers are the point of Stage 0; speed parity",
        "  is a Rust/pgvector-bit follow-up.",
        "- Codes saved to results/codes_d*.npy are the Stage 1 training targets.",
        "- Projection matrix G is derived from seed 42 at training time only (tenet #12).",
    ]
    report = os.path.join(RESULTS, "stage0_report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("report written to", report, flush=True)


if __name__ == "__main__":
    main()
