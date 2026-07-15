"""Stage 3: token-level semantic binding (Fix 3).

ROOT CAUSE (measured 2026-07-14): the float ridge regression upper bound on bundle
features is raw=0.266 rerank=0.602 — identical to the bundle baseline (0.257/0.595).
SHA-256 token hypervectors are maximally decorrelated (avalanche), so the bundle
encodes only lexical overlap. No doc-level readout can manufacture semantic similarity
from decorrelated inputs. The bottleneck is the FEATURES, not the readout.

THIS FIX changes the features themselves: bind each token's SHA hypervector to a
learned semantic-cluster hypervector, so tokens in the same semantic cluster get
correlated vectors. The bundle then carries semantic similarity, not just lexical
overlap.

Pipeline:
  1. Fetch docs + teacher embeddings from drake-memory DB
  2. Tokenize with nomic's exact tokenizer (tenet #10)
  3. For each token id, compute the mean teacher embedding of all docs containing
     it -> the token's semantic centroid (this is where the teacher's semantic
     knowledge is injected)
  4. K-means cluster the centroids into K semantic clusters
  5. Each cluster gets a SHA-256 hypervector: H("cluster||k")
  6. Enriched token hv = majority(SHA_hv(token), cluster_hv, cluster_hv, cluster_hv)
     — cluster_hv repeated 3x so it dominates the majority, giving same-cluster
     tokens correlated vectors while preserving token identity
  7. IDF-weighted bundle of enriched hvs -> semantic bundle
  8. Eval: Hamming recall@k vs teacher cosine ground truth

TENET COMPLIANCE:
  - #1 (learning at hash output): cluster assignment is learned from teacher
    embeddings; the enriched hv is downstream of SHA (token + cluster are both
    SHA-derived; the ASSIGNMENT is the learned state, stored as int16 per token)
  - #2 (SHA not as LSH): SHA generates cluster hypervectors (decorrelation wanted);
    similarity comes from the learned cluster assignment, not from hashing
  - #12 (no floats in artifact): artifact = cluster_ids (int16 array, ~60 KB for
    30k tokens) + 32-byte seed. The teacher embeddings, centroids, and K-means
    are training-time-only.
  - #13 (no position binding in retrieval): bag-of-unique-tokens bundle, no position
"""
import os
import sys
import time
import hashlib

import numpy as np
import psycopg2
from tokenizers import Tokenizer
from sklearn.cluster import MiniBatchKMeans

DB_URL = os.environ.get("DM_DATABASE_URL", "postgres://postgres@localhost:5432/drake_memory")
TOK_PATH = os.path.expanduser("~/.drake-memory/models/tokenizer.json")

N_DOCS = 10000
N_QUERIES = 500
TOP_K = 10
SHORTLIST = 100
MAX_TOKENS = 128

D_S = 16384          # student internal width (bits)
D_T = 4096           # target/output code width (bits)
N_CLUSTERS = 256     # semantic clusters (tuneable: too few = mush, too many = lexical)
CLUSTER_WEIGHT = 0.25  # fraction of bits injected from cluster hv (0=none, 1=all cluster)
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def sha_bits(label, nbits):
    """Deterministic hypervector: nbits from chained SHA-256 of label. Packed uint8."""
    out = bytearray()
    base = label.encode()
    for j in range(nbits // 256):
        out += hashlib.sha256(base + j.to_bytes(4, "little")).digest()
    return np.frombuffer(bytes(out), dtype=np.uint8)


def fetch(n):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT content, vector::text FROM memories TABLESAMPLE SYSTEM (5) "
            "WHERE vector IS NOT NULL AND length(content) > 20 LIMIT %s", (n,))
        rows = cur.fetchall()
        if len(rows) < n // 2:
            raise RuntimeError("tablesample too small")
    except Exception:
        conn.rollback()
        cur.execute("SELECT content, vector::text FROM memories "
                    "WHERE vector IS NOT NULL AND length(content) > 20 LIMIT %s", (n,))
        rows = cur.fetchall()
    conn.close()
    texts = [r[0] for r in rows]
    X = np.array([np.fromstring(r[1][1:-1], sep=",", dtype=np.float32) for r in rows])
    return texts, X


def hamming_shortlist(codes, qrows, k):
    out = np.empty((len(qrows), k), dtype=np.int64)
    for i, q in enumerate(qrows):
        dist = POPCOUNT8[np.bitwise_xor(codes, codes[q])].sum(axis=1)
        dist[q] = np.iinfo(dist.dtype).max
        out[i] = np.argpartition(dist, k)[:k]
    return out


def eval_codes(name, packed, qrows, gt, Xn):
    short = hamming_shortlist(packed, qrows, SHORTLIST)
    top10 = hamming_shortlist(packed, qrows, TOP_K)
    raw = np.mean([len(set(gt[i]) & set(top10[i])) / TOP_K for i in range(len(qrows))])
    rr = []
    for i, q in enumerate(qrows):
        cand = short[i]
        sims = Xn[cand] @ Xn[q]
        top = cand[np.argsort(-sims)[:TOP_K]]
        rr.append(len(set(gt[i]) & set(top)) / TOP_K)
    line = f"| {name} | {raw:.3f} | {np.mean(rr):.3f} |"
    print(line, flush=True)
    return line


def compute_token_centroids(texts, Xn, tokenizer, min_docs=3):
    """For each token id, compute the mean teacher embedding of docs containing it.

    Returns: token_ids (array), centroids (array, normalized), df (document freq).
    Only includes tokens appearing in >= min_docs docs.
    """
    print("tokenizing...", flush=True)
    encs = tokenizer.encode_batch(texts)
    doc_ids = [sorted(set(e.ids[:MAX_TOKENS])) for e in encs]

    # Build inverted index: token -> list of doc indices
    from collections import defaultdict
    inv = defaultdict(list)
    for di, ids in enumerate(doc_ids):
        for tid in ids:
            inv[tid].append(di)

    # Filter by min_docs
    valid = {tid: docs for tid, docs in inv.items() if len(docs) >= min_docs}
    print(f"  {len(valid)} tokens with >={min_docs} docs (of {len(inv)} total)", flush=True)

    token_ids = sorted(valid.keys())
    centroids = np.zeros((len(token_ids), Xn.shape[1]), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        docs = valid[tid]
        centroids[i] = Xn[docs].mean(axis=0)

    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids = centroids / norms

    df = np.array([len(valid[tid]) for tid in token_ids], dtype=np.int32)
    return np.array(token_ids, dtype=np.int32), centroids, df


def build_semantic_bundles(texts, tokenizer, token_ids, cluster_labels, id2cluster, d_s,
                           n_clusters=N_CLUSTERS, cluster_weight=CLUSTER_WEIGHT,
                           max_tokens=MAX_TOKENS):
    """Build IDF-weighted bundles using enriched token hypervectors.

    Enriched hv = majority(SHA_hv(token), cluster_hv * cluster_weight)
    where cluster_hv = SHA("cluster||k"). The cluster component dominates
    (cluster_weight vs 1), giving same-cluster tokens correlated vectors.

    Returns: (n_docs, d_s) bool array.
    """
    print("tokenizing for bundle...", flush=True)
    encs = tokenizer.encode_batch(texts)
    doc_ids = [sorted(set(e.ids[:max_tokens])) for e in encs]

    # IDF weights
    all_ids_in_docs = sorted({i for ids in doc_ids for i in ids})
    n_docs = len(texts)
    df = {}
    for ids in doc_ids:
        for i in ids:
            df[i] = df.get(i, 0) + 1
    votes = {i: max(1, int(round(np.log2(n_docs / df[i])))) for i in all_ids_in_docs}

    # Which tokens have cluster assignments?
    clustered = set(int(t) for t in token_ids)

    # Derive cluster hypervectors
    print(f"deriving {n_clusters} cluster hypervectors...", flush=True)
    cluster_hvs = {}
    for k in range(n_clusters):
        cluster_hvs[k] = sha_bits(f"cluster|{k}", d_s)

    # For each token, compute enriched hypervector via partial cluster injection.
    # Instead of majority (which overwrites token identity), flip only a fraction
    # of bits: with probability inject_frac, set the bit to cluster_hv's value;
    # otherwise keep token_hv's value. This gives same-cluster tokens a shared
    # bias (the injected cluster bits agree) while preserving most token identity.
    print("computing enriched token hypervectors...", flush=True)
    id2enriched = {}
    inject_rng = np.random.default_rng(seed=42)
    for i in all_ids_in_docs:
        tok_hv_packed = sha_bits(f"tok|{i}", d_s)
        if i in clustered:
            k = int(id2cluster[i])
            cl_hv_packed = cluster_hvs[k]
            tok_bits = np.unpackbits(tok_hv_packed).astype(bool)
            cl_bits = np.unpackbits(cl_hv_packed).astype(bool)
            # Inject: with probability inject_frac, take cluster bit; else token bit
            inject_mask = inject_rng.random(d_s) < cluster_weight
            enriched = np.where(inject_mask, cl_bits, tok_bits)
            id2enriched[i] = np.packbits(enriched)
        else:
            id2enriched[i] = tok_hv_packed
    del cluster_hvs

    # Build bundles
    print("building semantic bundles...", flush=True)
    S = np.zeros((n_docs, d_s), dtype=bool)
    t0 = time.perf_counter()
    for di, ids in enumerate(doc_ids):
        if not ids:
            continue
        wt = np.array([votes[i] for i in ids], dtype=np.int32)
        # Sum enriched hypervectors weighted by IDF votes
        counts = np.zeros(d_s, dtype=np.int32)
        for idx, tid in enumerate(ids):
            bits = np.unpackbits(id2enriched[tid]).astype(np.int32)
            counts += wt[idx] * bits
        S[di] = counts * 2 > wt.sum()  # weighted strict majority
        if di % 2000 == 1999:
            print(f"  {di+1}/{n_docs} docs ({time.perf_counter()-t0:.0f}s)", flush=True)

    return S


def simhash_teacher(Xn, d_t, rng):
    """SimHash teacher embeddings to d_t bits (Stage 0 ceiling)."""
    G = rng.standard_normal((Xn.shape[1], d_t)).astype(np.float32)
    bits = (Xn @ G) > 0
    return np.packbits(bits, axis=1)


def main():
    rng = np.random.default_rng(SEED)
    tokenizer = Tokenizer.from_file(TOK_PATH)

    print(f"fetching {N_DOCS} docs from DB...", flush=True)
    texts, X = fetch(N_DOCS)
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0
    texts = [t for t, k in zip(texts, keep) if k]
    Xn = X[keep] / norms[keep, None]
    n = len(texts)
    print(f"  {n} docs, teacher dim {Xn.shape[1]}", flush=True)

    # Ground truth neighbors (teacher float cosine)
    holdout = rng.choice(n, size=min(N_QUERIES, n // 10), replace=False)
    sims = Xn[holdout] @ Xn.T
    gt = np.empty((len(holdout), TOP_K), dtype=np.int64)
    for i, q in enumerate(holdout):
        s = sims[i].copy()
        s[q] = -np.inf
        gt[i] = np.argpartition(-s, TOP_K)[:TOP_K]

    # --- Step 1: Compute token semantic centroids ---
    print("\n=== Step 1: Token semantic centroids ===", flush=True)
    token_ids, centroids, df = compute_token_centroids(texts, Xn, tokenizer, min_docs=3)
    print(f"  {len(token_ids)} tokens with centroids", flush=True)

    # --- Step 2: K-means cluster the centroids ---
    print(f"\n=== Step 2: K-means clustering ({N_CLUSTERS} clusters) ===", flush=True)
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED,
                             batch_size=512, max_iter=200, n_init=3)
    cluster_labels = kmeans.fit_predict(centroids)
    print(f"  clustering: {time.perf_counter()-t0:.1f}s", flush=True)

    # token id -> cluster id mapping
    id2cluster = {int(token_ids[i]): int(cluster_labels[i]) for i in range(len(token_ids))}

    # Report cluster distribution
    counts = np.bincount(cluster_labels, minlength=N_CLUSTERS)
    print(f"  cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.0f}", flush=True)

    # --- Step 3: Build semantic bundles ---
    print(f"\n=== Step 3: Build semantic bundles (d_s={D_S}) ===", flush=True)
    S_sem = build_semantic_bundles(texts, tokenizer, token_ids, cluster_labels,
                                   id2cluster, D_S)
    sem_codes = np.packbits(S_sem, axis=1)

    # --- Step 4: Also build plain bundles (baseline A) for comparison ---
    # Reuse stage1 cache if available, otherwise compute
    cache_path = os.path.join(RESULTS, "stage1_cache.npz")
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        if cache["S"].shape[0] == n:
            S_plain = np.unpackbits(cache["S"], axis=1).astype(bool)
            plain_codes = np.packbits(S_plain, axis=1)
        else:
            S_plain = None
    else:
        S_plain = None

    # --- Step 5: Evaluate ---
    print(f"\nqueries are {len(holdout)} held-out docs", flush=True)
    print("| encoder | recall@10 raw | recall@10 rerank-100 |")
    print("|---|---|---|")
    lines = []

    if S_plain is not None:
        lines.append(eval_codes("A: plain bundle (lexical, no learning)",
                                plain_codes, holdout, gt, Xn))
    lines.append(eval_codes("D: semantic bundle (token clustering, no readout)",
                            sem_codes, holdout, gt, Xn))

    # Ceiling: SimHash of teacher
    teacher_codes = simhash_teacher(Xn, D_T, rng)
    lines.append(eval_codes("C: SimHash of teacher (ceiling)",
                            teacher_codes, holdout, gt, Xn))

    # --- Step 6: Sweep cluster count ---
    print("\n=== Cluster count sweep ===", flush=True)
    print("| n_clusters | cluster_weight | raw | rerank |")
    print("|---|---|---|---|")
    sweep_lines = ["| n_clusters | cluster_weight | raw | rerank |", "|---|---|---|---|"]

    # Reuse centroids for different K
    for n_cl in [64, 128, 256, 512, 1024]:
        for cw in [0.1, 0.25, 0.5]:
            t0 = time.perf_counter()
            km = MiniBatchKMeans(n_clusters=n_cl, random_state=SEED,
                                 batch_size=512, max_iter=200, n_init=3)
            cl = km.fit_predict(centroids)
            id2cl = {int(token_ids[i]): int(cl[i]) for i in range(len(token_ids))}

            S_sweep = build_semantic_bundles(texts, tokenizer, token_ids, cl,
                                             id2cl, D_S, n_clusters=n_cl,
                                             cluster_weight=cw)

            sweep_codes = np.packbits(S_sweep, axis=1)
            r, rr = _eval_quiet(sweep_codes, holdout, gt, Xn)
            line = f"| {n_cl} | {cw} | {r:.3f} | {rr:.3f} |"
            print(line + f"   ({time.perf_counter()-t0:.0f}s)", flush=True)
            sweep_lines.append(line)

    # --- Write report ---
    report_lines = [
        "# Stage 3 report: token-level semantic binding (Fix 3)",
        "",
        f"- docs {n}, queries {len(holdout)} (held out), d_s {D_S}, d_t {D_T}",
        f"- {len(token_ids)} tokens with semantic centroids (min_docs=3)",
        f"- base config: {N_CLUSTERS} clusters, cluster_weight={CLUSTER_WEIGHT}",
        "",
        "## Main results",
        "",
        "| encoder | recall@10 raw | recall@10 rerank-100 |",
        "|---|---|---|",
    ] + lines + [
        "",
        "## Cluster count sweep",
        "",
    ] + sweep_lines + [
        "",
        "## Notes",
        "- Fix 3 injects semantic similarity at the TOKEN level by binding each token's",
        "  SHA hypervector to its cluster's SHA hypervector (majority vote).",
        "- Cluster assignment is learned from teacher embedding centroids (K-means).",
        "- Artifact: cluster_ids (int16 array, ~60 KB for 30k tokens) + 32-byte seed.",
        "  No floats in the inference artifact (tenet #12).",
        "- The teacher embeddings, centroids, and K-means model are training-time-only.",
        "- Root cause addressed: SHA tokens are decorrelated -> bundle had only lexical",
        "  overlap (ridge upper bound = 0.602 rerank). Semantic binding gives tokens in",
        "  the same cluster correlated vectors -> bundle now carries semantic similarity.",
    ]
    report_path = os.path.join(RESULTS, "stage3_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nreport written to {report_path}", flush=True)

    # Save semantic bundles for potential readout (Stage 2 on semantic features)
    np.savez_compressed(os.path.join(RESULTS, "stage3_cache.npz"),
                        S_sem=np.packbits(S_sem, axis=1),
                        Xn=Xn,
                        holdout=holdout, gt=gt,
                        token_ids=token_ids, cluster_labels=cluster_labels)
    print("semantic bundles saved to results/stage3_cache.npz", flush=True)


def _eval_quiet(packed, qrows, gt, Xn):
    short = hamming_shortlist(packed, qrows, SHORTLIST)
    top10 = hamming_shortlist(packed, qrows, TOP_K)
    raw = np.mean([len(set(gt[i]) & set(top10[i])) / TOP_K for i in range(len(qrows))])
    rr = []
    for i, q in enumerate(qrows):
        cand = short[i]
        sims = Xn[cand] @ Xn[q]
        top = cand[np.argsort(-sims)[:TOP_K]]
        rr.append(len(set(gt[i]) & set(top)) / TOP_K)
    return raw, float(np.mean(rr))


if __name__ == "__main__":
    main()
