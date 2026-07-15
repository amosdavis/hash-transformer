"""Stage 1: single hash-native layer distilled from the nomic teacher.

Student pipeline (inference is SHA-256 + XOR + popcount + majority only):
  tokens (nomic's exact tokenizer, tenet #10)
    -> SHA-256 token hypervectors (d_s bits, derived not stored)
    -> XOR-bind with SHA-256 position hypervectors
    -> majority bundle over tokens -> s0 (d_s bits)
    -> m bucket tables addressed by bit-slices of s0 (locality comes from s0's
       bits, NOT from hashing floats - tenet #2; slice-as-integer is equivalent
       to SHA(slice) mod B up to a relabeling of buckets)
    -> sign of summed bucket counters -> student code (d_t bits)

Learning is Hebbian only (tenet #1): bucket counters accumulate bipolar targets
of the training docs that address them. Targets = SimHash(teacher embedding).

Evaluated per tenet #9 on neighborhood agreement with the teacher (recall@10
overlap vs float cosine ground truth), on held-out query docs.

Baselines reported:
  A. raw bundle s0 (no learning)      - how far pure lexical overlap gets
  B. student output (bucket memory)   - the trained model
  C. SimHash of teacher (Stage 0)     - the ceiling for d_t-bit codes
"""
import os
import time
import hashlib

import numpy as np
import psycopg2
from tokenizers import Tokenizer

DB_URL = os.environ.get("DM_DATABASE_URL", "postgres://postgres@localhost:5432/drake_memory")
TOK_PATH = os.path.expanduser("~/.drake-memory/models/tokenizer.json")

N_DOCS = 10000
N_QUERIES = 500
TOP_K = 10
SHORTLIST = 100
MAX_TOKENS = 128

D_S = 16384          # student internal width (bits)
D_T = 4096           # target/output code width (bits) - Stage 0 showed 4096+rerank is enough
M_TABLES = 16        # independent slice->bucket paths (tenet #6 analog of crossover gate)
ADDR_BITS = 12       # 4096 buckets per table
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


def encode_bundles(texts, tokenizer):
    """s0 for every doc: IDF-weighted majority bundle of SHA token hypervectors.

    v2 lessons (v1 scored 0.026, below the lexical baseline):
    - NO position binding: XOR-ing tokens with position hypervectors makes the
      same word at different positions contribute uncorrelated vectors, which
      destroys document similarity. Bag-of-unique-tokens bundle instead.
    - Integer IDF vote weights (stored int8 table indexed by token id = learned
      state at a hash output, tenet #1) so stopwords don't own the majority.
    """
    print("tokenizing...", flush=True)
    encs = tokenizer.encode_batch(texts)
    doc_ids = [sorted(set(e.ids[:MAX_TOKENS])) for e in encs]

    # document frequency -> integer IDF votes
    all_ids = sorted({i for ids in doc_ids for i in ids})
    df = {}
    for ids in doc_ids:
        for i in ids:
            df[i] = df.get(i, 0) + 1
    n_docs = len(texts)
    votes = {i: max(1, int(round(np.log2(n_docs / df[i])))) for i in all_ids}

    print(f"deriving SHA hypervectors for {len(all_ids)} distinct tokens...", flush=True)
    id2row = {i: r for r, i in enumerate(all_ids)}
    # keep the table packed (~31MB); unpack per doc to stay memory-light
    HVp = np.stack([sha_bits(f"tok|{i}", D_S) for i in all_ids])
    w = np.array([votes[i] for i in all_ids], dtype=np.int32)

    S = np.zeros((n_docs, D_S), dtype=bool)
    t0 = time.perf_counter()
    for di, ids in enumerate(doc_ids):
        if not ids:
            continue
        rows = [id2row[i] for i in ids]
        wt = w[rows]
        unp = np.unpackbits(HVp[rows], axis=1).astype(np.int32)  # (n_tok, D_S)
        counts = wt @ unp                                # weighted ones-count per bit
        S[di] = counts * 2 > wt.sum()                    # weighted strict majority
        if di % 5000 == 4999:
            print(f"  {di+1}/{n_docs} docs ({time.perf_counter()-t0:.0f}s)", flush=True)
    return S


def addresses(S, rng):
    """Per-table bucket address for every doc, from fixed random bit-slices of s0."""
    bitpos = rng.choice(D_S, size=(M_TABLES, ADDR_BITS), replace=False)
    weights = (1 << np.arange(ADDR_BITS)).astype(np.int64)
    return np.stack([S[:, bitpos[t]] @ weights for t in range(M_TABLES)], axis=0)  # (m, docs)


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


def main():
    rng = np.random.default_rng(SEED)
    tokenizer = Tokenizer.from_file(TOK_PATH)

    texts, X = fetch(N_DOCS)
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0
    texts = [t for t, k in zip(texts, keep) if k]
    Xn = X[keep] / norms[keep, None]
    n = len(texts)
    print(f"{n} docs, teacher dim {Xn.shape[1]}", flush=True)

    # --- targets: SimHash of teacher (training-time floats only, tenet #12)
    G = rng.standard_normal((Xn.shape[1], D_T)).astype(np.float32)
    Tbits = (Xn @ G) > 0
    targets = Tbits.astype(np.int16) * 2 - 1                      # bipolar
    teacher_codes = np.packbits(Tbits, axis=1)                    # baseline C

    # --- ground truth neighbors (teacher float cosine)
    holdout = rng.choice(n, size=min(N_QUERIES, n // 10), replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[holdout] = False
    sims = Xn[holdout] @ Xn.T
    gt = np.empty((len(holdout), TOP_K), dtype=np.int64)
    for i, q in enumerate(holdout):
        s = sims[i].copy()
        s[q] = -np.inf
        gt[i] = np.argpartition(-s, TOP_K)[:TOP_K]

    # --- student forward: bundles + addresses
    S = encode_bundles(texts, tokenizer)
    np.savez_compressed(os.path.join(RESULTS, "stage1_cache.npz"),
                        S=np.packbits(S, axis=1), Xn=Xn, targets=targets.astype(np.int8),
                        holdout=holdout, gt=gt)
    addr = addresses(S, rng)

    # --- Hebbian training on train split only (tenet #1: learning at hash outputs)
    print("training bucket memories (Hebbian)...", flush=True)
    buckets = np.zeros((M_TABLES, 1 << ADDR_BITS, D_T), dtype=np.int16)
    tr = np.where(train_mask)[0]
    for t in range(M_TABLES):
        np.add.at(buckets[t], addr[t, tr], targets[tr])

    # --- student codes for all docs
    # sign per retrieved bucket, then majority across the m tables, so a
    # heavily-populated bucket cannot outvote the other 15 tables
    acc = np.zeros((n, D_T), dtype=np.int16)
    for t in range(M_TABLES):
        acc += np.sign(buckets[t, addr[t]]).astype(np.int16)
    student_codes = np.packbits(acc > 0, axis=1)
    bundle_codes = np.packbits(S, axis=1)

    print(f"\nqueries are {len(holdout)} held-out docs (never trained on)")
    print("| codes | recall@10 raw | recall@10 rerank-100 |")
    print("|---|---|---|")
    lines = [eval_codes("A: raw bundle s0 (lexical, no learning)", bundle_codes, holdout, gt, Xn),
             eval_codes("B: student (bucket memory)", student_codes, holdout, gt, Xn),
             eval_codes("C: SimHash of teacher (Stage 0 ceiling)", teacher_codes, holdout, gt, Xn)]

    with open(os.path.join(RESULTS, "stage1_report.md"), "w") as f:
        f.write("# Stage 1 report: Hebbian hash-layer student vs teacher\n\n")
        f.write(f"- docs {n}, queries {len(holdout)} (held out), d_s {D_S}, d_t {D_T}, "
                f"tables {M_TABLES}, buckets/table {1 << ADDR_BITS}, max tokens {MAX_TOKENS}\n\n")
        f.write("| codes | recall@10 raw | recall@10 rerank-100 |\n|---|---|---|\n")
        f.write("\n".join(lines) + "\n")
    print("report written to results/stage1_report.md", flush=True)


if __name__ == "__main__":
    main()
