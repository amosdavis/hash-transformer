"""Stage 1b: sweep bucket-memory readout designs on cached Stage 1 bundles.

Loads results/stage1_cache.npz (bundles S, teacher floats Xn, bipolar targets,
holdout queries, ground truth) so each config takes seconds, not minutes.

Sweeps:
  - m tables x addr_bits (LSH sharpness vs table count)
  - readout: sign-per-bucket majority vs raw counter sum
  - one-shot Hebbian vs delta-rule epochs (perceptron-style error correction;
    still only learns at hash outputs, tenet #1)
Also reports the deployment combiner: union of bundle + student shortlists,
float-reranked.
"""
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
TOP_K = 10
SHORTLIST = 100
SEED = 7


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


def train_and_encode(S, targets, train_idx, m, addr_bits, epochs, readout, rng):
    n, d_s = S.shape
    d_t = targets.shape[1]
    B = 1 << addr_bits
    bitpos = rng.choice(d_s, size=(m, addr_bits), replace=False)
    weights = (1 << np.arange(addr_bits)).astype(np.int64)
    addr = np.stack([S[:, bitpos[t]] @ weights for t in range(m)], axis=0)  # (m, n)

    buckets = np.zeros((m, B, d_t), dtype=np.int16)  # counts are small (~n/B docs/bucket)
    tgt = targets.astype(np.int16)
    for t in range(m):
        np.add.at(buckets[t], addr[t, train_idx], tgt[train_idx])
    for _ in range(epochs - 1):  # delta rule: push only where prediction is wrong
        acc = np.zeros((len(train_idx), d_t), dtype=np.int32)
        for t in range(m):
            acc += np.sign(buckets[t, addr[t, train_idx]]).astype(np.int32) \
                if readout == "sign" else buckets[t, addr[t, train_idx]]
        pred = np.where(acc > 0, 1, -1).astype(np.int16)
        err = (pred != tgt[train_idx]).astype(np.int16) * tgt[train_idx]
        for t in range(m):
            np.add.at(buckets[t], addr[t, train_idx], err)

    acc = np.zeros((n, d_t), dtype=np.int32)
    for t in range(m):
        acc += np.sign(buckets[t, addr[t]]).astype(np.int32) \
            if readout == "sign" else buckets[t, addr[t]]
    return np.packbits(acc > 0, axis=1)


def main():
    cache = np.load(os.path.join(RESULTS, "stage1_cache.npz"))
    S = np.unpackbits(cache["S"], axis=1).astype(bool)
    Xn, targets = cache["Xn"], cache["targets"].astype(np.int32)
    holdout, gt = cache["holdout"], cache["gt"]
    n = len(S)
    train_idx = np.setdiff1d(np.arange(n), holdout)
    print(f"cache: {n} docs, d_s={S.shape[1]}, d_t={targets.shape[1]}", flush=True)

    bundle_codes = np.packbits(S, axis=1)
    bundle_short = hamming_shortlist(bundle_codes, holdout, SHORTLIST)
    a_raw, a_rr = recall_raw_rerank(bundle_codes, holdout, gt, Xn)
    print(f"baseline A (bundle): raw {a_raw:.3f} rerank {a_rr:.3f}\n", flush=True)

    header = "| m | addr_bits | epochs | readout | raw | rerank | A union B rerank |"
    print(header + "\n" + "|---" * 7 + "|", flush=True)
    lines = [header, "|---" * 7 + "|"]
    rng = np.random.default_rng(SEED)
    for m in (16, 64):
        for addr_bits in (8, 10, 12):
            if m == 64 and addr_bits == 12:
                continue  # 2GB+ of bucket counters; not worth the RAM
            for epochs, readout in ((1, "sign"), (1, "sum"), (3, "sum")):
                t0 = time.perf_counter()
                codes = train_and_encode(S, targets, train_idx, m, addr_bits,
                                         epochs, readout, rng)
                raw, rr = recall_raw_rerank(codes, holdout, gt, Xn)
                _, union_rr = recall_raw_rerank(codes, holdout, gt, Xn,
                                                extra_shortlist=bundle_short)
                line = (f"| {m} | {addr_bits} | {epochs} | {readout} "
                        f"| {raw:.3f} | {rr:.3f} | {union_rr:.3f} |")
                print(line + f"   ({time.perf_counter()-t0:.0f}s)", flush=True)
                lines.append(line)

    with open(os.path.join(RESULTS, "stage1b_report.md"), "w") as f:
        f.write("# Stage 1b: bucket readout sweep\n\n")
        f.write(f"baseline A (bundle): raw {a_raw:.3f} rerank {a_rr:.3f}\n\n")
        f.write("\n".join(lines) + "\n")
    print("report written to results/stage1b_report.md", flush=True)


if __name__ == "__main__":
    main()
