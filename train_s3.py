"""Train binary token codes on the nomic S3 corpus via streaming.

Streams documents from R2 (s3://contrastive/...) — no local copy needed.
Trains via Mode 2 (co-occurrence Hebbian bit-flip): pure bitops, no matmul,
no GPU, no teacher embeddings.

Usage:
    python train_s3.py                          # default: 500 shards, 3 epochs
    python train_s3.py --shards 1000 --epochs 5 # more data, more epochs
    python train_s3.py --datasets refinedweb gooaq_gzip msmarco  # pick datasets

Output:
    results/token_codes_s3.npz  — trained code table (60 MB)
    Copy to ~/.drake-memory/models/token_codes.npz for the hamming embedder.

The script streams one shard at a time, processes every document, and
applies SHA-masked bit-flip updates to token codes. Memory footprint:
~60 MB (code table) + ~2 MB per shard (streamed, not buffered).
"""
import os
import sys
import time
import json
import gzip
import hashlib
import argparse
import random

import numpy as np
import s3fs
from tokenizers import Tokenizer

TOK_PATH = os.path.expanduser("~/.drake-memory/models/tokenizer.json")
D = 16384
D_BYTES = D // 8
MAX_TOKENS = 128
N_NEG = 5
SEED = 42
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Datasets to sample from (s3://contrastive/<prefix>/*.jsonl.gz)
DEFAULT_DATASETS = [
    "refinedweb",
    "gooaq_gzip",
    "msmarco",
    "reddit-title-body",
]


# ---------------------------------------------------------------------------
# SHA-256 code generation (same as token_memory.py)
# ---------------------------------------------------------------------------
def sha_code(label, n_bytes=D_BYTES):
    out = bytearray()
    base = label.encode()
    for j in range(n_bytes // 32):
        out += hashlib.sha256(base + j.to_bytes(4, "little")).digest()
    return np.frombuffer(bytes(out), dtype=np.uint8)


def sha_flip_mask_bits(label, n_bits=D, density=0.1):
    n_bytes = (n_bits + 7) // 8
    raw = sha_code(label, n_bytes)
    bits = np.unpackbits(raw)[:n_bits]
    return bits < (density * 256 / 256.0)


# ---------------------------------------------------------------------------
# Bit operations
# ---------------------------------------------------------------------------
def bits_differ_mask(code_packed, bundle_packed, n_bits=D):
    a = np.unpackbits(code_packed)[:n_bits]
    b = np.unpackbits(bundle_packed)[:n_bits]
    return a != b


def bits_agree_mask(code_packed, bundle_packed, n_bits=D):
    a = np.unpackbits(code_packed)[:n_bits]
    b = np.unpackbits(bundle_packed)[:n_bits]
    return a == b


def flip_selected_bits(code_packed, bit_mask, n_bits=D):
    bits = np.unpackbits(code_packed).copy()
    bits[:n_bits] ^= bit_mask.astype(np.uint8)
    return np.packbits(bits)


# ---------------------------------------------------------------------------
# Integer-tally majority bundle
# ---------------------------------------------------------------------------
def bundle_codes(codes, weights):
    bits = np.unpackbits(codes, axis=1).astype(np.int16)
    bits *= weights[:, None].astype(np.int16)
    counts = bits.sum(axis=0, dtype=np.int16)
    total = weights.sum(dtype=np.int16)
    return np.packbits(counts * 2 > total)


# ---------------------------------------------------------------------------
# Token table init
# ---------------------------------------------------------------------------
def init_code_table(vocab_size, d_bytes=D_BYTES):
    table = np.zeros((vocab_size, d_bytes), dtype=np.uint8)
    for t in range(vocab_size):
        table[t] = sha_code(f"tok|{t}", d_bytes)
    return table


def compute_idf_from_batch(doc_ids_list, n_docs):
    df = {}
    for ids in doc_ids_list:
        for i in ids:
            df[i] = df.get(i, 0) + 1
    max_tok = max(df.keys()) + 1 if df else 1
    idf = np.ones(max_tok, dtype=np.int8)
    for i, count in df.items():
        idf[i] = max(1, min(127, int(round(np.log2(n_docs / count)))))
    return idf


# ---------------------------------------------------------------------------
# S3 streaming
# ---------------------------------------------------------------------------
def list_shards(fs, datasets, max_shards):
    """List shard files across datasets, sample up to max_shards."""
    all_shards = []
    for ds in datasets:
        pattern = f"contrastive/{ds}/*.jsonl.gz"
        try:
            files = fs.glob(f"s3://{pattern}")
            all_shards.extend(files)
            print(f"  {ds}: {len(files)} shards", flush=True)
        except Exception as e:
            print(f"  {ds}: error listing — {e}", flush=True)
    random.shuffle(all_shards)
    return all_shards[:max_shards]


def stream_documents(fs, shard_path):
    """Stream documents from a single shard. Yields text strings."""
    with fs.open(shard_path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Different datasets use different field names
                text = doc.get("text") or doc.get("document") or doc.get("content") or ""
                if not text or len(text) < 20:
                    continue
                yield text


# ---------------------------------------------------------------------------
# Training: co-occurrence bit-flip
# ---------------------------------------------------------------------------
def train_on_shard(table, idf_table, tokenizer, texts, epoch, density, rng):
    """Process all documents in a shard, applying bit-flip updates.

    Returns (n_docs, n_updates).
    """
    n_docs = 0
    n_updates = 0
    V = table.shape[0]

    # Negative sampling distribution from this shard's tokens
    token_freq = np.zeros(V, dtype=np.float64)
    for text in texts:
        enc = tokenizer.encode(text)
        ids = sorted(set(enc.ids[:MAX_TOKENS]))
        for t in ids:
            token_freq[t] += 1
        n_docs += 1
    token_freq = token_freq ** 0.75
    total_freq = token_freq.sum()
    if total_freq > 0:
        token_freq /= total_freq

    # Reset and re-iterate to apply updates (need token freq first)
    # Actually, let's just process inline — token_freq from this shard
    # is approximate but fine for negative sampling

    for text in texts:
        enc = tokenizer.encode(text)
        ids = sorted(set(enc.ids[:MAX_TOKENS]))
        if len(ids) < 2:
            continue

        codes = table[ids]
        wt = idf_table[ids] if max(ids) < len(idf_table) else np.ones(len(ids), dtype=np.int8)
        bundle = bundle_codes(codes, wt)

        doc_token_set = set(ids)

        # Positive: pull each token toward bundle
        for tid in ids:
            differ = bits_differ_mask(table[tid], bundle)
            mask = sha_flip_mask_bits(f"pos|{epoch}|{tid}", D, density)
            flip_mask = np.bitwise_and(differ, mask)
            table[tid] = flip_selected_bits(table[tid], flip_mask)
            n_updates += 1

        # Negative: push random non-co-occurring tokens away
        if total_freq > 0:
            neg_ids = rng.choice(V, size=N_NEG, p=token_freq, replace=False)
            for nt in neg_ids:
                if nt in doc_token_set:
                    continue
                agree = bits_agree_mask(table[nt], bundle)
                mask = sha_flip_mask_bits(f"neg|{epoch}|{nt}", D, density)
                flip_mask = np.bitwise_and(agree, mask)
                table[nt] = flip_selected_bits(table[nt], flip_mask)

    return n_docs, n_updates


def main():
    parser = argparse.ArgumentParser(description="Train binary token codes on nomic S3 corpus")
    parser.add_argument("--shards", type=int, default=500, help="Max shards to sample")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Dataset prefixes to sample from")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (mask density)")
    parser.add_argument("--vocab", type=int, default=30522, help="Vocab size (nomic tokenizer)")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    print("loading tokenizer...", flush=True)
    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"  vocab size: {args.vocab}", flush=True)

    print("initializing SHA token codes...", flush=True)
    t0 = time.perf_counter()
    table = init_code_table(args.vocab)
    print(f"  {time.perf_counter()-t0:.1f}s ({table.nbytes/1e6:.1f} MB)", flush=True)

    # Connect to R2 via s3fs (uses ~/.config/fsspec/s3.json)
    print("connecting to R2...", flush=True)
    fs = s3fs.S3FileSystem()
    print("  connected", flush=True)

    # List and sample shards
    print(f"listing shards from {args.datasets}...", flush=True)
    shards = list_shards(fs, args.datasets, args.shards)
    print(f"  sampled {len(shards)} shards", flush=True)

    if not shards:
        print("ERROR: no shards found. Check R2 credentials and dataset names.", flush=True)
        sys.exit(1)

    # First pass: compute IDF from a sample of documents
    print("computing IDF from sample...", flush=True)
    sample_docs_ids = []
    sample_shards = shards[:min(50, len(shards))]
    for sp in sample_shards:
        doc_ids_batch = []
        for text in stream_documents(fs, sp):
            enc = tokenizer.encode(text)
            ids = sorted(set(enc.ids[:MAX_TOKENS]))
            if ids:
                doc_ids_batch.append(ids)
            if len(doc_ids_batch) >= 200:
                break
        sample_docs_ids.extend(doc_ids_batch)
    idf_table = compute_idf_from_batch(sample_docs_ids, len(sample_docs_ids))
    print(f"  IDF computed from {len(sample_docs_ids)} docs", flush=True)

    # Training loop
    total_docs = 0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        epoch_docs = 0
        epoch_updates = 0
        density = max(args.lr * (0.5 ** epoch), 0.001)

        for si, shard_path in enumerate(shards):
            shard_t0 = time.perf_counter()
            texts = list(stream_documents(fs, shard_path))
            if not texts:
                continue
            n_docs, n_updates = train_on_shard(
                table, idf_table, tokenizer, texts, epoch, density, rng)
            epoch_docs += n_docs
            epoch_updates += n_updates
            total_docs += n_docs

            if (si + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  epoch {epoch+1}/{args.epochs}: shard {si+1}/{len(shards)} "
                      f"({epoch_docs} docs, {epoch_updates} updates, "
                      f"{elapsed:.0f}s)", flush=True)

        print(f"epoch {epoch+1}/{args.epochs} done: {epoch_docs} docs, "
              f"{epoch_updates} updates, density={density:.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # Save
    out_path = os.path.join(RESULTS, "token_codes_s3.npz")
    np.savez_compressed(out_path, codes=table, idf=idf_table)
    print(f"\nsaved {out_path} ({table.nbytes/1e6:.1f} MB)", flush=True)
    print(f"total docs processed: {total_docs}", flush=True)
    print(f"\nTo deploy: copy {out_path} to ~/.drake-memory/models/token_codes.npz", flush=True)
    print(f"Then set DM_EMBEDDING_BACKEND=hamming", flush=True)


if __name__ == "__main__":
    main()
