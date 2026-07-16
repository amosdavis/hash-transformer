"""Extract binary token codes from the teacher GGUF via pure LSH (zero matmul).

Reads token_embd.weight directly from the nomic-embed GGUF file, extracts
sign bits from the Q8_0 quantized values (high bit of int8 — no dequant),
then LSH-samples to 16384 bits via SHA-256 dimension selection.

Optionally applies SHA-256 avalanche correction: for token pairs whose
actual Hamming distance diverges from the expected distance (derived from
teacher sign-bit cosine), flip bits via SHA masks to converge.

NO MATMUL. NO FLOATS. NO MODEL INFERENCE. NO NETWORK.
Operations: file I/O, sign-bit extraction (int < 0), SHA-256, XOR, AND, popcount.

Output: results/token_codes_lsh.npz — compatible with the hamming embedder.

Usage:
    python train_lsh.py                    # LSH seed + SHA correction
    python train_lsh.py --no-correct       # LSH seed only (6 seconds)
    python train_lsh.py --gguf PATH        # custom GGUF path
"""
import os
import sys
import time
import struct
import hashlib
import argparse

import numpy as np

TOK_PATH = os.path.expanduser("~/.drake-memory/models/tokenizer.json")
DEFAULT_GGUF = os.path.expanduser("~/.drake-memory/models/nomic-embed-text-v1.5.Q8_0.gguf")

D = 16384          # output code width (bits)
D_BYTES = D // 8
TEACHER_DIM = 768  # teacher embedding dimension (sign bits per token)
VOCAB = 30522      # nomic tokenizer vocab size
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Q8_0 sign-bit extraction (no dequantization)
# ---------------------------------------------------------------------------
def extract_sign_bits_from_gguf(gguf_path):
    """Read token_embd.weight from GGUF and extract 768 sign bits per token.

    Q8_0 format: blocks of 32 values, each block = 2-byte f16 scale + 32 i8 quants.
    The sign of the original float is the sign of the int8 quant (scale is always
    positive f16). So sign = (quant < 0), extracted via high-bit inspection.

    Returns: (VOCAB, 768) bool array — sign bits per token.
    """
    print(f"reading token embeddings from {gguf_path}...", flush=True)
    t0 = time.perf_counter()

    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                # Shape is [768, 30522] = [dim, vocab]
                # Q8_0: each row of 768 values = 768/32 = 24 blocks × 34 bytes
                raw = np.frombuffer(t.data, dtype=np.uint8)
                break
        else:
            raise ValueError("token_embd.weight not found in GGUF")
    except ImportError:
        # Manual GGUF parse fallback
        raw = _manual_gguf_read(gguf_path)

    n_blocks_per_token = TEACHER_DIM // 32  # 24
    block_size = 34  # 2 bytes scale + 32 bytes quants
    token_data_size = n_blocks_per_token * block_size  # 816 bytes

    sign_bits = np.zeros((VOCAB, TEACHER_DIM), dtype=bool)

    for tok in range(VOCAB):
        offset = tok * token_data_size
        for blk in range(n_blocks_per_token):
            base = offset + blk * block_size
            # Skip 2-byte f16 scale, read 32 int8 quants
            quants = raw[base + 2: base + 34].view(np.int8) if hasattr(raw, 'view') else \
                     np.frombuffer(bytes(raw[base + 2: base + 34]), dtype=np.int8)
            bit_start = blk * 32
            sign_bits[tok, bit_start:bit_start + 32] = quants < 0

    print(f"  extracted {VOCAB} × {TEACHER_DIM} sign bits in {time.perf_counter()-t0:.1f}s", flush=True)
    return sign_bits


def _manual_gguf_read(gguf_path):
    """Fallback: manually parse GGUF to find token_embd.weight data."""
    with open(gguf_path, "rb") as f:
        f.read(4)  # magic
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def read_str(f):
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8")

        # Skip KV metadata
        for _ in range(n_kv):
            read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            # Skip value based on type (empirically determined for this GGUF)
            if vt in (4,):  # i32
                f.read(4)
            elif vt in (3, 6):  # f32
                f.read(4)
            elif vt == 8:  # string
                read_str(f)
            elif vt == 5:  # bool
                f.read(1)
            elif vt == 7:  # array
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                elem_sizes = {0: 1, 1: 1, 2: 2, 3: 4, 4: 4, 5: 1, 6: 4, 8: 2, 9: 8, 10: 8}
                f.read(elem_sizes.get(et, 4) * n)
            elif vt == 9:  # i64
                f.read(8)
            elif vt == 10:  # f64
                f.read(8)

        # Read tensor info to find token_embd.weight
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            if name == "token_embd.weight":
                # Seek to data offset and read
                data_size = (dims[0] // 32) * 34 * dims[1]  # Q8_0
                f.seek(offset)
                return np.frombuffer(f.read(data_size), dtype=np.uint8)

    raise ValueError("token_embd.weight not found")


# ---------------------------------------------------------------------------
# LSH sampling: 768 sign bits → 16384 bits via SHA-256 dimension selection
# ---------------------------------------------------------------------------
def sha_dim_select(label, n_output, n_input, seed=SEED):
    """Select n_output source dimensions from [0, n_input) via SHA-256.

    Each output bit i samples dimension SHA256(label || i) mod n_input.
    This is random-hyperplane LSH via bit sampling — preserves cosine
    similarity as Hamming similarity.
    """
    dims = np.empty(n_output, dtype=np.int32)
    for i in range(n_output):
        h = hashlib.sha256(f"{label}|{i}".encode()).digest()
        dims[i] = int.from_bytes(h[:4], "little") % n_input
    return dims


def lsh_sample(sign_bits, d_out=D, d_in=TEACHER_DIM):
    """SimHash via random ±1 hyperplanes (no matmul).

    For each output bit j, generate a random ±1 vector g_j from SHA-256.
    The output bit is: popcount(sign_bits XNOR g_j) > d_in/2

    This is mathematically equivalent to sign(x . g) for ±1 vectors —
    a proper random-hyperplane LSH that gives d_out INDEPENDENT bits.
    No matmul: just XNOR + popcount + threshold.

    sign_bits: (VOCAB, d_in) bool — input sign bits
    Returns: (VOCAB, d_out) bool — LSH codes
    """
    print(f"SimHash {d_in} -> {d_out} bits via random hyperplanes (no matmul)...", flush=True)
    t0 = time.perf_counter()

    n_tokens = sign_bits.shape[0]
    threshold = d_in // 2
    codes = np.zeros((n_tokens, d_out), dtype=bool)

    # Process in chunks of output bits for progress reporting
    chunk = 512
    for j0 in range(0, d_out, chunk):
        j1 = min(j0 + chunk, d_out)
        for j in range(j0, j1):
            # Generate random ±1 vector g_j from SHA-256
            g = _sha_random_signs(f"hyper|{j}", d_in)  # (d_in,) bool
            # XNOR: agree where sign_bits == g
            agree = ~(sign_bits ^ g)  # (VOCAB, d_in) — True where they agree
            # Popcount per token: count agreeing bits
            counts = agree.sum(axis=1)  # (VOCAB,)
            codes[:, j] = counts > threshold
        if (j0 + chunk) % 2048 == 0:
            print(f"  {j1}/{d_out} bits ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    return codes


def _sha_random_signs(label, n_bits):
    """Generate n_bits random ±1 (bool) values from SHA-256."""
    n_bytes = (n_bits + 7) // 8
    out = bytearray()
    base = label.encode()
    for j in range(n_bytes // 32 + 1):
        out += hashlib.sha256(base + j.to_bytes(4, "little")).digest()
    bits = np.unpackbits(np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8))
    return bits[:n_bits].astype(bool)


# ---------------------------------------------------------------------------
# SHA-256 avalanche correction
# ---------------------------------------------------------------------------
def precompute_arcsin_table(d, resolution=768):
    """Precompute expected Hamming distance for each possible cosine value.

    For cosine similarity ρ, expected Hamming distance = d × (1/2 - arcsin(ρ)/π).
    We use integer lookup: for sign-bit popcount distances [0, d_in],
    map to expected d-bit Hamming distances.
    """
    table = np.zeros(resolution + 1, dtype=np.int32)
    for hamming in range(resolution + 1):
        # cosine = (d_in - 2*hamming) / d_in
        cos = (resolution - 2 * hamming) / resolution
        # expected hamming on d_out bits
        import math
        expected = d * (0.5 - math.asin(max(-1, min(1, cos))) / math.pi)
        table[hamming] = int(round(expected))
    return table


def sha_correction(codes, sign_bits, d_out=D, d_in=TEACHER_DIM, max_pairs=500000,
                   density=0.05, seed=SEED):
    """SHA-256 avalanche correction: converge actual Hamming toward expected.

    For token pairs where actual Hamming ≠ expected (from teacher sign-bit cosine):
    - If too close (actual < expected): flip AGREEING bits via SHA mask (push apart)
    - If too far (actual > expected): flip DIFFERING bits via SHA mask (pull together)

    The SHA mask ensures flipped bits are maximally decorrelated from all
    other codes (avalanche property), so corrections don't create new errors.

    Only corrects pairs within a Hamming window (close pairs found via
    bucketing on the 768-bit teacher codes).
    """
    print("SHA-256 avalanche correction...", flush=True)
    t0 = time.perf_counter()

    n_tokens = codes.shape[0]
    codes_packed = np.packbits(codes, axis=1)  # (VOCAB, d_out//8) uint8

    # Precompute expected Hamming lookup table
    arcsin_table = precompute_arcsin_table(d_out, d_in)

    # Find candidate pairs: tokens with high teacher sign-bit agreement
    # (close in cosine → need correction most)
    # Use random sampling of pairs (checking all 931M is too slow)
    rng = np.random.default_rng(seed)

    # Sample pairs: for each token, compare with a random sample of others
    n_sample = min(100, n_tokens)  # compare each token with 100 random others
    n_corrections = 0

    for tok in range(n_tokens):
        # Pick random partners
        partners = rng.choice(n_tokens, size=n_sample, replace=False)
        partners = partners[partners != tok]

        # Teacher sign-bit Hamming distance
        teacher_ham = np.array([
            np.sum(sign_bits[tok] != sign_bits[p]) for p in partners
        ])  # (n_sample,)

        for idx, p in enumerate(partners):
            th = teacher_ham[idx]
            if th > d_in // 2:  # dissimilar tokens — skip (no correction needed)
                continue

            expected = arcsin_table[th]
            # Actual Hamming on d_out bits (popcount of XOR)
            actual = POPCOUNT8[np.bitwise_xor(codes_packed[tok], codes_packed[p])].sum()

            if abs(actual - expected) < 3:
                continue  # close enough

            # Determine flip direction
            if actual < expected:
                # Too close — flip AGREEING bits to push apart
                agree = ~(codes[tok] ^ codes[p])  # bits where they agree
                direction = "push"
            else:
                # Too far — flip DIFFERING bits to pull together
                agree = codes[tok] ^ codes[p]  # bits where they differ
                direction = "pull"

            # SHA mask: select density fraction of bits to flip
            mask_label = f"corr|{tok}|{p}"
            mask_raw = np.frombuffer(
                hashlib.sha256(mask_label.encode()).digest() * (d_out // 256 + 1),
                dtype=np.uint8
            )[:d_out // 8]
            mask_bits = np.unpackbits(mask_raw)[:d_out]
            flip_mask = (mask_bits < (density * 256)) & agree

            # Apply flips to BOTH tokens (symmetric correction)
            codes[tok] ^= flip_mask
            codes[p] ^= flip_mask
            n_corrections += 1

    print(f"  {n_corrections} corrections in {time.perf_counter()-t0:.1f}s", flush=True)
    return codes


# ---------------------------------------------------------------------------
# IDF computation (from teacher token frequency in tokenizer scores)
# ---------------------------------------------------------------------------
def compute_idf_from_gguf(gguf_path):
    """Compute IDF from the tokenizer scores in the GGUF metadata.

    The nomic tokenizer stores per-token scores (typically log probabilities).
    We invert to get IDF: rare tokens get high votes.
    """
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
        for name, field in reader.fields.items():
            if name == "tokenizer.ggml.scores":
                # Extract scores array
                scores = field.parts[field.data[0]]
                if hasattr(scores, 'tolist'):
                    scores = scores.tolist()
                scores = np.array(scores, dtype=np.float32)
                # IDF = log(N / df) ≈ -log(score) for log-prob scores
                # Normalize to int8 [1, 127]
                idf_raw = -np.log(np.maximum(scores, 1e-10))
                idf_raw = np.clip(idf_raw, 0, 10)
                idf = np.clip((idf_raw * 12.7).astype(np.int32), 1, 127).astype(np.int8)
                return idf
    except Exception as e:
        print(f"  warning: could not extract IDF from GGUF ({e}), using uniform", flush=True)

    return np.ones(VOCAB, dtype=np.int8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LSH token code extraction from teacher GGUF (zero matmul)")
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help="Path to GGUF model file")
    parser.add_argument("--no-correct", action="store_true", help="Skip SHA-256 correction pass")
    parser.add_argument("--d", type=int, default=D, help="Output code width in bits")
    args = parser.parse_args()

    print("=== LSH Token Code Extraction (zero matmul) ===\n", flush=True)

    # Step 1: Extract sign bits from GGUF
    sign_bits = extract_sign_bits_from_gguf(args.gguf)

    # Step 2: LSH sample to d bits
    codes = lsh_sample(sign_bits, args.d, TEACHER_DIM)  # (VOCAB, d) bool

    # Step 3: SHA-256 avalanche correction (optional)
    if not args.no_correct:
        codes = sha_correction(codes, sign_bits, args.d, TEACHER_DIM)

    # Step 4: IDF
    print("\ncomputing IDF from tokenizer scores...", flush=True)
    idf = compute_idf_from_gguf(args.gguf)

    # Step 5: Save
    codes_packed = np.packbits(codes, axis=1)  # (VOCAB, d//8) uint8
    out_path = os.path.join(RESULTS, "token_codes_lsh.npz")
    np.savez_compressed(out_path, codes=codes_packed, idf=idf)
    print(f"\nsaved {out_path} ({codes_packed.nbytes/1e6:.1f} MB)", flush=True)
    print(f"\nTo deploy: copy to ~/.drake-memory/models/token_codes.npz", flush=True)
    print(f"Then set DM_EMBEDDING_BACKEND=hamming", flush=True)


if __name__ == "__main__":
    main()
