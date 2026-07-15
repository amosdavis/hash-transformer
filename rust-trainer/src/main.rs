//! Binary Token Memory trainer — Rust port of train_s3.py.
//!
//! Pure bitops: SHA-256 init, XOR/AND/popcount training, integer-tally bundling.
//! Streams documents from R2 (s3://contrastive/...), trains binary token codes
//! via co-occurrence Hebbian bit-flip. No matmul, no floats, no GPU.
//!
//! Usage:
//!   train-s3                              # default: 500 shards, 3 epochs
//!   train-s3 --shards 1000 --epochs 5
//!   train-s3 --datasets refinedweb gooaq_gzip msmarco

use clap::Parser;
use rand::Rng;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Code width in bits.
const D_BITS: usize = 16384;
/// Code width in bytes.
const D_BYTES: usize = D_BITS / 8;
/// Max tokens per document.
const MAX_TOKENS: usize = 128;
/// Negative samples per document.
const N_NEG: usize = 5;
/// Vocabulary size (nomic tokenizer).
const VOCAB: usize = 30522;
/// Seed for reproducibility.
const SEED: u64 = 42;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Train binary token codes on nomic S3 corpus (pure bitops)")]
struct Args {
    /// Max shards to sample.
    #[arg(long, default_value = "500")]
    shards: usize,

    /// Training epochs.
    #[arg(long, default_value = "3")]
    epochs: usize,

    /// Initial learning rate (mask density).
    #[arg(long, default_value = "0.01")]
    lr: f64,

    /// Datasets to sample from.
    #[arg(long, value_delimiter = ' ', default_values = ["refinedweb", "gooaq_gzip", "msmarco", "reddit-title-body"])]
    datasets: Vec<String>,
}

// ─── SHA-256 code generation ─────────────────────────────────────────────────

/// Generate a deterministic binary code from SHA-256, packed as bytes.
fn sha_code(label: &str, n_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes);
    let base = label.as_bytes();
    for j in 0..(n_bytes / 32) {
        let mut hasher = Sha256::new();
        hasher.update(base);
        hasher.update(j.to_le_bytes());
        out.extend_from_slice(&hasher.finalize());
    }
    out
}

/// Generate a SHA-derived bit mask with ~density fraction of bits set.
/// Returns a Vec<bool> of length n_bits.
fn sha_flip_mask(label: &str, n_bits: usize, density: f64) -> Vec<bool> {
    let n_bytes = (n_bits + 7) / 8;
    let raw = sha_code(label, n_bytes);
    let threshold = (density * 256.0) as u8;
    let bits = unpack_bits(&raw);
    bits.into_iter().take(n_bits).map(|b| b < threshold).collect()
}

// ─── Bit operations ──────────────────────────────────────────────────────────

/// Unpack packed bytes to individual bits (MSB first, like numpy unpackbits).
fn unpack_bits(packed: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(packed.len() * 8);
    for &byte in packed {
        for bit in (0..8).rev() {
            bits.push((byte >> bit) & 1);
        }
    }
    bits
}

/// Pack individual bits back to bytes.
fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((bits.len() + 7) / 8);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            if b != 0 {
                byte |= 1 << (7 - i);
            }
        }
        packed.push(byte);
    }
    packed
}

/// Build IDF-weighted majority bundle from token codes.
/// Pure integer tally: count weighted 1-bits per position, threshold at half total.
fn bundle_codes(codes: &[&[u8]], weights: &[i8]) -> Vec<u8> {
    let n_tok = codes.len();
    if n_tok == 0 {
        return vec![0u8; D_BYTES];
    }

    // Tally per bit position using integer counters
    let mut counts = vec![0i32; D_BITS];
    let mut total_weight: i32 = 0;

    for (i, code) in codes.iter().enumerate() {
        let w = weights[i] as i32;
        if w == 0 {
            continue;
        }
        total_weight += w;
        let bits = unpack_bits(code);
        for (j, &b) in bits.iter().enumerate().take(D_BITS) {
            if b != 0 {
                counts[j] += w;
            }
        }
    }

    if total_weight == 0 {
        return vec![0u8; D_BYTES];
    }

    // Majority: bit is 1 if count*2 > total_weight
    let mut result_bits = vec![0u8; D_BITS];
    for i in 0..D_BITS {
        if counts[i] * 2 > total_weight {
            result_bits[i] = 1;
        }
    }
    pack_bits(&result_bits)
}

/// Flip selected bits in a packed code (XOR).
fn flip_selected_bits(code: &[u8], bit_mask: &[bool]) -> Vec<u8> {
    let bits = unpack_bits(code);
    let mut new_bits = bits.clone();
    for (i, &flip) in bit_mask.iter().enumerate().take(D_BITS) {
        if flip {
            new_bits[i] ^= 1;
        }
    }
    pack_bits(&new_bits)
}

/// Bits where code differs from bundle (per-bit, not per-byte).
fn bits_differ_mask(code: &[u8], bundle: &[u8]) -> Vec<bool> {
    let a = unpack_bits(code);
    let b = unpack_bits(bundle);
    a.iter().zip(b.iter()).take(D_BITS).map(|(x, y)| x != y).collect()
}

/// Bits where code agrees with bundle.
fn bits_agree_mask(code: &[u8], bundle: &[u8]) -> Vec<bool> {
    let a = unpack_bits(code);
    let b = unpack_bits(bundle);
    a.iter().zip(b.iter()).take(D_BITS).map(|(x, y)| x == y).collect()
}

// ─── Token code table ────────────────────────────────────────────────────────

/// Initialize token codes from SHA-256.
fn init_code_table() -> Vec<Vec<u8>> {
    (0..VOCAB)
        .into_par_iter()
        .map(|t| sha_code(&format!("tok|{t}"), D_BYTES))
        .collect()
}

// ─── Training ────────────────────────────────────────────────────────────────

/// Process a single document: tokenize, build bundle, apply bit-flip updates.
/// Returns (n_updates, token_ids).
fn train_document(
    table: &mut [Vec<u8>],
    idf_table: &[i8],
    token_ids: &[u32],
    epoch: usize,
    density: f64,
    neg_freq: &[f64],
    rng: &mut impl Rng,
) -> usize {
    if token_ids.len() < 2 {
        return 0;
    }

    let ids: Vec<usize> = {
        let mut s: Vec<u32> = token_ids.to_vec();
        s.sort_unstable();
        s.dedup();
        s.into_iter().map(|t| t as usize).collect()
    };

    // Build context bundle
    let codes: Vec<&[u8]> = ids.iter().filter_map(|&t| table.get(t).map(|c| c.as_slice())).collect();
    let weights: Vec<i8> = ids.iter()
        .map(|&t| if t < idf_table.len() { idf_table[t] } else { 1 })
        .collect();

    if codes.is_empty() || weights.is_empty() {
        return 0;
    }

    let bundle = bundle_codes(&codes, &weights);
    let doc_set: std::collections::HashSet<usize> = ids.iter().copied().collect();

    let mut n_updates = 0;

    // Positive: pull each token toward bundle
    for &tid in &ids {
        if tid >= table.len() {
            continue;
        }
        let differ = bits_differ_mask(&table[tid], &bundle);
        let mask = sha_flip_mask(&format!("pos|{epoch}|{tid}"), D_BITS, density);
        let flip: Vec<bool> = differ.iter().zip(mask.iter()).map(|(d, m)| *d && *m).collect();
        table[tid] = flip_selected_bits(&table[tid], &flip);
        n_updates += 1;
    }

    // Negative: push random non-co-occurring tokens away
    let total_freq: f64 = neg_freq.iter().sum();
    if total_freq > 0.0 {
        for _ in 0..N_NEG {
            let nt = weighted_sample(neg_freq, rng);
            if doc_set.contains(&nt) || nt >= table.len() {
                continue;
            }
            let agree = bits_agree_mask(&table[nt], &bundle);
            let mask = sha_flip_mask(&format!("neg|{epoch}|{nt}"), D_BITS, density);
            let flip: Vec<bool> = agree.iter().zip(mask.iter()).map(|(a, m)| *a && *m).collect();
            table[nt] = flip_selected_bits(&table[nt], &flip);
        }
    }

    n_updates
}

fn weighted_sample(weights: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..weights.len());
    }
    let mut r = rng.gen::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

// ─── S3 streaming ────────────────────────────────────────────────────────────

/// List shard files for a dataset, trying .jsonl and .jsonl.gz.
async fn list_shards(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    datasets: &[String],
    max_shards: usize,
) -> Vec<String> {
    let mut all_shards = Vec::new();

    for ds in datasets {
        let prefix = format!("{ds}/");
        let mut shard_count = 0;

        let mut resp = client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(&prefix)
            .send()
            .await
            .unwrap_or_else(|e| {
                eprintln!("  {ds}: list error — {e}");
                panic!();
            });

        if let Some(objects) = resp.contents {
            for obj in &objects {
                if let Some(key) = &obj.key {
                    if key.ends_with(".jsonl") || key.ends_with(".jsonl.gz") {
                        all_shards.push(key.clone());
                        shard_count += 1;
                    }
                }
            }
        }

        // Handle pagination for large datasets like refinedweb
        while let Some(token) = resp.next_continuation_token.clone() {
            resp = match client
                .list_objects_v2()
                .bucket(bucket)
                .prefix(&prefix)
                .continuation_token(token)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  {ds}: pagination error — {e}");
                    break;
                }
            };
            if let Some(objects) = resp.contents {
                for obj in &objects {
                    if let Some(key) = &obj.key {
                        if key.ends_with(".jsonl") || key.ends_with(".jsonl.gz") {
                            all_shards.push(key.clone());
                            shard_count += 1;
                        }
                    }
                }
            }
        }

        println!("  {ds}: {shard_count} shards", );
    }

    // Shuffle and sample
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    all_shards.shuffle(&mut rng);
    all_shards.truncate(max_shards);
    all_shards
}

/// Download a shard from S3 into a Vec<u8>.
async fn download_shard(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let resp = client.get_object().bucket(bucket).key(key).send().await?;
    let body = resp.body.collect().await?;
    Ok(body.into_bytes().to_vec())
}

/// Process a downloaded shard buffer. Returns list of (token_ids) per document.
fn process_shard_bytes(data: &[u8], is_gzip: bool, tokenizer: &tokenizers::Tokenizer) -> Vec<Vec<u32>> {
    let reader: Box<dyn Read> = if is_gzip {
        Box::new(flate2::read::GzDecoder::new(data))
    } else {
        Box::new(data)
    };

    let buf_reader = BufReader::new(reader);
    let mut results = Vec::new();

    for line in buf_reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let doc: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Extract text
        let text = doc.get("text")
            .or_else(|| doc.get("document"))
            .or_else(|| doc.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if let Some(pos) = doc.get("pos").and_then(|v| v.as_str()) {
                    pos.to_string()
                } else if let (Some(title), Some(body)) = (
                    doc.get("title").and_then(|v| v.as_str()),
                    doc.get("body").and_then(|v| v.as_str()),
                ) {
                    format!("{title} {body}")
                } else {
                    String::new()
                }
            });

        if text.len() < 20 {
            continue;
        }

        if let Ok(enc) = tokenizer.encode(text.as_str(), false) {
            let ids: Vec<u32> = enc.get_ids().iter().take(MAX_TOKENS).copied().collect();
            if ids.len() >= 2 {
                results.push(ids);
            }
        }
    }

    results
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Load R2 credentials from fsspec config
    let config_path = dirs_home().join(".config").join("fsspec").join("s3.json");
    let config_data = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read R2 config at {:?}: {e}", config_path);
            std::process::exit(1);
        });
    let config_json: serde_json::Value = serde_json::from_str(&config_data).unwrap();
    let s3_conf = &config_json["s3"];
    let access_key = s3_conf["key"].as_str().unwrap();
    let secret_key = s3_conf["secret"].as_str().unwrap();
    let endpoint = s3_conf["client_kwargs"]["endpoint_url"].as_str().unwrap();

    // Parse bucket name from endpoint URL
    // endpoint: https://<account_id>.r2.cloudflarestorage.com
    // bucket: "contrastive"
    let bucket = "contrastive";

    println!("loading tokenizer...", );
    let tok_path = dirs_home()
        .join(".drake-memory")
        .join("models")
        .join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).unwrap();
    let tokenizer = Arc::new(tokenizer);

    println!("initializing SHA token codes...", );
    let table = init_code_table();
    let table = Arc::new(Mutex::new(table));
    println!("  {} tokens, {:.1} MB", VOCAB, (VOCAB * D_BYTES) as f64 / 1e6);

    // Configure AWS SDK for R2
    let creds = aws_sdk_s3::config::Credentials::new(
        access_key,
        secret_key,
        None,
        None,
        "fsspec",
    );
    let s3_config = aws_sdk_s3::Config::builder()
        .credentials_provider(creds)
        .endpoint_url(endpoint)
        .region(aws_sdk_s3::config::Region::new("auto"))
        .behavior_version_latest()
        .build();
    let client = aws_sdk_s3::Client::from_conf(s3_config);

    println!("listing shards from {:?}...", args.datasets);
    let shards = list_shards(&client, bucket, &args.datasets, args.shards).await;
    println!("  sampled {} shards", shards.len());

    if shards.is_empty() {
        eprintln!("ERROR: no shards found. Check R2 credentials.");
        std::process::exit(1);
    }

    // First pass: compute IDF from sample
    println!("computing IDF from sample...", );
    let mut df: HashMap<u32, u32> = HashMap::new();
    let mut n_sample = 0u64;
    let sample_shards = &shards[..shards.len().min(50)];

    for key in sample_shards {
        let is_gzip = key.ends_with(".gz");
        let data = match download_shard(&client, bucket, key).await {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  download error for {key}: {e}");
                continue;
            }
        };
        let docs = process_shard_bytes(&data, is_gzip, &tokenizer);
        for ids in &docs {
            let mut unique: Vec<u32> = ids.clone();
            unique.sort_unstable();
            unique.dedup();
            for id in unique {
                *df.entry(id).or_insert(0) += 1;
            }
            n_sample += 1;
        }
    }
    println!("  IDF computed from {n_sample} docs");

    let n_sample_f = n_sample as f64;
    let idf_table: Vec<i8> = (0..VOCAB)
        .map(|t| {
            let count = *df.get(&(t as u32)).unwrap_or(&1) as f64;
            let val = (n_sample_f / count).log2().round() as i32;
            val.max(1).min(127) as i8
        })
        .collect();

    // Build approximate negative sampling distribution
    let neg_freq: Vec<f64> = (0..VOCAB)
        .map(|t| {
            let c = *df.get(&(t as u32)).unwrap_or(&0) as f64;
            (c + 1.0).powf(0.75)
        })
        .collect();

    // Training loop
    let total_docs = AtomicU64::new(0);

    for epoch in 0..args.epochs {
        let t0 = std::time::Instant::now();
        let density = (args.lr * (0.5_f64.powi(epoch as i32))).max(0.001);
        let epoch_docs = AtomicU64::new(0);
        let epoch_updates = AtomicU64::new(0);

        for (si, key) in shards.iter().enumerate() {
            let is_gzip = key.ends_with(".gz");

            // Download shard
            let data = match download_shard(&client, bucket, key).await {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  download error for {key}: {e}");
                    continue;
                }
            };

            // Tokenize all documents (synchronous, fast)
            let docs = process_shard_bytes(&data, is_gzip, &tokenizer);

            // Train: process each document, updating the shared table
            for ids in &docs {
                let mut rng = rand::thread_rng();
                let mut tbl = table.lock().unwrap();
                let n = train_document(&mut tbl, &idf_table, ids, epoch, density, &neg_freq, &mut rng);
                epoch_updates.fetch_add(n as u64, Ordering::Relaxed);
                epoch_docs.fetch_add(1, Ordering::Relaxed);
            }

            if (si + 1) % 5 == 0 || si == 0 {
                let docs = epoch_docs.load(Ordering::Relaxed);
                let updates = epoch_updates.load(Ordering::Relaxed);
                let elapsed = t0.elapsed().as_secs();
                println!("  epoch {}/{}: shard {}/{} ({} docs, {} updates, {}s)",
                    epoch + 1, args.epochs, si + 1, shards.len(),
                    docs, updates, elapsed);
            }
        }

        let docs = epoch_docs.load(Ordering::Relaxed);
        let updates = epoch_updates.load(Ordering::Relaxed);
        total_docs.fetch_add(docs, Ordering::Relaxed);
        println!("epoch {}/{} done: {} docs, {} updates, density={:.4} ({}s)",
            epoch + 1, args.epochs, docs, updates, density,
            t0.elapsed().as_secs());
    }

    // Save as .npy-compatible .npz
    let table = table.lock().unwrap();
    let results_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let out_path = results_dir.join("token_codes_s3_rust.npz");

    // Write as simple binary (codes + idf) — Python can load with numpy
    save_npz(&out_path, &table, &idf_table);
    println!("\nsaved {:?} ({:.1} MB)", out_path, (VOCAB * D_BYTES) as f64 / 1e6);
    println!("total docs processed: {}", total_docs.load(Ordering::Relaxed));
    println!("\nTo deploy: copy to ~/.drake-memory/models/token_codes.npz");
    println!("Then set DM_EMBEDDING_BACKEND=hamming");
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

/// Save as a ZIP (.npz) with two .npy files: codes (V, D//8) uint8, idf (V,) int8.
fn save_npz(path: &std::path::Path, codes: &[Vec<u8>], idf: &[i8]) {
    use std::io::Write;
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // codes.npy: shape (V, D_BYTES), dtype |u1
    zip.start_file("codes.npy", options).unwrap();
    write_npy_header(&mut zip, &[VOCAB, D_BYTES], "|u1", 1);
    for row in codes {
        zip.write_all(row).unwrap();
    }

    // idf.npy: shape (V,), dtype |i1
    zip.start_file("idf.npy", options).unwrap();
    write_npy_header(&mut zip, &[VOCAB], "|i1", 1);
    let idf_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(idf.as_ptr() as *const u8, idf.len())
    };
    zip.write_all(idf_bytes).unwrap();

    zip.finish().unwrap();
}

fn write_npy_header(zip: &mut zip::ZipWriter<std::fs::File>, shape: &[usize], descr: &str, _elem_size: usize) {
    use std::io::Write;
    let shape_str: String = shape.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    // Pad to make total header (including magic) a multiple of 64
    let header = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_str},), }}",
    );
    let header_len = header.len();
    // Magic: \x93NUMPY + version 1 + header_len (u16 LE) + header (padded to 64)
    let total = 10 + header_len;
    let pad = (64 - (total % 64)) % 64;
    let padded_len = header_len + pad;

    zip.write_all(b"\x93NUMPY").unwrap();
    zip.write_all(&[1u8, 0]).unwrap(); // version 1.0
    zip.write_all(&(padded_len as u16).to_le_bytes()).unwrap();
    zip.write_all(header.as_bytes()).unwrap();
    // Pad with spaces, ending with newline
    let padding = " ".repeat(pad - 1) + "\n";
    zip.write_all(padding.as_bytes()).unwrap();
}
