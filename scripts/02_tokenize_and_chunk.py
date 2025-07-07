import argparse, os, yaml, math, gzip, numpy as np, sentencepiece as spm, datasets, tqdm, random

def open_dataset(ds_cfg, cache_dir):
    if "path" in ds_cfg:                       # ── LOCAL FILE ──
        fmt = ds_cfg.get("format", "parquet")  # parquet / json / text …
        return datasets.load_dataset(
            fmt,
            data_files=ds_cfg["path"],
            split=ds_cfg.get("split", "train")
        )
    else:                                      # ── HUB REPO ID ──
        return datasets.load_dataset(
            ds_cfg["repo"] if "repo" in ds_cfg else ds_cfg["url"],
            split=ds_cfg.get("split", "train"),
            cache_dir=cache_dir,
            streaming=True
        )

def encode_example(batch, sp, eos_id):
    return {"ids": [sp.encode(t.strip()) + [eos_id] for t in batch["text"]]}

def stream_tokens(ds_cfg, sp, cache_dir):
    # dset = datasets.load_dataset(
    #     ds_cfg["path"],
    #     split=ds_cfg.get("split", "train"),
    #     cache_dir=cache_dir,
    #     streaming=True,
    # )
    dset = open_dataset(ds_cfg, cache_dir)

    max_samp = ds_cfg.get("max_samples", None)
    end_id = sp.piece_to_id("<|endoftext|>")        
    for i, row in enumerate(dset):
        yield from sp.encode(row["text"].strip()) + [end_id]
        if max_samp and (i + 1) >= max_samp:
            break

def write_npz(arr, split_dir, file_idx):
    path = os.path.join(split_dir, f"tokens_{file_idx:03d}.npz")
    np.savez_compressed(path, input_ids=arr.astype(np.uint16))

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    tk_path   = cfg["paths"]["tokenizer_path"] + "/tokenizer.model"
    out_root  = cfg["paths"]["processed_data"]
    cache_dir = cfg["paths"]["cache_dir"]
    os.makedirs(out_root, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=tk_path)
    max_len   = cfg["data"]["preprocessing"]["max_length"]
    stride    = cfg["data"]["preprocessing"]["stride"]
    min_len = cfg["data"]["preprocessing"].get("min_length", 0)
    chunk_sz  = cfg["data"]["storage"]["chunk_size"]

    train_ratio = cfg["data"]["preprocessing"]["train_split"]

    for ds_cfg in cfg["data"]["datasets"]:
        name = ds_cfg["name"]
        if ds_cfg.get("weight", 1.0) == 0.0:
            continue   # this dataset is only for later chat FT

        print(f"▶ Tokenising {name}")
        ids_stream = list(stream_tokens(ds_cfg, sp, cache_dir))
        if len(ids_stream) < min_len:
            print(f"!! {name} too small, skipping")
            continue

        # slide window
        chunks = []
        i = 0
        while i + max_len <= len(ids_stream):
            chunk = ids_stream[i : i + max_len]
            chunks.append(chunk)
            i += stride
        random.shuffle(chunks)

        # split train / val
        split_idx = int(len(chunks) * train_ratio)
        splits = {"train": chunks[:split_idx], "val": chunks[split_idx:]}

        # write out as npz
        for split, seqs in splits.items():
            split_dir = os.path.join(out_root, split, name)
            os.makedirs(split_dir, exist_ok=True)
            for j in range(0, len(seqs), chunk_sz):
                arr = np.array(seqs[j : j + chunk_sz], dtype=object)
                write_npz(arr, split_dir, j // chunk_sz)
            print(f"Saved {len(seqs)} sequences → {split}/{name}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/data.yaml")
    main(p.parse_args().cfg)
