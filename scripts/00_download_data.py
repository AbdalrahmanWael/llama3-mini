"""
Data download script for llama3-mini
Downloads FineWeb-Edu, TinyStories, UltraChat (or anything in the YAML).
"""

import argparse
import itertools
import os
from pathlib import Path

import yaml
from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


def load_config(config_path: str = "configs/data.yml") -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    """Create directory (recursively) if missing."""
    path.mkdir(parents=True, exist_ok=True)


def _hf_name_from_url(url_or_name: str) -> str:
    """
    Accept either a plain dataset name or a full
    https://huggingface.co/datasets/<name> URL and always return <name>.
    """
    if "huggingface.co/datasets/" in url_or_name:
        return url_or_name.split("/datasets/")[-1]
    return url_or_name


def download_dataset(dataset_cfg: dict, data_dir: Path) -> bool:
    """
    Download (or sample) one dataset and save to Parquet.
    Returns True on success.
    """
    name = dataset_cfg["name"]
    split = dataset_cfg.get("split", "train")
    max_samples = dataset_cfg.get("max_samples")  # can be None
    hf_name = _hf_name_from_url(dataset_cfg["url"])

    print(f" Downloading {name} ({hf_name}, split='{split}')")

    try:
        if max_samples:  
            it_ds: IterableDataset = load_dataset(
                hf_name,
                split=split,
                streaming=True,
                cache_dir=data_dir / "cache",
            )
            # materialise only the number of samples we need
            samples = list(itertools.islice(it_ds, max_samples))
            ds: Dataset = Dataset.from_list(samples)
        else:  
            ds: Dataset = load_dataset(
                hf_name,
                split=split,
                streaming=False,
                cache_dir=data_dir / "cache",
            )

        out_path = data_dir / f"{name}.parquet"
        ds.to_parquet(str(out_path))

        # table: pa.Table = ds.data 
        #
        # pq.write_table(
        #     table,
        #     str(out_path),
        #     # compression="snappy",  
        #     use_threads=False,
        # )

        print(
            f"{name}: {len(ds):,} samples → {out_path} "
            f"({out_path.stat().st_size/1024/1024:.1f} MB)"
        )
        print(f"    columns: {ds.column_names}")
        return True

    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return False


def verify_downloads(data_dir: Path, cfg: dict):
    """Simple existence / size check."""
    print("\nVerifying downloads …")
    for ds_cfg in cfg["data"]["datasets"]:
        fp = data_dir / f"{ds_cfg['name']}.parquet"
        if fp.exists():
            print(
                f"   {ds_cfg['name']}: {fp.stat().st_size/1024/1024:.1f} MB (ok)"
            )
        else:
            print(f"   {ds_cfg['name']}: missing")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for llama3-mini")
    parser.add_argument("--config", default="configs/data.yml", help="YAML config file")
    parser.add_argument(
        "--data-dir", default="./storage/raw_datasets/", help="Where to store parquet + cache"
    )
    parser.add_argument(
        "--dataset",
        help="If supplied, download only this dataset name (must match YAML 'name')",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir).expanduser()
    ensure_dir(data_dir)

    print(f"Data directory: {data_dir.resolve()}\n")

    # decide what to download
    all_ds = cfg["data"]["datasets"]
    if args.dataset:
        all_ds = [d for d in all_ds if d["name"] == args.dataset]
        if not all_ds:
            print(f"Dataset '{args.dataset}' not found in {args.config}!")
            return

    success = 0
    for ds_cfg in all_ds:
        if download_dataset(ds_cfg, data_dir):
            success += 1

    print(f"\nDownloaded {success}/{len(all_ds)} datasets")
    verify_downloads(data_dir, cfg)


if __name__ == "__main__":
    main()
