import argparse, os, sentencepiece as spm, yaml, datasets, tqdm

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def load_dataset_from_cfg(ds_cfg, cache_dir):
    if "path" in ds_cfg:                 # local file (parquet, json, txt, …)
        fmt = ds_cfg.get("format", "parquet")
        return datasets.load_dataset(
            fmt,
            data_files=ds_cfg["path"],
            split=ds_cfg.get("split", "train")
        )
    else:                                # a Hub repo ID
        return datasets.load_dataset(
            ds_cfg["repo"],
            split=ds_cfg.get("split", "train"),
            cache_dir=cache_dir,
            streaming=True
        )

def flatten_text(dset, field="text"):
    for ex in dset:
        txt = ex[field].strip().replace("\n\n", "\n")
        if txt:
            yield txt

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    tk_path = cfg["paths"]["tokenizer_path"]
    os.makedirs(tk_path, exist_ok=True)

    model_file = os.path.join(tk_path, "tokenizer.model")
    if os.path.exists(model_file):
        print("Tokenizer already exists, skip training.")
        return

    # 1) stream all datasets -> tmp file
    tmp_txt = os.path.join(tk_path, "corpus.txt")
    with open(tmp_txt, "w", encoding="utf-8") as fout:
        for ds_cfg in cfg["data"]["datasets"]:

            # dset = datasets.load_dataset(
            #     ds_cfg["url"],
            #     split=ds_cfg.get("split", "train"),
            #     cache_dir=cfg["paths"]["cache_dir"],
            #     streaming=True,
            # )
            dset = load_dataset_from_cfg(ds_cfg, cfg["paths"]["cache_dir"])

            max_samp = ds_cfg.get("max_samples", None)
            for i, ex in enumerate(flatten_text(dset)):
                fout.write(ex + "\n")
                if max_samp and i + 1 >= max_samp:
                    break

    # 2) train SentencePiece
    # spm.SentencePieceTrainer.Train(
    #     input=tmp_txt,
    #     model_prefix=os.path.join(tk_path, "tokenizer"),
    #     vocab_size=cfg["data"]["tokenizer"]["vocab_size"],
    #     model_type=cfg["data"]["tokenizer"]["model_type"],
    #     user_defined_symbols=",".join(SPECIALS),
    #     bos_id=-1,
    #     eos_id=SPECIALS.index("<|endoftext|>"),
    #     pad_id=SPECIALS.index("<|pad|>"),
    # )

    spm.SentencePieceTrainer.Train(
        input=tmp_txt,
        model_prefix=os.path.join(tk_path, "tokenizer"),
        vocab_size=cfg["data"]["tokenizer"]["vocab_size"],
        model_type=cfg["data"]["tokenizer"]["model_type"],
        user_defined_symbols=",".join(SPECIALS),
        bos_id=-1,       # no dedicated BOS
        eos_id=-1,       # no dedicated EOS; we’ll use <|endoftext|>
    )

    print("Tokenizer saved to", tk_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/data.yaml")
    main(parser.parse_args().cfg)
