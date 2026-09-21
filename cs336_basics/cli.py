"""Command-line entry point for assignment scripts and experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics import bpe
from cs336_basics.benchmark import run_with_stats


def _print_run_stats(label: str, stats) -> None:
    print(f"{label} time: {stats.elapsed_s:.1f}s ({stats.elapsed_s / 60:.1f} min)")
    print(f"{label} peak memory (process tree RSS): {stats.peak_rss_gb:.2f} GB")


def cmd_train_bpe_tinystories(_args: argparse.Namespace) -> None:
    """Train a BPE tokenizer on TinyStories and serialize vocab/merges."""
    (vocab, merges), stats = run_with_stats(
        lambda: bpe.run_train_bpe(
            "data/TinyStoriesV2-GPT4-train.txt",
            vocab_size=10_000,
            special_tokens=["<|endoftext|>"],
        )
    )
    _print_run_stats("train-bpe-tinystories", stats)

    longest_token = max(vocab.values(), key=len)
    print(f"longest token ({len(longest_token)} bytes): {longest_token!r}")

    os.makedirs("artifacts", exist_ok=True)
    vocab_json = {i: token.decode("latin-1") for i, token in vocab.items()}
    with open("artifacts/tinystories_vocab.json", "w") as f:
        json.dump(vocab_json, f, indent=2)
    merge_json = [(a.decode("latin-1"), b.decode("latin-1")) for a, b in merges]
    with open("artifacts/tinystories_merges.json", "w") as f:
        json.dump(merge_json, f, indent=2)
    with open("artifacts/tinystories_train_stats.json", "w") as f:
        json.dump(
            {
                "elapsed_s": stats.elapsed_s,
                "peak_rss_bytes": stats.peak_rss_bytes,
                "longest_token_bytes": len(longest_token),
                "longest_token": longest_token.decode("latin-1"),
                "vocab_size": len(vocab),
                "num_merges": len(merges),
            },
            f,
            indent=2,
        )


def cmd_train_bpe_owt(_args: argparse.Namespace) -> None:
    """Train a BPE tokenizer on OpenWebText and serialize vocab/merges."""
    (vocab, merges), stats = run_with_stats(
        lambda: bpe.run_train_bpe(
            "data/owt_train.txt",
            vocab_size=32_000,
            special_tokens=["<|endoftext|>"],
        )
    )
    _print_run_stats("train-bpe-owt", stats)

    longest_token = max(vocab.values(), key=len)
    print(f"longest token ({len(longest_token)} bytes): {longest_token!r}")

    os.makedirs("artifacts", exist_ok=True)
    vocab_json = {i: token.decode("latin-1") for i, token in vocab.items()}
    with open("artifacts/owt_vocab.json", "w") as f:
        json.dump(vocab_json, f, indent=2)
    merge_json = [(a.decode("latin-1"), b.decode("latin-1")) for a, b in merges]
    with open("artifacts/owt_merges.json", "w") as f:
        json.dump(merge_json, f, indent=2)
    with open("artifacts/owt_train_stats.json", "w") as f:
        json.dump(
            {
                "elapsed_s": stats.elapsed_s,
                "peak_rss_bytes": stats.peak_rss_bytes,
                "longest_token_bytes": len(longest_token),
                "longest_token": longest_token.decode("latin-1"),
                "vocab_size": len(vocab),
                "num_merges": len(merges),
            },
            f,
            indent=2,
        )


def _get_file_doc_ct(file, token="<|endoftext|>", chunk_size=1024 * 1024):
    count = 0
    remainder = ""
    while chunk := file.read(chunk_size):
        text = remainder + chunk
        count += text.count(token)

        # Keep enough characters to detect a token crossing chunks.
        remainder = text[-(len(token) - 1) :]
    return count


def _get_file_doc(file, indices, chunk_size=1024 * 1024):
    separator = "<|endoftext|>"
    requested = list(indices)

    if not requested:
        return []

    if any(index < 0 for index in requested):
        raise ValueError("Indices must be non-negative")

    wanted = set(requested)
    last_wanted = max(wanted)
    found = {}

    buffer = ""
    current_index = 0
    while chunk := file.read(chunk_size):
        buffer += chunk

        while (position := buffer.find(separator)) != -1:
            document = buffer[:position]
            buffer = buffer[position + len(separator) :]

            if current_index in wanted:
                found[current_index] = document

            if current_index >= last_wanted:
                return [found[index] for index in requested]

            current_index += 1

    # Final document if the file doesn't end with the separator.
    if current_index in wanted:
        found[current_index] = buffer

    missing = sorted(wanted - found.keys())
    if missing:
        raise IndexError(f"Documents do not exist: {missing}")

    return [found[index] for index in requested]


def _iter_documents(path, separator="<|endoftext|>", chunk_size=1024 * 1024):
    buffer = ""
    with open(path, encoding="utf-8") as file:
        while chunk := file.read(chunk_size):
            buffer += chunk

            while (position := buffer.find(separator)) != -1:
                yield buffer[: (position + len(separator))]
                buffer = buffer[position + len(separator) :]

    # Final document without a trailing separator
    if buffer:
        yield buffer


def _write_token_ids(
    tokenizer: bpe.Tokenizer,
    text_path: str | os.PathLike,
    *,
    chunk_size: int = 1_000_000,
) -> tuple[Path, Path, int]:
    """Stream a text corpus to a raw binary file, then to a NumPy array file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    text_path = Path(text_path)
    bin_path = text_path.with_suffix(".bin")
    npy_path = text_path.with_suffix(".npy")
    token_ids = iter(tokenizer.encode_iterable(_iter_documents(text_path)))

    token_count = 0
    with bin_path.open("wb") as out_file:
        while True:
            token_chunk = np.fromiter(islice(token_ids, chunk_size), dtype=np.uint16)
            if token_chunk.size == 0:
                break
            token_chunk.tofile(out_file)
            token_count += token_chunk.size

    # A .bin file has no dtype or shape metadata. Convert it to a standard .npy
    # file in chunks so the entire corpus is never held in memory.
    numpy_tokens = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint16, shape=(token_count,))
    if token_count:
        raw_tokens = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(token_count,))
        for start in range(0, token_count, chunk_size):
            stop = min(start + chunk_size, token_count)
            numpy_tokens[start:stop] = raw_tokens[start:stop]
        del raw_tokens
    numpy_tokens.flush()
    del numpy_tokens

    return bin_path, npy_path, token_count


def cmd_tokenizer_experiments(_args: argparse.Namespace) -> None:
    data_path_list = [
        (
            "tinystories",
            (
                "data/TinyStoriesV2-GPT4-train.txt",
                "data/TinyStoriesV2-GPT4-valid.txt",
                "artifacts/tinystories_vocab.json",
                "artifacts/tinystories_merges.json",
            ),
        ),
        ("owt", ("data/owt_train.txt", "data/owt_valid.txt", "artifacts/owt_vocab.json", "artifacts/owt_merges.json")),
    ]
    tokenizer_map = {}
    for name, (_, path, vocab_path, merge_path) in data_path_list:
        tokenizer_map[name] = bpe.Tokenizer.from_files(vocab_path, merge_path, ["<|endoftext|>"])

    for name, (train_path, valid_path, _, _) in data_path_list:
        for path in [train_path, valid_path]:
            bin_path, npy_path, token_count = _write_token_ids(tokenizer_map[name], path)
            print(f"encoded {path}: {token_count:,} tokens -> {bin_path} and {npy_path}")


def cmd_train(_args: argparse.Namespace) -> None:
    """Run the transformer LM training loop."""
    raise NotImplementedError


def cmd_generate(_args: argparse.Namespace) -> None:
    """Generate text from a trained checkpoint."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cs336",
        description="CS336 assignment scripts and experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_bpe_tinystories = subparsers.add_parser(
        "train-bpe-tinystories",
        help="train BPE on TinyStories (Problem train_bpe_tinystories)",
    )
    train_bpe_tinystories.set_defaults(func=cmd_train_bpe_tinystories)

    train_bpe_owt = subparsers.add_parser(
        "train-bpe-owt",
        help="train BPE on OpenWebText (Problem train_bpe_expts_owt)",
    )
    train_bpe_owt.set_defaults(func=cmd_train_bpe_owt)
    tok_exp = subparsers.add_parser(
        "tok-exp",
        help="Experiments w/ tokenizers",
    )
    tok_exp.set_defaults(func=cmd_tokenizer_experiments)

    train = subparsers.add_parser(
        "train",
        help="train a transformer LM (Problem training_together)",
    )
    train.set_defaults(func=cmd_train)

    generate = subparsers.add_parser(
        "generate",
        help="generate text from a trained model",
    )
    generate.set_defaults(func=cmd_generate)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
