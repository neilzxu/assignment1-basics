"""Command-line entry point for assignment scripts and experiments."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from jsonargparse import ArgumentParser

from cs336_basics import bpe, train, transformer
from cs336_basics.benchmark import run_with_stats
from cs336_basics.config import TrainConfig


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
) -> tuple[Path, Path, int]:
    """Tokenize a text corpus and write raw and NumPy token arrays."""
    text_path = Path(text_path)
    bin_path = text_path.with_suffix(".bin")
    npy_path = text_path.with_suffix(".npy")
    text = text_path.read_text(encoding="utf-8")
    tokens = np.asarray(tokenizer.encode(text), dtype=np.uint16)

    tokens.tofile(bin_path)
    np.save(npy_path, tokens)
    token_count = tokens.size

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


def cmd_sgd_toy(_args: argparse.Namespace) -> None:
    init_val = 5 * torch.randn((10, 10))
    # small lr (1, 10) too small to get close. Large lr ( 1e3) get close super fast but then just get worse. 1e2 is good though
    for lr in [1, 10, 1e2, 1e3]:
        weights = torch.nn.Parameter(init_val)
        opt = transformer.SGD([weights], lr=lr)
        losses = []
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            loss.backward()  # Run backward pass, which computes gradients.
            losses.append(loss.item())
            opt.step()  # Run optimizer step.
        print(f"(lr {lr}) has losses {losses}")


def cmd_train(config: TrainConfig) -> None:
    train.train_lm(config)


def cmd_generate(_args: argparse.Namespace) -> None:
    """Generate text from a trained checkpoint."""
    raise NotImplementedError


COMMAND_HANDLERS = {
    "train-bpe-tinystories": cmd_train_bpe_tinystories,
    "train-bpe-owt": cmd_train_bpe_owt,
    "tok-exp": cmd_tokenizer_experiments,
    "sgd-toy": cmd_sgd_toy,
    "train": cmd_train,
    "generate": cmd_generate,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="cs336",
        description="CS336 assignment scripts and experiments",
    )
    subcommands = parser.add_subcommands(required=True)

    for name in COMMAND_HANDLERS:
        command_parser = ArgumentParser()

        if name == "train":
            command_parser.add_argument("--config", action="config")
            command_parser.add_class_arguments(TrainConfig)

        subcommands.add_subcommand(name, command_parser)

    return parser


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         prog="cs336",
#         description="CS336 assignment scripts and experiments",
#     )
#     subparsers = parser.add_subparsers(dest="command", required=True)
#
#     train_bpe_tinystories = subparsers.add_parser(
#         "train-bpe-tinystories",
#         help="train BPE on TinyStories (Problem train_bpe_tinystories)",
#     )
#     train_bpe_tinystories.set_defaults(func=cmd_train_bpe_tinystories)
#
#     train_bpe_owt = subparsers.add_parser(
#         "train-bpe-owt",
#         help="train BPE on OpenWebText (Problem train_bpe_expts_owt)",
#     )
#     train_bpe_owt.set_defaults(func=cmd_train_bpe_owt)
#     tok_exp = subparsers.add_parser(
#         "tok-exp",
#         help="Experiments w/ tokenizers",
#     )
#     tok_exp.set_defaults(func=cmd_tokenizer_experiments)
#     sgd_toy = subparsers.add_parser(
#         "sgd-toy",
#         help="Toy SGD example",
#     )
#     sgd_toy.set_defaults(func=cmd_sgd_toy)
#
#     train = subparsers.add_parser(
#         "train",
#         help="train a transformer LM (Problem training_together)",
#     )
#     train.set_defaults(func=cmd_train)
#
#     generate = subparsers.add_parser(
#         "generate",
#         help="generate text from a trained model",
#     )
#     generate.set_defaults(func=cmd_generate)
#
#     return parser
#


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    instantiated = parser.instantiate(parsed)

    command = instantiated.subcommand
    command_args = instantiated[command]

    if command == "train":
        config_f = TrainConfig(
            data=command_args.data,
            model=command_args.model,
            optimizer=command_args.optimizer,
            training=command_args.training,
            validation=command_args.validation,
            checkpoint=command_args.checkpoint,
        )
        config_f.validate()
        cmd_train(config_f)
    else:
        COMMAND_HANDLERS[command](command_args)


if __name__ == "__main__":
    main()
