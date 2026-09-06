"""Command-line entry point for assignment scripts and experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

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
    with open("artifacts/tinystories_merges.txt", "w") as f:
        f.writelines(f"{a.decode('latin-1')} {b.decode('latin-1')}\n" for a, b in merges)
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
    raise NotImplementedError


def cmd_profile_merge(_args: argparse.Namespace) -> None:
    """Profile the BPE merge loop."""
    raise NotImplementedError


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

    profile_merge = subparsers.add_parser(
        "profile-merge",
        help="profile the BPE merge loop",
    )
    profile_merge.set_defaults(func=cmd_profile_merge)

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
