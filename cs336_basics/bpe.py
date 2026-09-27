from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, BinaryIO

import regex as re


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_chunk(input_path, special_tokens: list[str], start: int, end: int):
    """Process a chunk of text into a map of counts of pairs of bytes"""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_chunks = re.split("|".join([re.escape(st) for st in special_tokens]), chunk)

        bp_map = {}
        pretoken_map = {}
        for split_chunk in split_chunks:
            for match in re.finditer(_PAT, split_chunk):
                pretoken = match.group(0)
                if pretoken in pretoken_map:
                    bytestr, prev_ct = pretoken_map[pretoken]
                    pretoken_map[pretoken][1] = prev_ct + 1
                else:
                    bytestr = pretoken.encode("utf-8")
                    pretoken_map[pretoken] = [list(bytestr), 1]
        for pretoken, [bytestr, ct] in pretoken_map.items():
            for i in range(len(bytestr) - 1):
                cur_idx = bytestr[i]
                next_idx = bytestr[i + 1]
                pair = (cur_idx, next_idx)
                bp_map[pair] = bp_map.get(pair, 0) + ct
    return bp_map, pretoken_map


def _decr_del(
    rem_key: tuple[int, int],
    tid_pair_ct_map: dict[tuple[int, int], int],
    occurences: int = 1,
    rem_neg=True,
) -> None:
    tid_pair_ct_map[rem_key] = tid_pair_ct_map.get(rem_key, 0) - occurences
    if tid_pair_ct_map[rem_key] <= 0 and rem_neg:
        del tid_pair_ct_map[rem_key]


def _incr(add_key: tuple[int, int], tid_pair_ct_map: dict[tuple[int, int], int], occurences: int = 1):
    tid_pair_ct_map[add_key] = tid_pair_ct_map.get(add_key, 0) + occurences


def update_and_count_idx_sent_list(
    idx_pair: tuple[int, int],
    repl_idx: int,
    i: int,
    count_map: dict[tuple[int, int], int],
    pair_i_delta: set[tuple[bool, tuple[int, int], int]],
    idx_sent_corpus: list[tuple[list[int], int]],
) -> None:
    idx_1, idx_2 = idx_pair

    [idx_sent, occurences, sent_count_map] = idx_sent_corpus[i]

    count_delta = {}
    write_i = 0
    match_flag = False
    for read_i in range(len(idx_sent)):
        cur_idx = idx_sent[read_i]
        if match_flag:
            if cur_idx == idx_2:
                idx_sent[write_i] = repl_idx
                # update count_map:
                if write_i > 0:
                    prev_idx = idx_sent[write_i - 1]
                    rem_key = (prev_idx, idx_1)
                    count_delta[rem_key] = count_delta.get(rem_key, 0) - 1
                    add_key = (prev_idx, repl_idx)
                    count_delta[add_key] = count_delta.get(add_key, 0) + 1

                if read_i < len(idx_sent) - 1:
                    next_idx = idx_sent[read_i + 1]
                    rem_key = (idx_2, next_idx)
                    count_delta[rem_key] = count_delta.get(rem_key, 0) - 1
                    add_key = (repl_idx, next_idx)
                    count_delta[add_key] = count_delta.get(add_key, 0) + 1
                count_delta[idx_pair] = count_delta.get(idx_pair, 0) - 1
                write_i += 1
                match_flag = False
            elif cur_idx == idx_1:
                idx_sent[write_i] = idx_1
                write_i += 1
            else:
                idx_sent[write_i] = idx_1
                idx_sent[write_i + 1] = cur_idx
                write_i += 2
                match_flag = False

        elif cur_idx == idx_1:
            match_flag = True
        else:
            idx_sent[write_i] = cur_idx
            write_i += 1
    if match_flag:
        idx_sent[write_i] = idx_1
        write_i += 1

    # update based on delta
    for pair, delta in count_delta.items():
        count_map[pair] = count_map.get(pair, 0) + delta * occurences
        if pair not in sent_count_map and delta > 0:
            pair_i_delta.add((True, pair, i))
        sent_count_map[pair] = sent_count_map.get(pair, 0) + delta
        if sent_count_map[pair] <= 0:
            del sent_count_map[pair]
            pair_i_delta.add((False, pair, i))

    del idx_sent[write_i:]


BYTE_UB = 256


def diff_pair_count_maps(
    got: dict[tuple[int, int], int] | Counter[tuple[int, int]],
    want: dict[tuple[int, int], int] | Counter[tuple[int, int]],
    idx_vocab_map: dict[int, bytes],
    *,
    label: str = "got vs want",
    limit: int | None = 20,
) -> list[tuple[tuple[bytes, bytes], int, int, int]]:
    """Print pair-count differences between two maps keyed by token-id pairs.

    Returns a list of (bytes_pair, got_count, want_count, delta) for each mismatch.
    """
    all_pairs = set(got) | set(want)
    mismatches = []
    for pair in all_pairs:
        got_count = got.get(pair, 0)
        want_count = want.get(pair, 0)
        if got_count != want_count:
            bpair = (idx_vocab_map[pair[0]], idx_vocab_map[pair[1]])
            mismatches.append((bpair, got_count, want_count, got_count - want_count))

    mismatches.sort(key=lambda row: (abs(row[3]), row[0]), reverse=True)

    print(f"pair count diff ({label}): {len(mismatches)} mismatches")
    for bpair, got_count, want_count, delta in mismatches[:limit]:
        print(f"  {bpair!r}: got={got_count}, want={want_count}, delta={delta:+d}")
    if limit is not None and len(mismatches) > limit:
        print(f"  ... {len(mismatches) - limit} more")

    return mismatches


@dataclass(order=False, slots=True)
class MergeCandidate:
    count: int
    pair: tuple[int, int]
    vpair: tuple[bytes, bytes]

    def __lt__(self, other: MergeCandidate) -> bool:
        if self.count != other.count:
            return self.count > other.count  # higher count pops first
        return self.vpair > other.vpair  # tie: lex-greater pair pops first


def _process_chunk(args):
    return process_chunk(*args)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 12,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_token_bytestrs = set([token.encode("utf-8") for token in special_tokens])
    with open(input_path, "rb") as f:
        boundaries = list(find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8")))

    bp_count_map = {}
    pretoken_agg_map = {}

    with mp.Pool(processes=num_processes) as p:
        for bp_map, pretoken_map in p.imap_unordered(
            _process_chunk,
            [(input_path, special_tokens, start, end) for start, end in pairwise(boundaries)],
        ):
            for k, v in bp_map.items():
                bp_count_map[k] = bp_count_map.get(k, 0) + v
            for pretoken, [bytestr, ct] in pretoken_map.items():
                if pretoken in pretoken_agg_map:
                    pretoken_agg_map[pretoken][1] += ct
                else:
                    pretoken_agg_map[pretoken] = [bytestr, ct, Counter(pairwise(bytestr))]

    idx_sent_corpus = list(pretoken_agg_map.values())
    pair_i_map = {}
    for idx, [_, _, pair_count_map] in enumerate(idx_sent_corpus):
        for pair in pair_count_map:
            if pair not in pair_i_map:
                pair_i_map[pair] = set()
            pair_i_map[pair].add(idx)

    vocab_idx_map = {bytes([i]): i for i in range(BYTE_UB)}
    idx_vocab_map = {i: bytes([i]) for i in range(BYTE_UB)}
    next_idx = BYTE_UB
    pair_heap = [
        MergeCandidate(count=count, vpair=(idx_vocab_map[pair[0]], idx_vocab_map[pair[1]]), pair=pair)
        for pair, count in bp_count_map.items()
    ]
    heapq.heapify(pair_heap)
    merges = []
    while next_idx < vocab_size - len(special_tokens) and pair_heap:
        # for pair, i_set in pair_i_map.items():
        #     for i in i_set:
        #         assert pair in idx_sent_corpus[i][2], (pair, i)
        # for i, [_, _, sent_count_map] in enumerate(idx_sent_corpus):
        #     for pair in sent_count_map:
        #         assert i in pair_i_map[pair], (pair, i)

        while pair_heap:
            mc = heapq.heappop(pair_heap)
            actual_count = bp_count_map.get(mc.pair, 0)
            if actual_count > 0 and actual_count == mc.count:
                break
        else:
            break

        idx_1, idx_2 = mc.pair
        vocab_1, vocab_2 = (idx_vocab_map[idx_1], idx_vocab_map[idx_2])
        new_vocab = vocab_1 + vocab_2

        assert new_vocab not in special_token_bytestrs
        if new_vocab in special_token_bytestrs:
            continue
        merges.append((vocab_1, vocab_2))

        if new_vocab not in vocab_idx_map:
            vocab_idx_map[new_vocab] = next_idx
            idx_vocab_map[next_idx] = new_vocab

        agg_ct_map = {}
        pair_i_delta = set()
        for i in pair_i_map[mc.pair]:
            update_and_count_idx_sent_list(
                mc.pair, vocab_idx_map[new_vocab], i, agg_ct_map, pair_i_delta, idx_sent_corpus
            )

        for update_dir, pair, i in pair_i_delta:
            if update_dir:  # increment
                if pair not in pair_i_map:
                    pair_i_map[pair] = set()
                pair_i_map[pair].add(i)
            else:  # decrement
                if pair in pair_i_map and i in pair_i_map[pair]:
                    pair_i_map[pair].remove(i)
                    if not pair_i_map[pair]:
                        del pair_i_map[pair]

        for pair, delta in agg_ct_map.items():
            if delta != 0:
                bp_count_map[pair] = bp_count_map.get(pair, 0) + delta
                assert bp_count_map[pair] >= 0
                if bp_count_map[pair] == 0:
                    del bp_count_map[pair]
                else:
                    heapq.heappush(
                        pair_heap,
                        MergeCandidate(
                            pair=pair,
                            vpair=(idx_vocab_map[pair[0]], idx_vocab_map[pair[1]]),
                            count=bp_count_map[pair],
                        ),
                    )

        next_idx += 1

    idx_vocab_map = {
        **idx_vocab_map,
        **{vocab_size - 1 - i: st.encode("utf-8") for i, st in enumerate(special_tokens[::-1])},
    }
    return idx_vocab_map, merges


def apply_merge(tid_pair: tuple[int, int], repl_tid: int, tid_list: list[int]) -> dict[tuple[int, int], int]:
    tid_1, tid_2 = tid_pair
    write_i = 0
    match_flag = False
    delta_map = {}
    for read_i in range(len(tid_list)):
        cur_idx = tid_list[read_i]
        if match_flag:
            if cur_idx == tid_2:
                tid_list[write_i] = repl_tid

                if write_i > 0:
                    prev_idx = tid_list[write_i - 1]
                    rem_key = (prev_idx, tid_1)
                    delta_map[rem_key] = delta_map.get(rem_key, 0) - 1
                    add_key = (prev_idx, repl_tid)
                    delta_map[add_key] = delta_map.get(add_key, 0) + 1

                if read_i < len(tid_list) - 1:
                    next_idx = tid_list[read_i + 1]
                    rem_key = (tid_2, next_idx)
                    delta_map[rem_key] = delta_map.get(rem_key, 0) - 1
                    add_key = (repl_tid, next_idx)
                    delta_map[add_key] = delta_map.get(add_key, 0) + 1
                delta_map[tid_pair] = delta_map.get(tid_pair, 0) - 1
                write_i += 1
                match_flag = False
            elif cur_idx == tid_1:
                tid_list[write_i] = tid_1
                write_i += 1
            else:
                tid_list[write_i] = tid_1
                tid_list[write_i + 1] = cur_idx
                write_i += 2
                match_flag = False

        elif cur_idx == tid_1:
            match_flag = True
        else:
            tid_list[write_i] = cur_idx
            write_i += 1
    if match_flag:
        tid_list[write_i] = tid_1
        write_i += 1

    del tid_list[write_i:]
    return delta_map


def _encode_pretoken(pretoken, vocab_idx_map, tid_pair_rank_map: dict[tuple[int, int], tuple[int, int]], cache):
    if pretoken in cache:
        return cache[pretoken]
    tid_list = [vocab_idx_map[bytes([x])] for x in pretoken.encode("utf-8")]
    tid_pair_ct_map = Counter(pairwise(tid_list))

    rank_heap = [
        (tid_pair_rank_map[pair][0], pair, tid_pair_rank_map[pair][1])
        for pair in tid_pair_ct_map
        if pair in tid_pair_rank_map
    ]
    heapq.heapify(rank_heap)
    while rank_heap:
        _, tid_pair, repl_tid = heapq.heappop(rank_heap)
        if tid_pair not in tid_pair_ct_map or tid_pair_ct_map[tid_pair] <= 0:
            continue
        delta_map = apply_merge(tid_pair=tid_pair, repl_tid=repl_tid, tid_list=tid_list)
        for tid_pair, delta in delta_map.items():
            if tid_pair not in tid_pair_ct_map and delta > 0 and tid_pair in tid_pair_rank_map:
                rank, repl_tid = tid_pair_rank_map[tid_pair]
                heapq.heappush(rank_heap, (rank, tid_pair, repl_tid))
            tid_pair_ct_map[tid_pair] = tid_pair_ct_map.get(tid_pair, 0) + delta
            if tid_pair_ct_map[tid_pair] <= 0:
                del tid_pair_ct_map[tid_pair]
    cache[pretoken] = tid_list
    return tid_list


from tqdm import tqdm


def _encode_text(text, sorted_special_tokens: list[str] | None, vocab_idx_map, tid_pair_rank_map):
    """Process a chunk of text into a map of counts of pairs of bytes"""
    tids = []
    cache = {}
    if sorted_special_tokens is None or not sorted_special_tokens:
        for match in re.finditer(_PAT, text):
            pretoken = match.group()
            tids.extend(_encode_pretoken(pretoken, vocab_idx_map, tid_pair_rank_map, cache))
    else:
        special_re = re.compile("|".join([re.escape(st) for st in sorted_special_tokens]))
        cursor = 0
        for special in tqdm(special_re.finditer(text), desc="EOT processed"):
            special_token = special.group()
            for match in re.finditer(_PAT, text, pos=cursor, endpos=special.start()):
                pretoken = match.group()
                tid_list = _encode_pretoken(pretoken, vocab_idx_map, tid_pair_rank_map, cache)
                tids.extend(list(tid_list))
            tids.append(vocab_idx_map[special_token.encode("utf-8")])
            cursor = special.end()
        for match in re.finditer(_PAT, text, pos=cursor):
            pretoken = match.group()
            tid_list = _encode_pretoken(pretoken, vocab_idx_map, tid_pair_rank_map, cache)
            tids.extend(tid_list)
    return tids


@dataclass
class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def __post_init__(self):
        self._vocab_idx_map = {v: k for k, v in self.vocab.items()}
        if not self.special_tokens is None:
            self._sorted_special_tokens = sorted(self.special_tokens, key=len)[::-1]
        else:
            self._sorted_special_tokens = None

        self._tid_pair_rank_map = {
            (self._vocab_idx_map[token_1], self._vocab_idx_map[token_2]): (rank, self._vocab_idx_map[token_1 + token_2])
            for rank, (token_1, token_2) in enumerate(self.merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath) as in_f:
            idx_vocab_map = json.load(in_f)
        with open(merges_filepath) as in_f:
            merges = json.load(in_f)
        vocab_idx_map = {int(k): v.encode("latin-1") for k, v in idx_vocab_map.items()}
        merges_bytestrs = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in merges]
        return cls(vocab_idx_map, merges_bytestrs, special_tokens)

    def encode(self, text: str) -> list[int]:
        return _encode_text(text, self._sorted_special_tokens, self._vocab_idx_map, self._tid_pair_rank_map)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        joined_bytes = b"".join(self.vocab[id] for id in ids)
        return (joined_bytes).decode("utf-8", errors="replace")
