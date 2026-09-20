from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex as re
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


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
    if (idx_1, idx_2) not in sent_count_map:
        return

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
                    # _decr_del(rem_key, count_map, occurences, rem_neg=False)
                    # _decr_del(rem_key, sent_count_map)
                    add_key = (prev_idx, repl_idx)
                    count_delta[add_key] = count_delta.get(add_key, 0) + 1
                    # _incr(add_key, count_map, occurences)
                    # _incr(add_key, sent_count_map)

                if read_i < len(idx_sent) - 1:
                    next_idx = idx_sent[read_i + 1]
                    rem_key = (idx_2, next_idx)
                    count_delta[rem_key] = count_delta.get(rem_key, 0) - 1
                    # _decr_del(rem_key, count_map, occurences, rem_neg=False)
                    # _decr_del(rem_key, sent_count_map)
                    add_key = (repl_idx, next_idx)
                    count_delta[add_key] = count_delta.get(add_key, 0) + 1
                    # _incr(add_key, count_map, occurences)
                    # _incr(add_key, sent_count_map)
                # _decr_del(idx_pair, count_map, occurences, rem_neg=False)
                # _decr_del(idx_pair, sent_count_map)
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
            update_and_count_idx_sent_list(mc.pair, next_idx, i, agg_ct_map, pair_i_delta, idx_sent_corpus)

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


def apply_merge(
    tid_pair: tuple[int, int], repl_tid: int, tid_list: list[int], tid_pair_ct_map: dict[tuple[int, int], int]
) -> None:
    tid_1, tid_2 = tid_pair
    write_i = 0
    match_flag = False
    for read_i in range(len(tid_list)):
        cur_idx = tid_list[read_i]
        if match_flag:
            if cur_idx == tid_2:
                tid_list[write_i] = repl_tid

                if write_i > 0:
                    prev_idx = tid_list[write_i - 1]
                    rem_key = (prev_idx, tid_1)
                    _decr_del(rem_key, tid_pair_ct_map)
                    add_key = (prev_idx, repl_tid)
                    tid_pair_ct_map[add_key] = tid_pair_ct_map.get(add_key, 0) + 1

                if read_i < len(tid_list) - 1:
                    next_idx = tid_list[read_i + 1]
                    rem_key = (tid_2, next_idx)
                    _decr_del(rem_key, tid_pair_ct_map)
                    add_key = (repl_tid, next_idx)
                    tid_pair_ct_map[add_key] = tid_pair_ct_map.get(add_key, 0) + 1
                _decr_del(tid_pair, tid_pair_ct_map)
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


def _encode_pretoken(pretoken, vocab_idx_map, merges, cache):
    if pretoken in cache:
        return cache[pretoken]
    tid_list = [vocab_idx_map[bytes([x])] for x in pretoken.encode("utf-8")]
    tid_pair_ct_map = Counter(pairwise(tid_list))

    for bytestr_1, bytestr_2 in merges:
        tid_pair = (vocab_idx_map[bytestr_1], vocab_idx_map[bytestr_2])
        repl_tid = vocab_idx_map[bytestr_1 + bytestr_2]
        apply_merge(tid_pair=tid_pair, repl_tid=repl_tid, tid_list=tid_list, tid_pair_ct_map=tid_pair_ct_map)
    cache[pretoken] = tid_list
    return tid_list


def _encode_text(text, sorted_special_tokens: list[str] | None, vocab_idx_map, merges):
    """Process a chunk of text into a map of counts of pairs of bytes"""
    tids = []
    cache = {}
    if sorted_special_tokens is None or not sorted_special_tokens:
        for match in re.finditer(_PAT, text):
            pretoken = match.group()
            tids.extend(_encode_pretoken(pretoken, vocab_idx_map, merges, cache))
    else:
        special_re = re.compile("|".join([re.escape(st) for st in sorted_special_tokens]))
        cursor = 0
        for special in special_re.finditer(text):
            special_token = special.group()
            for match in re.finditer(_PAT, text, pos=cursor, endpos=special.start()):
                pretoken = match.group()
                tid_list = _encode_pretoken(pretoken, vocab_idx_map, merges, cache)
                tids.extend(list(tid_list))
            tids.append(vocab_idx_map[special_token.encode("utf-8")])
            cursor = special.end()
        for match in re.finditer(_PAT, text, pos=cursor):
            pretoken = match.group()
            tid_list = _encode_pretoken(pretoken, vocab_idx_map, merges, cache)
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
        return _encode_text(text, self._sorted_special_tokens, self._vocab_idx_map, self.merges)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        joined_bytes = b"".join(self.vocab[id] for id in ids)
        return (joined_bytes).decode("utf-8", errors="replace")
