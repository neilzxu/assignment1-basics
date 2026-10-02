---
author: Neil Xu
date: Sep 30, 2026
---
# Transformer resource accounting


## Transformer LM

Each layer of Transformer LM requires matrix multiplies (among other things). The big matrix multiplies are:
- SwiGLU/SiLU/FFN
- Attention

### SwiGLU

- `(d_ff, d_model) x (d_model, token) ` w1, x
- `(d_ff, d_model) x (d_model, token) ` w3, x
- `(d_model, d_ff) x (d_ff, token)` w2 times activations

So that's `6 x d_ff x d_model x token` FLOPS from matmul

### Attention

- 3 x `(d_model, d_model) x (d_model, token)` for Q, K, V (`6 x token x d_model^2 `)
- 2 x `token` x `d_model // num_heads // 2` x `num_heads` x `(2, 2) x (2, 1)` for RoPE on Q, K. (8 x `token x d_model`)
- `heads` x `(token, d_model // num_heads) x (d_model // num_heads, token)` for the QK transpose computation (`2 x token^2 x d_model`)
- `heads` x `(token, token) x (token, d_model // num_heads)` for the softmax probs V computation (`2 x token^2 x d_model`).

This is a total of `(6 x d_model + 8 + 4 x token) x token x d_model` or `(6 x d_model + 8) x d_model x token + 4 x d_model x token^2` FLOPS

### Decoding

If we do decoding at each step, we have the `(vocab, d_model) x (d_model, token)` which is `2 x vocab x d_model x token` FLOPS.


### Concrete results

We get a total of
```
layers x (6 x d_ff + 6 x d_model + 8 + 4 x token) x d_model x token
+ 2 x vocab x d_model x token
```
FLOPS.

A formula for number of parameters is similar with it being
- `3 x d_ff x d_model` parameters per FFN
- `3 x d_model x d_model` per Attention layer
- `2 x d_model` for LayerNorm per Layer
- 2 x `vocab x d_model` for embedding table and output matrix
- `d_model` for final LayerNorm

So it'd be `(layers x ((3 x (d_ff + d_model) + 2) + 2 x vocab + 1) x d_model `.

With GPT-2 XL setup that's
- 20.58 M params per FFN and 987.9552 M parameters
- 7.68 M params per attention and 368.64 M parameters
- 3.2k params per LayerNorms and 153.6k parameters
- 80.4112 M params per vocab matrix for a total of 160.8224 M parameters
- 1.6k parameters for final LayerNorm

That makes a total of 1.5175728 B parameters. If each is a 32 bit floating point (4 bytes), that's 6.04 Bil bytes or 6.04 GB approximately.

In terms of compute this is:
- So that's 41.1648 MFLOPS per token per layer for FFN. So 1.9759104 GFLOPS per token. Hence a max of 2.0233322496 TFLOPS at max context length.
- 15.3728 MFLOPS per token + 6.4k per token\^2 per layer. That's 737.8944 MFLOPS per token 307.2k per token\^2 across layers. 1.0787504128 TFLOPS at max context length.
- For output layer, we get 160.8224 MFLOPS per token, so a maximum of only 164.6821376 GFLOPS

This gets us a total of 3.2667648 TFLOPS, where essentially 2/3 is FFN and 1/3 is attention if we include the dense matmuls. If we split attention into the actual attention part (QK and AV) vs the embedding dense matmuls we get ~754 GFLOPS from the dense part and hence only ~300 GFLOPS from the attention part. So the computation is dominated by Dense matmuls.

In fact, we get approximately `14 x d_model^2 x layers x token` as FLOPS for dense matmuls and `4 x d_model x layers x token^2` for the attention part.

If we look at this for other models:

| Model              | vocab | ctx   | layers | d_model | d_ff | heads | FFN (TFLOP) | Attn dense (TFLOP) | Attn (TFLOP) | Total (TFLOP) | FFN % | Attn dense % | Attn % |
|--------------------|-------|-------|--------|---------|------|-------|-------------|--------------------|--------------|---------------|-------|--------------|--------|
| GPT2 small         | 50257 | 1024  | 12     | 768     | 2048 | 12    | 0.116       | 0.0436             | 0.0387       | 0.198         | 58.5% | 22.0%        | 19.5%  |
| GPT2 medium        | 50257 | 1024  | 24     | 1024    | 2752 | 16    | 0.416       | 0.155              | 0.103        | 0.673         | 61.7% | 23.0%        | 15.3%  |
| GPT2 large         | 50257 | 1024  | 36     | 1280    | 3392 | 20    | 0.960       | 0.363              | 0.193        | 1.516         | 63.3% | 23.9%        | 12.7%  |
| GPT2-XL            | 50257 | 1024  | 48     | 1600    | 4288 | 25    | 2.023       | 0.756              | 0.322        | 3.101         | 65.2% | 24.4%        | 10.4%  |
| GPT2-XL (long ctx) | 50257 | 16384 | 48     | 1600    | 4288 | 25    | 32.37       | 12.09              | 82.46        | 126.93        | 25.5% | 9.5%         | 65.0%  |

we acn see that increasing the size (and keeping context length fixed) increases the proportion of FLOPs that are from dense matmuls. On the other hand, if we increase the context length, we see that computation becomes dominated by the attention computation.

# AdamW resource accounting

How much memory does training require? It would be something like.

As we know the formula for parameters is `(layers x ((3 x (11/3 d_model) + 2) + 2 x vocab + 1) x d_model`.

Activations, we can break down as follows:
- RMSNorm: `batch x context_length x layer x 2 x d_model`
- Attention: QKVO projections and weighted sum of values: `batch x context_length x layer x 5 x d_model` . `QK^T` and softmax is `batch x 2 x context_length^2 x layer`.
- FFN: W_1 out, W_2 out, SiLU out, elementwise product are a total of `batch x context_length x layers x 4 x d_ff`. Including W_3 out this is `batch x context_length x layers x (4 x 8 / 3 + 1 = 11 2/3) x d_model`.
- Final RMS Norm is just `batch x context_length x d_model`.
- Output embedding is `batch x vocab_size` and the cross entropy term is just `batch x 1`.

So this gets us a total of `batch x ((context_length x layer + 1) x 18 2/3 x d_model + layer x 2 x context_length^2 + vocab_size + 1)` activations.
Gradients are computed on parameters and activations, and optimizer state is only computed for parameters once since the gradients are already accumulated (though it computes 2 state values per param).

So our calculation means that we have `2A + 4P` where A is activations and P is parameters.

This means we use 35.125 times batch plus 26.169 GB, which is batch size 1 under 80 GB assumption.

Running AdamW for one step takes basically 3x forward (forward + backward gradient pass), so about 9.3 TFLOPS. The other optimization stuff scales with the size of parameters which is dominated by the matmul TFLOPS.

400K * 1024 * 9.3 TFLOPS / 250 TFLOPS/s = ~4277 hrs = ~178 days.

