"""
PyTorch Native Self-Attention (reference / CPU-GPU implementation)
===================================================================

This module provides a functionally equivalent PyTorch implementation of the
fused self-attention kernel in `self_attention_nki_kernel.py`.  It is used:

  1. As a readable reference to verify the NKI kernel's correctness.
  2. As a drop-in fallback when running on non-Neuron hardware (CPU / CUDA GPU).
  3. As a training/fine-tuning baseline that can be compared against the
     hardware-optimised NKI version.

Two implementation variants are provided:

  A. ``SelfAttentionNaive``
     Pure PyTorch, no SDPA.  Materialises the full (seq × seq) attention
     score matrix.  Easy to read, mirrors the NKI kernel step by step.

  B. ``SelfAttentionOptimised``
     Uses ``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which
     dispatches to FlashAttention-2 on CUDA ≥ 8.0 and to a memory-efficient
     kernel on older GPUs.  This is the production-grade PyTorch baseline.

Layout convention (same as the NKI kernel wrapper ``flash_self_attn_fwd_bhsd``)
  Input  Q/K/V : (batch, n_heads, seq, d_head)
  Output        : (batch, n_heads, seq, d_head)

Quick usage
-----------
>>> import torch
>>> from self_attention_pytorch import SelfAttentionNaive, SelfAttentionOptimised
>>>
>>> bs, n_heads, seq, d_head = 2, 8, 512, 64
>>> q = torch.randn(bs, n_heads, seq, d_head)
>>> k = torch.randn(bs, n_heads, seq, d_head)
>>> v = torch.randn(bs, n_heads, seq, d_head)
>>>
>>> naive = SelfAttentionNaive(n_heads=n_heads, d_head=d_head)
>>> out_naive = naive(q, k, v, causal=True)
>>>
>>> opt = SelfAttentionOptimised(n_heads=n_heads, d_head=d_head)
>>> out_opt = opt(q, k, v, causal=True)
>>>
>>> torch.allclose(out_naive, out_opt, atol=1e-4)   # True
"""

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# A.  Naive implementation – mirrors the NKI kernel exactly
# ---------------------------------------------------------------------------

class SelfAttentionNaive(nn.Module):
    """
    Pure-PyTorch self-attention with explicit online-softmax–style comments.

    This class deliberately avoids ``F.scaled_dot_product_attention`` so that
    every step is visible and directly corresponds to the NKI kernel stages:

      Stage 1 – QK^T score matrix
      Stage 2 – Causal mask application
      Stage 3 – Softmax (numerically stable, subtract row-max first)
      Stage 4 – Weighted sum with V
    """

    def __init__(
        self,
        n_heads:         int,
        d_head:          int,
        dropout_p:       float = 0.0,
        softmax_scale:   Optional[float] = None,
    ):
        super().__init__()
        self.n_heads       = n_heads
        self.d_head        = d_head
        self.dropout_p     = dropout_p
        self.softmax_scale = softmax_scale or (1.0 / math.sqrt(d_head))
        self.dropout       = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(
        self,
        q:      torch.Tensor,          # (bs, n_heads, seq_q, d_head)
        k:      torch.Tensor,          # (bs, n_heads, seq_k, d_head)
        v:      torch.Tensor,          # (bs, n_heads, seq_v, d_head)
        causal: bool = True,
        attn_bias: Optional[torch.Tensor] = None,  # (bs, n_heads, seq_q, seq_k) additive bias
    ) -> torch.Tensor:
        """
        Compute  O = softmax( scale * Q @ K^T + bias ) @ V.

        Parameters
        ----------
        q, k, v   : (bs, n_heads, seq, d_head) float tensors.
        causal     : If True apply a lower-triangular causal mask.
        attn_bias  : Optional additive bias (e.g. ALiBi slopes).

        Returns
        -------
        out : (bs, n_heads, seq_q, d_head)
        """
        bs, n_heads, seq_q, d_head = q.shape
        seq_k = k.shape[2]

        # ------------------------------------------------------------------
        # Stage 1: Score matrix  S = scale * Q @ K^T
        #   Q : (bs, n_heads, seq_q, d_head)
        #   K : (bs, n_heads, seq_k, d_head)
        #   S : (bs, n_heads, seq_q, seq_k)
        # ------------------------------------------------------------------
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        # scores : (bs, n_heads, seq_q, seq_k)

        # ------------------------------------------------------------------
        # Optional additive bias (ALiBi, relative positional, etc.)
        # ------------------------------------------------------------------
        if attn_bias is not None:
            scores = scores + attn_bias

        # ------------------------------------------------------------------
        # Stage 2: Causal mask
        #   Positions where col > row are future tokens → mask to -inf.
        # ------------------------------------------------------------------
        if causal and seq_q == seq_k:
            # Efficient: create once, reuse
            mask = torch.triu(
                torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float('-inf'))
        elif causal and seq_q != seq_k:
            # Generalised causal mask for cross-attention with different seq lengths
            row_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)   # (seq_q, 1)
            col_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)   # (1,  seq_k)
            mask = col_idx > row_idx                                        # (seq_q, seq_k)
            scores = scores.masked_fill(mask, float('-inf'))

        # ------------------------------------------------------------------
        # Stage 3: Numerically stable softmax
        #   (a) subtract row-max for stability   ← mirrors NKI online-max trick
        #   (b) exponentiate
        #   (c) normalise by row-sum
        # ------------------------------------------------------------------
        row_max  = scores.amax(dim=-1, keepdim=True)         # (bs, n_heads, seq_q, 1)
        exp_s    = torch.exp(scores - row_max)                # (bs, n_heads, seq_q, seq_k)
        row_sum  = exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_w   = exp_s / row_sum                            # (bs, n_heads, seq_q, seq_k)

        attn_w = self.dropout(attn_w)

        # ------------------------------------------------------------------
        # Stage 4: Weighted sum over V
        #   attn_w : (bs, n_heads, seq_q, seq_k)
        #   v      : (bs, n_heads, seq_k, d_head)
        #   out    : (bs, n_heads, seq_q, d_head)
        # ------------------------------------------------------------------
        out = torch.matmul(attn_w, v)
        return out

    def extra_repr(self) -> str:
        return (f"n_heads={self.n_heads}, d_head={self.d_head}, "
                f"scale={self.softmax_scale:.4f}, dropout={self.dropout_p}")


# ---------------------------------------------------------------------------
# B.  Optimised implementation using torch SDPA
# ---------------------------------------------------------------------------

class SelfAttentionOptimised(nn.Module):
    """
    Production-grade self-attention using ``F.scaled_dot_product_attention``.

    On CUDA Ampere+ (sm80+) this transparently uses FlashAttention-2.
    On CPU it uses a memory-efficient implementation.
    On AWS Trainium/Inferentia it falls back to the eager path unless the NKI
    kernel is explicitly used via ``torch_neuronx``.

    API is identical to ``SelfAttentionNaive`` for easy swapping.
    """

    def __init__(
        self,
        n_heads:       int,
        d_head:        int,
        dropout_p:     float = 0.0,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        self.n_heads       = n_heads
        self.d_head        = d_head
        self.dropout_p     = dropout_p
        self.softmax_scale = softmax_scale or (1.0 / math.sqrt(d_head))

    def forward(
        self,
        q:         torch.Tensor,
        k:         torch.Tensor,
        v:         torch.Tensor,
        causal:    bool = True,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
            is_causal=causal if attn_bias is None else False,
            scale=self.softmax_scale,
        )

    def extra_repr(self) -> str:
        return (f"n_heads={self.n_heads}, d_head={self.d_head}, "
                f"scale={self.softmax_scale:.4f}, dropout={self.dropout_p}")


# ---------------------------------------------------------------------------
# C.  Full multi-head attention module (with QKV projections)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Drop-in multi-head self-attention block.

    Includes:
    - Fused QKV linear projection (one matmul instead of three)
    - Optional output projection
    - Choice of naive or SDPA-backed attention core

    Parameters
    ----------
    embed_dim   : model embedding dimension  (= n_heads * d_head)
    n_heads     : number of attention heads
    dropout_p   : attention dropout probability
    use_sdpa    : if True use the optimised SDPA core, else use the naive core
    bias        : include bias in QKV and output projections
    """

    def __init__(
        self,
        embed_dim:   int,
        n_heads:     int,
        dropout_p:   float = 0.0,
        use_sdpa:    bool  = True,
        bias:        bool  = True,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.d_head    = embed_dim // n_heads
        self.dropout_p = dropout_p

        # Fused Q, K, V projection  (3 × embed_dim output = Q + K + V stacked)
        self.qkv_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_core = (
            SelfAttentionOptimised(self.n_heads, self.d_head, dropout_p, softmax_scale)
            if use_sdpa
            else SelfAttentionNaive(self.n_heads, self.d_head, dropout_p, softmax_scale)
        )

    def forward(
        self,
        x:         torch.Tensor,           # (bs, seq, embed_dim)
        causal:    bool = True,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (bs, seq, embed_dim)
        causal     : apply causal mask
        attn_bias  : optional additive attention bias (bs, n_heads, seq, seq)

        Returns
        -------
        out : (bs, seq, embed_dim)
        """
        bs, seq, _ = x.shape

        # ---- QKV projection ------------------------------------------------
        # qkv : (bs, seq, 3 * embed_dim)
        qkv = self.qkv_proj(x)

        # Split and reshape to (bs, n_heads, seq, d_head)
        qkv = qkv.view(bs, seq, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # (3, bs, n_heads, seq, d_head)
        q, k, v = qkv.unbind(dim=0)          # each (bs, n_heads, seq, d_head)

        # ---- Attention core ------------------------------------------------
        attn_out = self.attn_core(q, k, v, causal=causal, attn_bias=attn_bias)
        # attn_out : (bs, n_heads, seq, d_head)

        # ---- Reshape and output projection --------------------------------
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()  # (bs, seq, n_heads, d_head)
        attn_out = attn_out.view(bs, seq, self.embed_dim)       # (bs, seq, embed_dim)
        out = self.out_proj(attn_out)
        return out

    def extra_repr(self) -> str:
        return (f"embed_dim={self.embed_dim}, n_heads={self.n_heads}, "
                f"d_head={self.d_head}, dropout={self.dropout_p}")


# ---------------------------------------------------------------------------
# D.  Numerical accuracy & performance tests
# ---------------------------------------------------------------------------

def _make_qkv(bs, n_heads, seq, d_head, dtype=torch.float32, device="cpu"):
    g = torch.Generator(device=device).manual_seed(42)
    kwargs = dict(dtype=dtype, device=device, generator=g)
    q = torch.randn(bs, n_heads, seq, d_head, **kwargs) * 0.02
    k = torch.randn(bs, n_heads, seq, d_head, **kwargs) * 0.02
    v = torch.randn(bs, n_heads, seq, d_head, **kwargs) * 0.02
    return q, k, v


def test_naive_vs_sdpa(
    bs=2, n_heads=4, seq=128, d_head=64, causal=True, atol=1e-4
):
    """Verify SelfAttentionNaive ≈ SelfAttentionOptimised."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _make_qkv(bs, n_heads, seq, d_head, device=device)

    naive = SelfAttentionNaive(n_heads, d_head).to(device).eval()
    sdpa  = SelfAttentionOptimised(n_heads, d_head).to(device).eval()

    with torch.no_grad():
        out_naive = naive(q, k, v, causal=causal)
        out_sdpa  = sdpa(q, k, v, causal=causal)

    max_err = (out_naive - out_sdpa).abs().max().item()
    print(f"[test_naive_vs_sdpa] max |naive - sdpa| = {max_err:.2e}  (atol={atol})")
    assert max_err < atol, f"FAILED: {max_err} > {atol}"
    print("[test_naive_vs_sdpa] PASSED")


def test_mhsa_forward(bs=2, n_heads=4, seq=64, embed_dim=256, causal=True):
    """Smoke-test the full MultiHeadSelfAttention module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(bs, seq, embed_dim, generator=g, device=device)

    for use_sdpa in [True, False]:
        mhsa = MultiHeadSelfAttention(embed_dim, n_heads, use_sdpa=use_sdpa).to(device).eval()
        with torch.no_grad():
            out = mhsa(x, causal=causal)
        assert out.shape == (bs, seq, embed_dim), f"Shape mismatch: {out.shape}"
        print(f"[test_mhsa_forward] use_sdpa={use_sdpa}  shape={out.shape}  PASSED")


def benchmark_attention(
    bs=2, n_heads=8, seq=512, d_head=64,
    warmup=20, iters=200, causal=True,
):
    """Wall-clock benchmark on CPU or CUDA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    q, k, v = _make_qkv(bs, n_heads, seq, d_head, dtype=dtype, device=device)

    modules = {
        "Naive":     SelfAttentionNaive(n_heads, d_head).to(device, dtype=dtype).eval(),
        "Optimised": SelfAttentionOptimised(n_heads, d_head).to(device, dtype=dtype).eval(),
    }

    print(f"\nBenchmark: bs={bs}, n_heads={n_heads}, seq={seq}, d_head={d_head}, "
          f"device={device}, dtype={dtype}")

    for name, mod in modules.items():
        # Warm-up
        with torch.no_grad():
            for _ in range(warmup):
                _ = mod(q, k, v, causal=causal)
            if device == "cuda":
                torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = mod(q, k, v, causal=causal)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / iters * 1000

        # Theoretical FLOPS: 2 * bs * n_heads * seq * seq * d_head (QK^T) +
        #                     2 * bs * n_heads * seq * seq * d_head (AV)
        flops = 4 * bs * n_heads * seq * seq * d_head
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        print(f"  {name:12s}: {elapsed_ms:.3f} ms/iter   ~{tflops:.2f} TFLOPS")


def compare_with_pytorch_mha(bs=2, seq=64, embed_dim=256, n_heads=4):
    """
    Compare our MultiHeadSelfAttention output against torch.nn.MultiheadAttention
    (causal-masked, batch_first=True).
    """
    device = "cpu"
    g = torch.Generator().manual_seed(7)
    x = torch.randn(bs, seq, embed_dim, generator=g)

    # Our module
    our = MultiHeadSelfAttention(embed_dim, n_heads, use_sdpa=False, bias=False).eval()

    # PyTorch reference (batch_first)
    ref = nn.MultiheadAttention(
        embed_dim, n_heads, bias=False, batch_first=True
    ).eval()

    # Copy weights so both use the same parameters
    # PyTorch MHA uses in_proj_weight of shape (3*embed, embed)
    with torch.no_grad():
        ref.in_proj_weight.copy_(our.qkv_proj.weight)
        ref.out_proj.weight.copy_(our.out_proj.weight)

    causal_mask = nn.Transformer.generate_square_subsequent_mask(seq)  # additive mask

    with torch.no_grad():
        our_out = our(x, causal=True)
        ref_out, _ = ref(x, x, x, attn_mask=causal_mask, need_weights=False)

    max_err = (our_out - ref_out).abs().max().item()
    print(f"[compare_with_pytorch_mha] max |ours - torch.nn.MHA| = {max_err:.2e}")
    # Small differences expected due to implementation details (e.g. fused vs split)
    return max_err


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Self-Attention PyTorch Tests")
    print("=" * 60)

    test_naive_vs_sdpa()
    test_mhsa_forward()
    benchmark_attention()
    compare_with_pytorch_mha()