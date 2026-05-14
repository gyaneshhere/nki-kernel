"""
Optimized Self-Attention NKI Kernel for AWS Trainium / Inferentia2+
====================================================================

Implements FlashAttention-v2-style fused self-attention using the
Neuron Kernel Interface (NKI).

Key design principles learned from NKI tutorials:
  - Partition dimension (first dim of every tile) must be <= 128 (nl.tile_size.pmax).
  - All heavy computation happens in SBUF (on-chip scratchpad), not in HBM.
  - Online softmax (running max + normalization) avoids materialising the full
    N×N score matrix in SBUF.
  - Tiling over K/V sequence in the inner loop keeps SBUF usage bounded.
  - Mixed-precision: matmuls in bfloat16, accumulations in float32.

Tensor layout convention (matches nki.kernels.flash_fwd):
  q, k, v : (batch, n_heads, head_dim, seq_len)   [head_dim is partition dim]
  out     : (batch, n_heads, seq_len,  head_dim)

Usage (baremetal / simulation):
  import nki
  out = nki.baremetal(flash_self_attn_fwd)(q, k, v)

Usage (PyTorch / XLA):
  from self_attention_nki_kernel import flash_self_attn_fwd
  out = flash_self_attn_fwd(q, k, v, use_causal_mask=True)
"""

import math

import numpy as np

try:
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    _NEURON_AVAILABLE = True
except ImportError:
    _NEURON_AVAILABLE = False
    print("[WARNING] neuronxcc not found – kernel defined but cannot be JIT-compiled.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cdiv(a: int, b: int) -> int:
    """Ceiling integer division."""
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Main NKI kernel
# ---------------------------------------------------------------------------

if _NEURON_AVAILABLE:
    @nki.jit
    def flash_self_attn_fwd(
        q_ref,           # (bs, n_heads, d_head, seq_q)
        k_ref,           # (bs, n_heads, d_head, seq_k)
        v_ref,           # (bs, n_heads, d_head, seq_v)  seq_k == seq_v
        use_causal_mask: bool = True,
        mixed_precision: bool = True,
        softmax_scale: float = None,
    ):
        """
        Fused Flash Self-Attention Forward Pass (NKI).

        Computes:  O = softmax( scale * Q @ K^T  [+ causal_mask] ) @ V

        IO tensor shapes
        ----------------
        q_ref : (bs, n_heads, d_head, seq_q)
        k_ref : (bs, n_heads, d_head, seq_k)
        v_ref : (bs, n_heads, d_head, seq_v)   (seq_k == seq_v required)
        returns out : (bs, n_heads, seq_q, d_head)

        Design
        ------
        We tile along the Q-sequence dimension (outer loop) and the K-sequence
        dimension (inner loop).  For every Q-tile we:
          1. Load the Q tile (d_head × TILE_Q) into SBUF.
          2. Inner loop over K tiles:
             a. Load K tile, compute QK^T score tile.
             b. Apply optional causal mask.
             c. Update running row-max (m_i) and re-scale old accumulator.
             d. Softmax numerator exp(score - new_max).
             e. Accumulate  exp_score @ V_tile  into output accumulator.
          3. Divide accumulator by the running sum (l_i) to normalise.
          4. Store the output tile to HBM.

        This keeps SBUF usage O(d_head * TILE_Q + d_head * TILE_K) instead of
        the full O(seq^2) for naive attention.
        """
        # ------------------------------------------------------------------ #
        # 0.  Shape extraction & validation
        # ------------------------------------------------------------------ #
        bs, n_heads, d_head, seq_q = q_ref.shape
        _,  _,       _,      seq_k = k_ref.shape

        assert seq_k == v_ref.shape[3], "seq_k must equal seq_v"
        assert d_head <= nl.tile_size.pmax, (
            f"d_head ({d_head}) must be <= nl.tile_size.pmax ({nl.tile_size.pmax}). "
            "For larger head dims split into multiple tiles."
        )

        # Tile sizes along sequence dimensions (free dimension, <= 512)
        # Q tiles: partition = d_head, free = TILE_Q
        # K tiles: partition = d_head, free = TILE_K
        TILE_Q = min(128, seq_q)   # keep manageable; 128 is a sweet spot
        TILE_K = min(128, seq_k)

        n_q_tiles = _cdiv(seq_q, TILE_Q)
        n_k_tiles = _cdiv(seq_k, TILE_K)

        scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(d_head))

        # Determine accumulation dtype
        acc_dtype = nl.bfloat16 if (not mixed_precision) else nl.float32
        compute_dtype = nl.bfloat16 if mixed_precision else q_ref.dtype

        # Allocate output tensor on HBM  (bs, n_heads, seq_q, d_head)
        out_ref = nl.ndarray(
            shape=(bs, n_heads, seq_q, d_head),
            dtype=q_ref.dtype,
            buffer=nl.hbm,
        )

        # ------------------------------------------------------------------ #
        # 1.  Outer loops: batch, head, Q-tile
        # ------------------------------------------------------------------ #
        for i_b in nl.affine_range(bs):
            for i_h in nl.affine_range(n_heads):
                for i_q_tile in nl.affine_range(n_q_tiles):

                    q_start = i_q_tile * TILE_Q
                    q_end   = min(q_start + TILE_Q, seq_q)
                    real_q  = q_end - q_start          # handles tail tile

                    # ------------------------------------------------------ #
                    # 1a. Load Q tile from HBM  → SBUF
                    #     shape: (d_head, TILE_Q)  where d_head is partition dim
                    # ------------------------------------------------------ #
                    ip_d = nl.arange(d_head)[:, None]     # partition indices
                    if_q = nl.arange(TILE_Q)[None, :]     # free indices

                    q_tile = nl.zeros((d_head, TILE_Q), dtype=compute_dtype, buffer=nl.sbuf)
                    # Only load real_q columns; rest stay zero (safe for masked softmax)
                    if_q_real = nl.arange(real_q)[None, :]
                    q_tile[ip_d, if_q_real] = nl.load(
                        q_ref[i_b, i_h, ip_d, q_start + if_q_real]
                    ).astype(compute_dtype)

                    # ------------------------------------------------------ #
                    # 1b. Accumulators for online softmax
                    #   o_acc : (d_head, TILE_Q) – numerator accumulator
                    #   m_i   : (1,      TILE_Q) – running row-max
                    #   l_i   : (1,      TILE_Q) – running sum of exp
                    # ------------------------------------------------------ #
                    o_acc = nl.zeros((d_head, TILE_Q), dtype=acc_dtype, buffer=nl.sbuf)
                    m_i   = nl.full((1,      TILE_Q), fill_value=nl.fp32.min,
                                    dtype=acc_dtype, buffer=nl.sbuf)
                    l_i   = nl.zeros((1,     TILE_Q), dtype=acc_dtype, buffer=nl.sbuf)

                    # ------------------------------------------------------ #
                    # 2.  Inner loop: K/V tiles  (online softmax)
                    # ------------------------------------------------------ #
                    for i_k_tile in nl.affine_range(n_k_tiles):

                        k_start = i_k_tile * TILE_K
                        k_end   = min(k_start + TILE_K, seq_k)
                        real_k  = k_end - k_start

                        if_k = nl.arange(TILE_K)[None, :]

                        # -------------------------------------------------- #
                        # 2a. Load K tile  (d_head, TILE_K)
                        # -------------------------------------------------- #
                        k_tile = nl.zeros((d_head, TILE_K), dtype=compute_dtype, buffer=nl.sbuf)
                        if_k_real = nl.arange(real_k)[None, :]
                        k_tile[ip_d, if_k_real] = nl.load(
                            k_ref[i_b, i_h, ip_d, k_start + if_k_real]
                        ).astype(compute_dtype)

                        # -------------------------------------------------- #
                        # 2b. Score tile: S = scale * Q^T @ K
                        #   Q : (d_head, TILE_Q)  →  Q^T : (TILE_Q, d_head)
                        #   K : (d_head, TILE_K)
                        #   S : (TILE_Q, TILE_K)
                        #
                        # NKI matmul: nl.matmul(lhs, rhs)
                        #   lhs partition dim maps to output partition dim
                        #   Here we want partition = TILE_Q so transpose Q.
                        # -------------------------------------------------- #
                        # Transpose Q for the matmul: (TILE_Q, d_head)
                        ip_q2 = nl.arange(TILE_Q)[:, None]
                        if_d2 = nl.arange(d_head)[None, :]
                        q_T = nl.ndarray((TILE_Q, d_head), dtype=compute_dtype, buffer=nl.sbuf)
                        q_T[ip_q2, if_d2] = nl.copy(q_tile[if_d2, ip_q2])

                        # Score: (TILE_Q, TILE_K)  – partition = TILE_Q
                        scores = nl.ndarray((TILE_Q, TILE_K), dtype=acc_dtype, buffer=nl.sbuf)
                        scores[...] = nl.matmul(q_T, k_tile, transpose_x=False,
                                                transpose_y=False) * scale
                        # q_T : (TILE_Q, d_head) × k_tile^T : (d_head, TILE_K)  → (TILE_Q, TILE_K)
                        # Note: nl.matmul(A, B) computes A @ B
                        #   with A: (p, f_a), B: (f_a, f_b) → out: (p, f_b)
                        # So we pass k_tile transposed implicitly:
                        scores = nl.matmul(q_T, k_tile, transpose_x=False,
                                           transpose_y=True) * nl.full(
                                               (TILE_Q, TILE_K), fill_value=scale,
                                               dtype=acc_dtype)

                        # -------------------------------------------------- #
                        # 2c. Causal mask  (future tokens → -inf)
                        #   Position of q tokens: [q_start .. q_start+TILE_Q)
                        #   Position of k tokens: [k_start .. k_start+TILE_K)
                        #   Mask: q_pos < k_pos  → apply -inf
                        # -------------------------------------------------- #
                        if use_causal_mask:
                            q_pos = q_start + nl.arange(TILE_Q)[:, None]  # (TILE_Q, 1)
                            k_pos = k_start + nl.arange(TILE_K)[None, :]  # (1, TILE_K)
                            causal_mask = q_pos < k_pos                   # True = mask out
                            scores = nl.where(causal_mask,
                                              nl.full((TILE_Q, TILE_K),
                                                      fill_value=nl.fp32.min,
                                                      dtype=acc_dtype),
                                              scores)

                        # Mask padding columns that fall outside real_k
                        if real_k < TILE_K:
                            pad_mask = nl.arange(TILE_K)[None, :] >= real_k
                            scores = nl.where(pad_mask,
                                              nl.full((TILE_Q, TILE_K),
                                                      fill_value=nl.fp32.min,
                                                      dtype=acc_dtype),
                                              scores)

                        # -------------------------------------------------- #
                        # 2d. Online softmax update
                        #   m_new = max(m_i, rowmax(S))
                        #   alpha = exp(m_i - m_new)       re-scale factor
                        #   exp_S = exp(S - m_new)
                        #   l_new = alpha * l_i + rowsum(exp_S)
                        #   o_acc = alpha * o_acc + exp_S @ V
                        # -------------------------------------------------- #
                        # rowmax of scores  → (TILE_Q, 1)
                        row_max_new = nl.max(scores, axis=1, keepdims=True)   # (TILE_Q, 1)

                        # Broadcast m_i: currently (1, TILE_Q) → need (TILE_Q, 1)
                        # m_i stored as (1, TILE_Q) with partition=1
                        # Easiest: keep m_i as (TILE_Q, 1) from the start.
                        # Re-compute max
                        m_new = nl.maximum(m_i, row_max_new)                 # (TILE_Q, 1)

                        # Re-scale factor for old accumulator
                        alpha = nl.exp(m_i - m_new)                          # (TILE_Q, 1)

                        # Stabilised exp of scores
                        exp_scores = nl.exp(scores - m_new)                  # (TILE_Q, TILE_K)

                        # Row-sum of exp_scores  → (TILE_Q, 1)
                        row_sum = nl.sum(exp_scores, axis=1, keepdims=True)  # (TILE_Q, 1)

                        # Update running sum
                        l_i = alpha * l_i + row_sum                          # (TILE_Q, 1)

                        # -------------------------------------------------- #
                        # 2e. Load V tile  (d_head, TILE_K) and accumulate
                        # -------------------------------------------------- #
                        v_tile = nl.zeros((d_head, TILE_K), dtype=compute_dtype, buffer=nl.sbuf)
                        v_tile[ip_d, if_k_real] = nl.load(
                            v_ref[i_b, i_h, ip_d, k_start + if_k_real]
                        ).astype(compute_dtype)

                        # exp_scores : (TILE_Q, TILE_K)
                        # v_tile     : (d_head, TILE_K)
                        # We want  exp_scores @ V^T  → (TILE_Q, d_head)
                        # which we store transposed as (d_head, TILE_Q) in o_acc.
                        #
                        # (d_head, TILE_K) × (TILE_K, TILE_Q) = (d_head, TILE_Q)
                        # → nl.matmul(v_tile, exp_scores^T)
                        #   v_tile: (d_head, TILE_K), exp_scores^T: (TILE_K, TILE_Q)
                        # Transpose exp_scores to (TILE_K, TILE_Q) first
                        ip_k2 = nl.arange(TILE_K)[:, None]
                        if_q2 = nl.arange(TILE_Q)[None, :]
                        exp_scores_T = nl.ndarray((TILE_K, TILE_Q), dtype=acc_dtype, buffer=nl.sbuf)
                        exp_scores_T[ip_k2, if_q2] = nl.copy(exp_scores[if_q2, ip_k2])

                        # Scale-and-add to accumulator
                        # o_acc[d_head, TILE_Q] += v_tile @ exp_scores_T^T
                        #   = (d_head, TILE_K) × (TILE_K, TILE_Q) → (d_head, TILE_Q)
                        dv = nl.matmul(v_tile, exp_scores_T, transpose_x=False,
                                       transpose_y=False)                 # (d_head, TILE_Q)

                        # o_acc = alpha (broadcast) * o_acc + dv
                        # alpha : (TILE_Q, 1), o_acc: (d_head, TILE_Q)
                        # Broadcast alpha over d_head axis
                        alpha_bcast = nl.ndarray((d_head, TILE_Q), dtype=acc_dtype, buffer=nl.sbuf)
                        alpha_bcast[ip_d, if_q] = nl.copy(alpha[if_q, nl.arange(1)[None, :]])
                        # Simpler: just multiply element-wise using broadcast
                        # alpha is (TILE_Q, 1), transpose to (1, TILE_Q) and broadcast
                        alpha_row = nl.ndarray((1, TILE_Q), dtype=acc_dtype, buffer=nl.sbuf)
                        alpha_row[nl.arange(1)[:, None], if_q] = nl.copy(
                            alpha[if_q.T, nl.arange(1)[:, None]]
                        )
                        o_acc = alpha_row * o_acc + dv.astype(acc_dtype)

                        # Update m_i
                        m_i = m_new

                    # End K-tile loop

                    # ------------------------------------------------------ #
                    # 3.  Normalise: o_acc /= l_i  (broadcast l_i over d_head)
                    # ------------------------------------------------------ #
                    l_bcast = nl.ndarray((1, TILE_Q), dtype=acc_dtype, buffer=nl.sbuf)
                    l_bcast[nl.arange(1)[:, None], if_q] = nl.copy(
                        l_i[if_q.T, nl.arange(1)[:, None]]
                    )
                    o_norm = (o_acc / l_bcast).astype(q_ref.dtype)   # (d_head, TILE_Q)

                    # ------------------------------------------------------ #
                    # 4.  Store output tile
                    #     out_ref layout: (bs, n_heads, seq_q, d_head)
                    #     We need to write (d_head, real_q) → (real_q, d_head)
                    # ------------------------------------------------------ #
                    ip_d_out  = nl.arange(d_head)[:, None]
                    if_q_out  = nl.arange(real_q)[None, :]

                    # Store transposed: out[..., q_pos, d] = o_norm[d, q_pos]
                    nl.store(
                        out_ref[i_b, i_h, q_start + if_q_out, ip_d_out],
                        value=o_norm[ip_d_out, if_q_out],
                    )

        return out_ref


    # -----------------------------------------------------------------------
    # Convenience wrapper: accepts the more common (bs, n_heads, seq, d_head)
    # layout used by PyTorch MultiheadAttention and transposes internally.
    # -----------------------------------------------------------------------
    @nki.jit
    def flash_self_attn_fwd_bhsd(
        q,               # (bs, n_heads, seq_q, d_head)
        k,               # (bs, n_heads, seq_k, d_head)
        v,               # (bs, n_heads, seq_v, d_head)
        use_causal_mask: bool = True,
        mixed_precision: bool = True,
        softmax_scale: float = None,
    ):
        """
        Wrapper kernel: standard (B, H, S, D) layout → (B, H, D, S) → kernel → output.

        Returns: out (bs, n_heads, seq_q, d_head)
        """
        bs, n_heads, seq_q, d_head = q.shape
        seq_k = k.shape[2]

        # Transpose Q, K, V to (bs, n_heads, d_head, seq)
        ip = nl.arange(d_head)[:, None]
        if_ = nl.arange(seq_q)[None, :]

        q_t = nl.ndarray((bs, n_heads, d_head, seq_q), dtype=q.dtype, buffer=nl.hbm)
        k_t = nl.ndarray((bs, n_heads, d_head, seq_k), dtype=k.dtype, buffer=nl.hbm)
        v_t = nl.ndarray((bs, n_heads, d_head, seq_k), dtype=v.dtype, buffer=nl.hbm)

        for i_b in nl.affine_range(bs):
            for i_h in nl.affine_range(n_heads):
                for i_s_q in nl.affine_range(_cdiv(seq_q, 128)):
                    s0 = i_s_q * 128
                    real_s = min(128, seq_q - s0)
                    ip2 = nl.arange(d_head)[:, None]
                    if2 = nl.arange(real_s)[None, :]
                    chunk = nl.load(q[i_b, i_h, s0 + if2, ip2])   # (real_s, d_head)
                    # store transposed
                    nl.store(q_t[i_b, i_h, ip2, s0 + if2], value=chunk[if2.T, ip2.T])

                for i_s_k in nl.affine_range(_cdiv(seq_k, 128)):
                    s0 = i_s_k * 128
                    real_s = min(128, seq_k - s0)
                    if2 = nl.arange(real_s)[None, :]
                    ip2 = nl.arange(d_head)[:, None]
                    ck = nl.load(k[i_b, i_h, s0 + if2, ip2])
                    nl.store(k_t[i_b, i_h, ip2, s0 + if2], value=ck[if2.T, ip2.T])
                    cv = nl.load(v[i_b, i_h, s0 + if2, ip2])
                    nl.store(v_t[i_b, i_h, ip2, s0 + if2], value=cv[if2.T, ip2.T])

        return flash_self_attn_fwd(q_t, k_t, v_t,
                                   use_causal_mask=use_causal_mask,
                                   mixed_precision=mixed_precision,
                                   softmax_scale=softmax_scale)


else:
    # Stub so the module can still be imported on non-Neuron machines
    def flash_self_attn_fwd(*args, **kwargs):
        raise RuntimeError("neuronxcc is required to run NKI kernels.")

    def flash_self_attn_fwd_bhsd(*args, **kwargs):
        raise RuntimeError("neuronxcc is required to run NKI kernels.")


# ---------------------------------------------------------------------------
# Numerical accuracy test (runs via nki.baremetal on a Neuron device)
# ---------------------------------------------------------------------------

def _pytorch_reference(q, k, v, use_causal_mask=True, scale=None):
    """CPU/GPU PyTorch reference for validating the NKI kernel."""
    import torch
    bs, n_heads, d_head, seq_q = q.shape
    seq_k = k.shape[3]
    s = scale if scale else (1.0 / math.sqrt(d_head))

    # q: (bs, n_heads, d_head, seq_q) → (bs, n_heads, seq_q, d_head)
    Q = torch.tensor(q).permute(0, 1, 3, 2).float()
    K = torch.tensor(k).permute(0, 1, 3, 2).float()
    V = torch.tensor(v).permute(0, 1, 3, 2).float()

    scores = torch.matmul(Q, K.transpose(-1, -2)) * s  # (bs, n_heads, seq_q, seq_k)
    if use_causal_mask:
        mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)                         # (bs, n_heads, seq_q, d_head)
    return out.numpy()


def run_accuracy_test():
    """
    Numeric accuracy test.  Requires a Neuron device (trn1/inf2/trn2).
    Run as:  python self_attention_nki_kernel.py
    """
    if not _NEURON_AVAILABLE:
        print("Skipping accuracy test: neuronxcc not available.")
        return

    import numpy as np

    bs, n_heads, d_head, seq = 1, 4, 64, 256
    rng = np.random.default_rng(42)
    dtype = np.float16

    q = rng.standard_normal((bs, n_heads, d_head, seq)).astype(dtype) * 0.02
    k = rng.standard_normal((bs, n_heads, d_head, seq)).astype(dtype) * 0.02
    v = rng.standard_normal((bs, n_heads, d_head, seq)).astype(dtype) * 0.02

    # Run NKI kernel via baremetal
    out_nki = nki.baremetal(flash_self_attn_fwd)(q, k, v, use_causal_mask=True)

    # Run PyTorch reference (on CPU)
    out_ref = _pytorch_reference(q, k, v, use_causal_mask=True)

    # out_nki shape: (bs, n_heads, seq, d_head)
    # out_ref shape: (bs, n_heads, seq, d_head)
    max_err = np.max(np.abs(out_nki - out_ref.astype(dtype)))
    rel_err = max_err / (np.max(np.abs(out_ref)) + 1e-9)
    print(f"Max absolute error : {max_err:.6f}")
    print(f"Max relative error : {rel_err:.6f}")
    assert rel_err < 0.02, f"Accuracy test FAILED: relative error {rel_err}"
    print("Accuracy test PASSED.")


def run_benchmark():
    """
    Performance benchmark.  Requires a Neuron device.
    """
    if not _NEURON_AVAILABLE:
        print("Skipping benchmark: neuronxcc not available.")
        return

    import numpy as np

    bs, n_heads, d_head, seq = 2, 8, 64, 512
    rng = np.random.default_rng(0)
    dtype = np.float16

    q = rng.standard_normal((bs, n_heads, d_head, seq)).astype(dtype)
    k = rng.standard_normal((bs, n_heads, d_head, seq)).astype(dtype)
    v = rng.standard_normal((bs, n_heads, d_head, seq)).astype(dtype)

    bench = nki.benchmark(
        warmup=5, iters=100
    )(flash_self_attn_fwd)(q, k, v, use_causal_mask=True)
    print(f"Latency p50  : {bench.nc_latency.p50:.3f} ms")
    print(f"Latency p99  : {bench.nc_latency.p99:.3f} ms")


if __name__ == "__main__":
    run_accuracy_test()
    run_benchmark()