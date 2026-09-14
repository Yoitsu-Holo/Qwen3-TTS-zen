"""Rotary embedding for zen.

The talker uses `rope_scaling = {"interleaved": True, "mrope_section": [24,20,20]}`.
For TTS there is no image modality: all three mrope position streams are the
same index (`position_ids = cache_position.expand(3, B, -1)`), and the
interleaved section shuffle permutes between three *identical* streams, so it is
a no-op. The whole thing therefore reduces to a standard 1-D RoPE, which we
precompute once into a static table.

Verified against `Qwen3TTSTalkerRotaryEmbedding`: identical cos/sin (see
`tests/test_rope.py`).
"""

from __future__ import annotations

import torch


class RopeTable:
    """Precomputed cos/sin lookup tables of shape (max_len, head_dim).

    The tables live on a fixed device and are indexed by a position *tensor*
    (`index_select`), so the lookup stays inside a CUDA graph and dynamo can
    guard it like any other tensor op.
    """

    def __init__(self, head_dim: int, theta: float, max_len: int, device, dtype: torch.dtype):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        pos = torch.arange(max_len, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)  # (max_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_len, head_dim)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def __call__(self, pos: torch.Tensor):
        """pos: (S,) long tensor -> cos/sin of shape (1, 1, S, head_dim)."""
        cos = self.cos.index_select(0, pos).view(1, 1, -1, self.cos.shape[-1])
        sin = self.sin.index_select(0, pos).view(1, 1, -1, self.sin.shape[-1])
        return cos, sin
