"""Graph-compiled talker decode step.

The stock talker decode is 28 transformer layers run one token at a time. On an
A3000 each layer costs ~1.1 ms while its arithmetic takes ~1.3 us, i.e. the
step is ~99.9% kernel-launch / host-dispatch bound. `torch.compile(mode=
"reduce-overhead")` captures the whole step into a CUDA graph, collapsing
hundreds of launches into one replay.

Design constraints that make the capture work (the stock module trips on all of
them):
  * Static shapes. Batch is 1 and the sequence is always 1 token; the KV cache
    is a preallocated (1, n_kv, max_len, head_dim) buffer.
  * No data-dependent slicing. `cache[:, :, pos:pos+1]` makes dynamo try to
    guard on `pos` and fail, so the cache is written with `index_copy_` and the
    attention always covers the full cache, masked by a preallocated additive
    mask built *outside* the graph.
  * No host round-trips. RoPE tables and the mask are device tensors; the only
    inputs that change per step are `hidden`, `pos` and `mask`, all at stable
    addresses (cudagraph_trees requirement).

Weights are referenced from the already-loaded `Qwen3TTSTalkerModel`, so zen
costs no extra VRAM and inherits any fine-tuning automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RopeTable


class _Norm(nn.Module):
    """RMSNorm that keeps the accumulate-in-fp32 behaviour of the original."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.data.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return self.weight * (xf * torch.rsqrt(var + self.eps)).to(dt)


class _MLP(nn.Module):
    def __init__(self, gate: nn.Linear, up: nn.Linear, down: nn.Linear):
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _Attention(nn.Module):
    def __init__(self, attn: nn.Module, head_dim: int, n_heads: int, n_kv: int, eps: float):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = _Norm(attn.q_norm.weight, eps)
        self.k_norm = _Norm(attn.k_norm.weight, eps)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv = n_kv

    def forward(self, x, cos, sin, k_cache, v_cache, pos, mask):
        b, s = x.shape[0], x.shape[1]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv
        q = self.q_norm(self.q_proj(x).view(b, s, nh, hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, nkv, hd)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, nkv, hd).transpose(1, 2)

        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)

        idx = torch.arange(s, device=x.device) + pos
        k_cache.index_copy_(2, idx, k)
        v_cache.index_copy_(2, idx, v)

        if nkv != nh:
            kk = k_cache.repeat_interleave(nh // nkv, dim=1)
            vv = v_cache.repeat_interleave(nh // nkv, dim=1)
        else:
            kk, vv = k_cache, v_cache
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, nh * hd))


def _rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard rotate_half RoPE (the interleaved mrope reduces to this)."""
    half = t.shape[-1] // 2
    x1, x2 = t[..., :half], t[..., half:]
    return torch.cat((-x2, x1), dim=-1) * sin + t * cos


class _Layer(nn.Module):
    def __init__(self, layer: nn.Module, head_dim: int, n_heads: int, n_kv: int, eps: float):
        super().__init__()
        self.self_attn = _Attention(layer.self_attn, head_dim, n_heads, n_kv, eps)
        self.mlp = _MLP(layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
        self.input_layernorm = _Norm(layer.input_layernorm.weight, eps)
        self.post_attention_layernorm = _Norm(layer.post_attention_layernorm.weight, eps)

    def forward(self, x, cos, sin, kc, vc, pos, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kc, vc, pos, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class TalkerStep(nn.Module):
    """One decode step of the talker: (hidden, pos, mask) -> hidden.

    `hidden` is the embedding of the token about to be fed in (zen does the
    embedding / code-predictor fusion outside so this stays a pure transformer
    block). The returned hidden is the *last layer's* hidden state, which is what
    the talker head and the code predictor consume.
    """

    def __init__(self, talker: nn.Module, max_len: int = 4096, dtype=torch.bfloat16):
        super().__init__()
        cfg = talker.config
        self.cfg = cfg
        self.max_len = max_len
        self.num_layers = cfg.num_hidden_layers
        eps = cfg.rms_norm_eps
        self.layers = nn.ModuleList(
            [_Layer(talker.model.layers[i], cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, eps)
             for i in range(self.num_layers)]
        )
        self.norm = _Norm(talker.model.norm.weight, eps)
        self.rope = RopeTable(cfg.head_dim, cfg.rope_theta, max_len, _param_device(talker), dtype)

        dev = _param_device(talker)
        self.k_caches = [
            torch.zeros(1, cfg.num_key_value_heads, max_len, cfg.head_dim, dtype=dtype, device=dev)
            for _ in range(self.num_layers)
        ]
        self.v_caches = [
            torch.zeros(1, cfg.num_key_value_heads, max_len, cfg.head_dim, dtype=dtype, device=dev)
            for _ in range(self.num_layers)
        ]
        # cudagraph_trees replays against fixed addresses, so every graph input
        # must be a preallocated buffer that is updated in place. `_mask` used to
        # build a fresh bf16 tensor each step, which forced a recapture and an
        # ~2.3x slowdown (96.8 ms vs 41.4 ms measured).
        self._mask_buf = torch.zeros(1, 1, 1, max_len, dtype=dtype, device=dev)
        self._neg = float("-inf")

    # ---- cache / mask management (host side, outside any graph) ----
    def reset(self):
        for kc in self.k_caches:
            kc.zero_()
        for vc in self.v_caches:
            vc.zero_()

    def _mask(self, pos: int) -> torch.Tensor:
        """Update the mask buffer in place and return it (stable address)."""
        # Fail loudly on the host instead of tripping a device-side assert in
        # index_select, which would take down the whole worker process.
        if not 0 <= pos < self.max_len:
            raise IndexError(
                f"position {pos} outside talker window [0, {self.max_len}); "
                f"max_frames must cover prompt + generated frames"
            )
        m = self._mask_buf
        m[..., : pos + 1] = 0.0
        m[..., pos + 1:] = self._neg
        return m

    def forward(self, x, pos, mask):
        cos, sin = self.rope(pos.reshape(-1))
        for i in range(self.num_layers):
            x = self.layers[i](x, cos, sin, self.k_caches[i], self.v_caches[i], pos, mask)
        return self.norm(x)


def _param_device(m: nn.Module) -> torch.device:
    return next(m.parameters()).device
