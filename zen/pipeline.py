"""zen inference pipeline: prefill with the stock talker, decode with graphs.

The stock `Qwen3TTSForConditionalGeneration.generate` builds the prompt, runs a
`GenerationMixin.generate` over the talker (28 layers x one token per audio
frame), and inside *every* talker step runs a nested 15-sub-step code predictor
generate. Both loops are launch-bound.

zen keeps the prompt construction and the codec untouched — those are already
fine — and replaces the two hot loops with CUDA-graph-compiled equivalents:

    prefill (stock, once)  ->  per frame:
        zen.talker.TalkerStep       (28 layers, 1 graph replay)
        zen.code_predictor.CodePredictorDecode (15 sub-steps, 1 graph replay)
    ->  codec decode (stock)

Weights are shared with the loaded model (no copy, no extra VRAM).
"""

from __future__ import annotations

import os
import time

import torch

from .code_predictor import CodePredictorDecode
from .talker import TalkerStep


class ZenTTS:
    """Wraps a loaded `Qwen3TTSModel` and serves the talker loop through graphs."""

    def __init__(self, model, max_frames: int = 4096, compile: bool = True,
                 do_sample: bool = True, top_k: int = 50, top_p: float = 1.0,
                 temperature: float = 0.9,
                 subtalker_do_sample: bool = True, subtalker_top_k: int = 50,
                 subtalker_top_p: float = 1.0, subtalker_temperature: float = 0.9,
                 repetition_penalty: float = 1.05):
        self.model = model
        raw = model.model
        self.raw = raw
        self.talker = raw.talker
        cfg = self.talker.config
        self.cfg = cfg
        self.eos_id = cfg.codec_eos_token_id
        self.max_frames = max_frames

        self.talker_step = TalkerStep(self.talker, max_len=max_frames)
        self.code_pred = CodePredictorDecode(
            self.talker.code_predictor, cfg.num_code_groups, cfg.hidden_size,
            do_sample=subtalker_do_sample, top_k=subtalker_top_k,
            top_p=subtalker_top_p, temperature=subtalker_temperature,
            max_len=cfg.num_code_groups,
        )
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        # the stock talker blocks the top 1024 ids except EOS
        self.suppress_from = cfg.vocab_size - 1024

        # cudagraph_trees replays against the addresses captured at warmup, so
        # every graph input is a preallocated buffer updated in place. Fresh
        # per-step tensors silently push execution onto the slow path (measured
        # 96.8 ms vs 41.4 ms for the talker step) and can re-trigger capture.
        p0 = next(self.talker.parameters())
        self._dev, self._dt = p0.device, p0.dtype
        self._hidden_buf = torch.zeros(1, 1, cfg.hidden_size, dtype=self._dt, device=self._dev)
        self._pos_buf = torch.zeros(1, dtype=torch.long, device=self._dev)
        self._emb_buf = torch.zeros(1, 1, cfg.hidden_size, dtype=self._dt, device=self._dev)

        if compile:
            # Initialize the code predictor's StaticCache *eagerly* first:
            # `StaticLayer.lazy_initialization` allocates the backing tensors and
            # calls `torch._dynamo.mark_static_address`. If that happens inside
            # the captured graph the cache is treated as graph output and every
            # later reset dies with "output ... has been overwritten".
            ph = self._hidden_buf
            with torch.no_grad():
                self.reset_code_cache()
                self.code_pred(ph, ph)
                self.talker_step.reset()
                self.talker_step(ph, self._pos_buf, self.talker_step._mask(0))
                torch.cuda.synchronize()

                self.talker_step = torch.compile(self.talker_step, mode="reduce-overhead", fullgraph=True)
                self.code_pred = torch.compile(self.code_pred, mode="reduce-overhead", fullgraph=True)

            # Capture both graphs so the first real request does not pay for it.
            self.warmup()

    def reset_code_cache(self):
        """Zero the code-predictor StaticCache from OUTSIDE any captured graph.

        The capture treats the cache as graph state; touching it (the reset's
        `zero_`) from inside a later capture is what triggers
        'output ... has been overwritten'. Each decode overwrites all 15 slots,
        so this only needs to run between requests."""
        mod = getattr(self.code_pred, "_orig_mod", self.code_pred)
        mod._cache.reset()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def warmup(self, dtype=None, device=None):
        """Capture both graphs.

        Uses the same preallocated buffers the decode loop will use, because
        cudagraph_trees binds a graph to the addresses present at capture time.
        `torch.no_grad()` matters: with grad enabled the outputs are retained for
        a backward that never comes and cudagraph_trees refuses its fast path.
        """
        # Capture with the exact buffers the decode loop will feed, otherwise
        # the addresses differ and cudagraph_trees recaptures on first request.
        for _ in range(3):
            self.talker_step.reset()
            self.talker_step(self._emb_buf, self._pos_buf, self.talker_step._mask(0))
        self.reset_code_cache()
        cp = self.code_pred
        inner = getattr(cp, "_orig_mod", cp)
        inner.load_inputs(self._hidden_buf, self._hidden_buf)
        torch.compiler.cudagraph_mark_step_begin()
        cp(inner._in_a, inner._in_b)
        torch.cuda.synchronize()


    def _absorb_prefill_cache(self, cache, n: int):
        """Copy the prefill KV blocks into zen's flat caches.

        `n` is passed in rather than read from `cache.get_seq_length()`, which is
        `(keys[0,0].any(dim=-1)).sum()` — a data-dependent guess that under-counts
        whenever a position's key vector is all zeros, silently leaving the tail
        of the cache uninitialised. The caller knows the prompt length exactly.

        The caller must synchronise afterwards: these buffers belong to the
        captured talker graph, and a replay issued before the copy lands can read
        a partially-written cache.
        """
        tstep = self.talker_step
        for i in range(tstep.num_layers):
            k, v = cache[i]
            tstep.k_caches[i][:, :, :n, :].copy_(k[:, :, :n, :])
            tstep.v_caches[i][:, :, :n, :].copy_(v[:, :, :n, :])

    def _sample(self, logits: torch.Tensor, seen: torch.Tensor | None = None) -> torch.Tensor:
        """Talker sampler matching the stock call.

        Stock passes `repetition_penalty=1.05` and `suppress_tokens` blocking the
        top 1024 ids (EOS excepted). HF's warper order is Temperature -> TopK ->
        TopP, applied *after* the repetition penalty.
        """
        logits = logits.to(torch.float32).clone()
        saved_eos = logits[:, self.eos_id].clone()
        logits[:, self.suppress_from:] = float("-inf")
        logits[:, self.eos_id] = saved_eos  # EOS is excluded from the suppression list

        if seen is not None and seen.numel() > 0 and self.repetition_penalty != 1.0:
            v = torch.gather(logits, -1, seen)
            v = torch.where(v > 0, v / self.repetition_penalty, v * self.repetition_penalty)
            logits.scatter_(-1, seen, v)

        if not self.do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        if self.temperature and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_k and self.top_k > 0:
            k = min(self.top_k, logits.shape[-1])
            thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < thresh, float("-inf"), logits)
        if self.top_p and self.top_p < 1.0:
            srt, idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(srt, dim=-1)
            cum = probs.cumsum(dim=-1) - probs
            srt = torch.where(cum > self.top_p, float("-inf"), srt)
            logits = torch.empty_like(logits).scatter_(-1, idx, srt)
        return torch.multinomial(torch.softmax(logits, dim=-1), 1)


def _make_static_cache(talker, max_len):
    from transformers.cache_utils import StaticCache
    p = next(talker.parameters())
    return StaticCache(config=talker.config, max_cache_len=max_len, device=p.device, dtype=p.dtype)
