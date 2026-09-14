"""Graph-compiled code-predictor decode.

After each talker step the code predictor emits the remaining 15 of the 16 code
groups for that frame. It is a 5-layer transformer run autoregressively 15
times, hidden size 1024, sequence length 1 — the worst possible shape for a GPU:
~75 tiny layer forwards per frame, all launch-bound. Measured on the stock code
this is ~65% of total request time.

`torch.compile(mode="reduce-overhead", fullgraph=True)` captures the entire
15-sub-step loop into one CUDA graph. Shapes are static: the prefill always
feeds 2 tokens and every step feeds exactly 1 token at a known cache index.

Sampling stays faithful: each sub-step applies exactly the same top-k/top-p/
temperature logits processing and draws from `torch.multinomial`, so with a
seeded RNG the emitted code sequence matches the stock `_subtalker_decode`.

Note: unlike the talker, the cache here is a `StaticCache` because that is what
`code_predictor.forward` expects; it is index-addressed by `cache_position`, so
it stays capturable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers.cache_utils import StaticCache



class CodePredictorDecode(nn.Module):
    """Full 15-sub-step code predictor decode for one talker frame.

    Inputs are the two embeddings the stock code feeds:
      * `past_hidden`   (B, 1, H_talker) — the talker's last hidden state
      * `last_id_hidden`(B, 1, H_talker) — embedding of the talker's own code
    Output is the *sequence of sampled codes* (B, num_code_groups - 1), exactly
    like `_subtalker_decode`, so downstream code can build the fused embedding.
    """

    def __init__(self, code_predictor: nn.Module, num_code_groups: int, talker_hidden_size: int,
                 do_sample=True, top_k=50, top_p=1.0, temperature=0.9, max_len: int | None = None):
        super().__init__()
        self.cp = code_predictor
        self.num_code_groups = num_code_groups
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        dev = next(code_predictor.parameters()).device
        dt = next(code_predictor.parameters()).dtype
        self._cache = StaticCache(
            config=code_predictor.config, max_cache_len=max_len or num_code_groups,
            device=dev, dtype=dt,
        )
        self._cpos0 = torch.arange(0, 2, device=dev)
        self._cpos1 = torch.arange(2, 2 + num_code_groups - 2, device=dev)
        # cudagraph_trees binds a captured graph to the addresses present at
        # capture time, so the two inputs are copied into these fixed buffers
        # instead of being fed directly. Both are the talker hidden size;
        # `small_to_mtp_projection` narrows them to the predictor's own width.
        self._in_a = torch.zeros(1, 1, talker_hidden_size, dtype=dt, device=dev)
        self._in_b = torch.zeros(1, 1, talker_hidden_size, dtype=dt, device=dev)

    def load_inputs(self, past_hidden, last_id_hidden):
        """Copy the caller's tensors into the static input buffers.

        Must run *outside* the compiled call: dynamo guards the strides of the
        arguments, and the real caller passes a non-contiguous view
        (`last_hidden_state[:, -1:, :]`) whose stride differs from the buffers
        captured at warmup, which would force a recompile.
        """
        self._in_a.copy_(past_hidden)
        self._in_b.copy_(last_id_hidden)

    def forward(self, past_hidden, last_id_hidden):
        ng = self.num_code_groups
        cache = self._cache
        cp = self.cp

        out = cp(input_ids=None, inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                 past_key_values=cache, cache_position=self._cpos0,
                 use_cache=True, generation_steps=0, attention_mask=None)
        prev = self._sample(out.logits[:, -1, :])
        codes = [prev]

        for i in range(ng - 2):
            out = cp(input_ids=prev, inputs_embeds=None, past_key_values=cache,
                     cache_position=self._cpos1[i:i + 1],
                     use_cache=True, generation_steps=1 + i, attention_mask=None)
            prev = self._sample(out.logits[:, -1, :])
            codes.append(prev)

        return torch.cat(codes, dim=-1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.to(torch.float32)
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
