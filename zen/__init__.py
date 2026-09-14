"""zen entry point.

The prompt assembly in `Qwen3TTSForConditionalGeneration.generate` is intricate
(role tokens, codec control ids, language/dialect selection, speaker or ICL
prompt, trailing-text alignment). Reimplementing it risks subtle drift, so zen
does not: it *hijacks the talker's generate call* and leaves everything else —
prompt building, codec encode/decode, batching — to the stock library.

    model.generate_voice_design(...)         # stock, unchanged
        -> Qwen3TTSForConditionalGeneration.generate(...)
             -> self.talker.generate(...)    # <-- zen replaces this
                  -> graph-compiled talker + code predictor loops

Because the call signature is stable (it receives `inputs_embeds`,
`trailing_text_hidden`, `tts_pad_embed` and the generation kwargs) this is a
small, auditable seam.

`zen` carries its own copy of the upstream `qwen_tts` package under
`zen/_vendor`, so it does not depend on any external checkout of that library.
This vendored copy is *unmodified upstream*: the hand-rolled `_subtalker_decode`
that the `Qwen3-TTS-opt` fork adds lives on the `talker.forward()` path, which
zen bypasses entirely (zen replaces `talker.generate`), so it is dead weight here.
"""

from __future__ import annotations

import os
import sys

import torch

from .pipeline import ZenTTS, _make_static_cache

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")


def _use_vendored_qwen_tts():
    """Make `import qwen_tts` resolve to the copy shipped inside zen.

    Prepended to `sys.path` so it wins over any installed wheel. Idempotent.
    """
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)


def load_model(model_path, device_map="cuda:0", dtype=torch.bfloat16, **kwargs):
    """Load a checkpoint with zen's vendored `qwen_tts`.

    Equivalent to `Qwen3TTSModel.from_pretrained`, without requiring the caller
    to install or import `qwen_tts` itself.
    """
    _use_vendored_qwen_tts()
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(
        model_path, device_map=device_map, dtype=dtype, **kwargs)


def install(model, **kwargs) -> ZenTTS:
    """Monkeypatch `model.model.talker.generate` to use zen's graph loops.

    Returns the `ZenTTS` instance. Call `uninstall` to restore the stock path.
    """
    raw = model.model
    zen = ZenTTS(model, **kwargs)
    if not hasattr(raw, "_zen_stock_generate"):
        raw._zen_stock_generate = raw.talker.generate
    raw.talker.generate = _make_talker_generate(raw, zen)
    return zen


def uninstall(model):
    raw = model.model
    if hasattr(raw, "_zen_stock_generate"):
        raw.talker.generate = raw._zen_stock_generate
        del raw._zen_stock_generate


def _make_talker_generate(raw, zen: ZenTTS):
    """Build a drop-in for `talker.generate` returning the same object.

    The stock caller only consumes `talker_result.hidden_states`, which it reads
    as `(hidden_states_tuple, codec_ids_tensor)` pairs per step. We reproduce
    that contract exactly.
    """
    @torch.no_grad()
    def talker_generate(inputs_embeds=None, attention_mask=None, trailing_text_hidden=None,
                        tts_pad_embed=None, eos_token_id=None, max_new_tokens=4096, **kw):
        return _zen_run(raw, zen, inputs_embeds, trailing_text_hidden, tts_pad_embed,
                        max_new_tokens)

    return talker_generate


class _ZenResult:
    """Minimal stand-in exposing just what the stock caller reads."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


@torch.no_grad()
def _zen_run(raw, zen: ZenTTS, inputs_embeds, trailing_text_hidden, tts_pad_embed, max_new_tokens):
    talker = raw.talker
    cfg = talker.config
    dev = inputs_embeds.device
    S = inputs_embeds.shape[1]
    codec_emb = talker.get_input_embeddings()
    cp_embs = talker.code_predictor.get_input_embeddings()
    eos_id = cfg.codec_eos_token_id

    # `zen.max_frames` is the length of the RoPE table / flat KV cache / mask
    # buffer, i.e. the total number of positions available to prompt + generated
    # frames. The decode loop starts at position `S`, so the number of frames it
    # may still emit is `max_frames - S`. Using `max_frames` directly here made
    # `pos = S + frame` exceed the table for any non-empty prompt, which trips
    # `indexSelectSmallIndex: Assertion srcIndex < srcSelectDimSize` and kills
    # the worker process with a device-side assert.
    room = zen.max_frames - S
    if room <= 0:
        raise ValueError(
            f"prompt of {S} positions leaves no room in a {zen.max_frames}-position "
            f"talker window; raise ZenTTS(max_frames=...)"
        )
    limit = min(max_new_tokens, room)

    # ---- prefill (stock, once) ----
    cache = _make_static_cache(talker, zen.max_frames + 8)
    cpos = torch.arange(0, S, device=dev)
    pids = cpos.view(1, 1, -1).expand(3, 1, -1)
    out = talker.model(inputs_embeds=inputs_embeds, past_key_values=cache, cache_position=cpos,
                       position_ids=pids, use_cache=True, attention_mask=None)
    hidden = out.last_hidden_state[:, -1:, :]
    # HF seeds `input_ids` with a placeholder of the prompt length when called
    # with `inputs_embeds`, and the repetition penalty counts the whole history.
    # So the penalty must also see those S zeros, otherwise token 0 is penalised
    # differently from the stock run.
    hist0 = torch.zeros(1, S, dtype=torch.long, device=dev)
    code0 = zen._sample(talker.codec_head(hidden)[:, -1, :], hist0)
    seen = [hist0, code0]

    zen.talker_step.reset()
    zen.reset_code_cache()
    zen._absorb_prefill_cache(cache, S)
    # The prefill KV copy writes into buffers that the captured talker graph owns.
    # Without a barrier the first replay can observe a partially-written cache,
    # which showed up as an intermittent divergence in the first frames of a
    # fresh process. One sync per request is negligible next to 800+ graph replays.
    torch.cuda.synchronize()

    # EOS detection must not stall every frame: `int(code0.item())` per frame
    # measured 86.8% of request time in pure synchronisation. But the previous
    # "non-blocking copy into a pinned flag" attempt was unsound — the DMA has no
    # completion guarantee, so a request could observe a *stale* flag (including
    # one still in flight from the previous request) and abort after a single
    # frame, which showed up as intermittent 1-second outputs.
    #
    # Instead, collect the emitted code0 ids into a fixed GPU buffer and check the
    # whole window for EOS once every EOS_CHECK_EVERY frames. That is one real
    # synchronisation per window instead of per frame, and it is exact: when EOS
    # is found the generator is truncated to the frame that produced it.
    EOS_CHECK_EVERY = 16
    code_buf = torch.zeros(limit, dtype=torch.long, device=dev)
    stop_frame = None
    done = False

    steps = []
    zstep = zen.talker_step
    pos_buf = zen._pos_buf
    emb_buf = zen._emb_buf

    for frame in range(limit):
        # `hidden` produced by this step is what the code predictor and the head
        # consume; clone so the graph's output buffer can be reused safely.
        step_hidden = hidden.clone()

        last_id_hidden = codec_emb(code0)
        torch.compiler.cudagraph_mark_step_begin()
        cp = zen.code_pred
        inner = getattr(cp, "_orig_mod", cp)
        inner.load_inputs(hidden, last_id_hidden)
        extra = cp(inner._in_a, inner._in_b).clone()

        # Fuse the 16 code embeddings + trailing text into the preallocated
        # buffer: the graph replays against this address, so it must not move.
        emb_buf.copy_(last_id_hidden)
        for i in range(cfg.num_code_groups - 1):
            emb_buf.add_(cp_embs[i](extra[:, i:i + 1]))
        if frame < trailing_text_hidden.shape[1]:
            emb_buf.add_(trailing_text_hidden[:, frame].unsqueeze(1))
        else:
            emb_buf.add_(tts_pad_embed)

        codec_ids = torch.cat((code0, extra), dim=-1)
        # the stock caller does `hid[-1]` -> codec ids and `hid[0][-1][:, -1:]` ->
        # last-layer hidden, so each element must be ((hidden_tuple,), codec_ids).
        # Only the last layer's hidden is read, so a 1-tuple is enough.
        steps.append(((step_hidden,), codec_ids))
        code_buf[frame] = code0.reshape(())

        if done:
            break
        if (frame + 1) % EOS_CHECK_EVERY == 0:
            # one synchronisation per window; exact, no stale reads possible
            window = code_buf[: frame + 1]
            hit = (window == eos_id).nonzero()
            if hit.numel():
                stop_frame = int(hit[0])
                done = True

        pos = S + frame
        pos_buf.fill_(pos)
        torch.compiler.cudagraph_mark_step_begin()
        hidden = zstep(emb_buf, pos_buf, zstep._mask(pos)).clone()

        code0 = zen._sample(talker.codec_head(hidden)[:, -1, :], torch.cat(seen, dim=-1))
        seen.append(code0)

    if stop_frame is None:
        hit = (code_buf[: len(steps)] == eos_id).nonzero()
        stop_frame = int(hit[0]) if hit.numel() else None
    if stop_frame is not None:
        # keep the EOS frame itself, matching the stock truncation
        steps = steps[: stop_frame + 1]

    return _ZenResult(steps)
