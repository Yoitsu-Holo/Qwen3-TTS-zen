# Qwen-TTS-zen

A drop-in acceleration layer for Qwen3-TTS inference. It does not retrain, quantize
or approximate the model — it removes the CPU-side launch bottleneck that keeps the
stock implementation from using the GPU it runs on.

zen is self-contained: it ships its own copy of the upstream `qwen_tts` library
under `zen/_vendor`, so no separate checkout or installed wheel of that package is
needed. Loading goes through `zen.load_model(...)`.

> The vendored copy is **unmodified upstream**. The `Qwen3-TTS-opt` fork's only
> functional change was a hand-rolled `_subtalker_decode` on the
> `talker.forward()` path — a path zen bypasses entirely, because zen replaces
> `talker.generate` itself. That optimization is therefore dead weight here, and
> its replacement (`zen/code_predictor.py`) is faster anyway (15.2 ms vs 84.1 ms
> for the 15 sub-steps).

## Why the stock path is slow

Profiling a real request on an RTX A3000 (sms_86) shows the step is almost entirely
host-dispatch bound, not compute bound:

| stage | share of wall time |
|---|---|
| code predictor (15 AR sub-steps x 5 layers, per frame) | **65%** |
| talker (28 layers, one token per frame) | 27% |
| codec decode + prompt handling | <8% |

A single talker layer does `2048x6144` GEMMs on a 1-token sequence: ~1.3 us of
arithmetic. It measures ~1.1 ms. That is a ~800x gap, i.e. ~99.9% launch overhead.
Every frame issues ~1000 CUDA launches; a 4-second clip issues ~65 000.

## What zen does

Two changes, both fully weight-preserving:

1. **The talker decode step is rewritten** as a statically-shaped module
   (`zen/talker.py`) with its own RoPE tables, its own flat KV cache written by
   `index_copy_`, and its own preallocated additive mask. It then compiles under
   `torch.compile(mode="reduce-overhead", fullgraph=True)`, which captures the whole
   28-layer step into one CUDA graph — one replay instead of hundreds of launches.

2. **The code predictor's 15 sub-steps are captured into a single graph**
   (`zen/code_predictor.py`) the same way.

Prompt construction, the codec, tokenization and batching are untouched: zen
hijacks `talker.generate` and leaves everything else to the stock library
(`zen/__init__.py`).

### What made capture possible

The stock modules defeat CUDA-graph capture in four ways; zen avoids each:

| stock behaviour | why it breaks capture | zen |
|---|---|---|
| `cache[:, :, pos:pos+1] = k` | data-dependent slice, dynamo cannot guard `pos` | `index_copy_` with a position tensor |
| `create_causal_mask(...)` each step | builds a new mask inside the graph | one preallocated mask buffer, shaped outside |
| `@dynamic_rope_update` + per-step `cos`/`sin` | Python branch, host scalar work | precomputed table + `index_select` |
| `StaticCache` lazy init inside the graph | cache registers as graph *output*, so its later reset fails with "output has been overwritten" | cache initialised eagerly before compiling |

Three further requirements are easy to miss:

1. **`torch.no_grad()` around every graph call.** With grad enabled the outputs
   stay alive for a backward that never happens, and cudagraph_trees refuses its
   fast path.
2. **Every graph input must be a preallocated buffer updated in place.** Feeding a
   fresh tensor per step changes its address; measured 41.4 ms (stable buffer) vs
   96.8 ms (new tensor) for the talker step.
3. **Input *strides* are guarded too.** The real caller passes a non-contiguous
   view (`last_hidden_state[:, -1:, :]`); passing it straight to the compiled code
   predictor recompiles on the first request. `CodePredictorDecode.load_inputs`
   copies into fixed buffers *outside* the compiled call.

### Window sizing

`ZenTTS(max_frames=N)` sizes the RoPE table, the flat KV cache and the mask
buffer, and covers **prompt + generated frames**. The decode loop starts at
position `S` (the prompt length), so `_zen_run` emits at most `N - S` frames.
Getting this wrong is fatal rather than slow: `pos = S + frame` past the table
trips `indexSelectSmallIndex` and kills the worker with a device-side assert.
`TalkerStep._mask` now raises `IndexError` on the host instead, and `_zen_run`
raises `ValueError` up front when the prompt alone fills the window.

### What is still eager

The talker's `GenerationMixin.generate` scaffolding and the codec run eagerly;
only the inner module calls are captured. That is the floor for this design —
going below it means replacing `generate` entirely, which zen avoids so the
prompt semantics stay the library's.

## Usage

```python
import zen

# zen resolves its own vendored qwen_tts; no import of that package is needed.
model = zen.load_model(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)

zen.install(model, compile=True, max_frames=3072)   # compiles + captures both graphs

wavs, sr = model.generate_voice_design(text="...", instruct="...", language="Chinese")
```

`max_frames` is the total RoPE/KV/mask window, covering **prompt + generated
frames** — pick it to comfortably exceed the longest prompt you intend to send.
`zen.install` returns the `ZenTTS` handle and blocks for ~70 s while both graphs
compile, after which **every** request — including the first — runs at
steady-state speed. `zen.uninstall(model)` restores the stock path.

### Constraint: one thread

A captured graph may only be replayed from the thread that captured it
(cudagraph_trees keeps its tree manager in thread-local storage). A server must
therefore own a dedicated inference thread: install, warm, and every request on
the same thread. `server.py` does this with a single-worker `ThreadPoolExecutor`.

## Measured

Voice-design, 1.7B bf16, RTX A3000 Laptop, same prompt and seed:

| | wall | audio | RTF |
|---|---|---|---|
| stock | 5.38 s | 3.76 s | 1.43 |
| zen | 2.06 s | 4.16 s | **0.50** |

First request equals steady state. The gain comes from collapsing launch and
synchronisation overhead, not from making the arithmetic faster: the decode step
is bandwidth-bound, reading the talker's 2.82 GB of weights per frame against
~207 GB/s of measured memory bandwidth (≈13.6 ms of unavoidable traffic per step,
against ~15.6 ms observed).

Component numbers:

| | eager | captured |
|---|---|---|
| talker step (28 layers) | 25.7 ms | 16.6 ms |
| code predictor (15 sub-steps) | 84.1 ms | 15.2 ms |

### Reproducibility

Both paths are individually deterministic under a fixed seed, but they do **not**
agree with each other, and they are *not* different by sampling noise alone.

Greedy decoding (no sampling, single fixed seed, long prompt), run in separate
processes:

| path | frames | code-sequence sha1 |
|---|---|---|
| eager | 1151 | `1fda53fc1448b80a` |
| compiled | 1151 | `e32af0942e570c44` |

Identical length, different content. Greedy removes the sampling-threshold
amplifier, so if the gap were sampling noise the two would match; they do not.
The remaining difference is `torch.compile` fusion changing bf16 rounding points
in the forward pass. It is deterministic per path and inaudible enough that the
output was judged acceptable by ear, but it means **compiled output is not a
bit-faithful reproduction of eager** — treat them as two closely-related variants,
not the same computation.

Earlier revisions of this file claimed the divergence was "sampling noise" and
quoted a 0.46 ms talker step. Both were wrong: the sampling claim compared frame
counts without comparing contents, and the 0.46 ms probe never advanced `pos`, so
it measured a degenerate path. The real step is ~15 ms.

## Layout

```
zen/rope.py            precomputed RoPE tables (the interleaved mrope reduces to 1-D for text)
zen/talker.py          statically-shaped 28-layer decode step, graph-compiled
zen/code_predictor.py  15-sub-step code predictor decode, graph-compiled
zen/pipeline.py        ZenTTS: caches, masks, warmup, prefill absorption, sampler
zen/__init__.py        load_model / install / uninstall + the talker.generate drop-in
zen/_vendor/qwen_tts/  upstream qwen_tts, unmodified (model classes, codec, inference wrapper)
tests/                 parity, component and end-to-end checks
```

## Tests

```
python tests/test_talker.py         # TalkerStep stability + capture speedup
python tests/test_code_predictor.py # per-sub-step logits parity vs _subtalker_decode
python tests/test_window.py         # prompt+decode stay inside the RoPE/KV window
python tests/test_e2e.py            # zen vs stock end-to-end: code match, wall time, RTF
```
