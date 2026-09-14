"""End-to-end: zen vs stock, same seed, same prompt.

Checks
  1. the emitted code sequence matches the stock path (zen is meant to be a
     drop-in for the hot loops, not a different model)
  2. wall time and RTF improve

The code sequence is captured by spying on `speech_tokenizer.decode`, whose
argument is exactly the talker's codec output — the ground truth for "similar
result".

run:  ZEN_DEV=cuda:1 python tests/test_e2e.py
"""
import os, sys, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/zen-e2e")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

TEXT = "你好，今天天气不错，我们一起去公园散步吧。"
INSTRUCT = "平静温和清晰的女声"


def main():

    dev = os.environ.get("ZEN_DEV", "cuda:1")
    model_path = os.environ.get("ZEN_MODEL",
                                "/home/yoitsuholo/Code/Model/Qwen-TTS/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    m = zen.load_model(model_path, device_map=dev, dtype=torch.bfloat16)
    tok = m.model.speech_tokenizer

    captured = {}

    def spy(tag):
        orig = tok.decode

        def wrap(item, *a, **kw):
            c = item[0]["audio_codes"] if isinstance(item, list) else item
            captured[tag] = c.detach().cpu().clone()
            return orig(item, *a, **kw)
        return orig, wrap

    def run(tag, n=2):
        orig, wrap = spy(tag)
        tok.decode = wrap
        ts = []
        try:
            for _ in range(n):
                torch.manual_seed(1234)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                w = m.generate_voice_design(text=TEXT, instruct=INSTRUCT, language="Chinese")
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        finally:
            tok.decode = orig
        wavs, fs = w
        wav = wavs[0]
        print(f"{tag}: {[f'{t:.2f}' for t in ts]}  min={min(ts):.2f}s  "
              f"audio={len(wav)/fs:.2f}s  RTF={min(ts)/(len(wav)/fs):.3f}", flush=True)
        return min(ts), len(wav) / fs

    print("--- stock ---", flush=True)
    t_stock, a_stock = run("stock")

    print("--- install zen ---", flush=True)
    import zen
    t_inst = time.perf_counter()
    z = zen.install(m, compile=True, max_frames=2048)
    print(f"installed (+warm) in {time.perf_counter()-t_inst:.1f}s", flush=True)

    print("--- zen ---", flush=True)
    t_zen, a_zen = run("zen")

    cs, cz = captured.get("stock"), captured.get("zen")
    if cs is not None and cz is not None:
        print(f"stock codes {tuple(cs.shape)}   zen codes {tuple(cz.shape)}", flush=True)
        n = min(cs.shape[0], cz.shape[0])
        same0 = (cs[:n, 0] == cz[:n, 0])
        print(f"frames={n}  code0 match={int(same0.sum())}/{n}  "
              f"exact={bool(same0.all())}", flush=True)
        print(f"stock frame0: {cs[0].tolist()}", flush=True)
        print(f"zen   frame0: {cz[0].tolist()}", flush=True)
    print(f"\nSUMMARY  stock {t_stock:.2f}s (RTF {t_stock/a_stock:.3f})  "
          f"zen {t_zen:.2f}s (RTF {t_zen/a_zen:.3f})  "
          f"speedup {t_stock/t_zen:.2f}x", flush=True)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
