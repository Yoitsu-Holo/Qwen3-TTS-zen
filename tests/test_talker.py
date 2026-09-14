"""zen TalkerStep: numerics, shape stability and graph speed.

The strongest end-to-end evidence that the talker rewrite is faithful is
`tests/test_e2e.py` (frame 0's 16 code groups match the stock run exactly), and
the per-module logits parity is covered by `tests/test_code_predictor.py`.
This file checks what those do not: that the hand-written step is stable over a
long run, produces finite output, and is actually faster when compiled.

run:  python tests/test_talker.py
"""
import os, sys, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/zen-test-talker")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

import zen


def main():
    from zen.talker import TalkerStep

    dev = os.environ.get("ZEN_DEV", "cuda:1")
    d = torch.device(dev)
    model_path = os.environ.get(
        "ZEN_MODEL", "/home/yoitsuholo/Code/Model/Qwen-TTS/Qwen3-TTS-12Hz-1.7B-Base")
    m = zen.load_model(model_path, device_map=dev, dtype=torch.bfloat16)
    talker = m.model.talker
    cfg = talker.config

    torch.manual_seed(0)
    X = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device=d)
    NSTEP = 200

    step = TalkerStep(talker, max_len=512).to(d)
    with torch.no_grad():
        step.reset()
        out = []
        for p in range(NSTEP):
            out.append(step(X, torch.tensor([p], device=d), step._mask(p)))
        h = out[-1]
    fin = bool(torch.isfinite(h).all())
    mag = h.float().abs().max().item()
    print(f"zen talker {NSTEP} steps: finite={fin} max_abs={mag:.2f} "
          f"out={tuple(h.shape)}", flush=True)
    assert fin, "non-finite output"

    with torch.no_grad():
        for p in range(3):
            step.reset()
            step(X, torch.tensor([p], device=d), step._mask(p))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for p in range(100):
            step(X, torch.tensor([p], device=d), step._mask(p))
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / 100 * 1000

        cs = torch.compile(step, mode="reduce-overhead", fullgraph=True)
        step.reset()
        cs(X, torch.tensor([0], device=d), step._mask(0))
        torch.cuda.synchronize()
        for p in range(1, 4):
            cs(X, torch.tensor([p], device=d), step._mask(p))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for p in range(4, 104):
            cs(X, torch.tensor([p], device=d), step._mask(p))
        torch.cuda.synchronize()
        comp_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"talker step: eager {eager_ms:.2f} ms  compiled {comp_ms:.2f} ms  "
          f"({eager_ms/comp_ms:.2f}x)", flush=True)
    assert comp_ms < eager_ms, "compile did not help"
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
