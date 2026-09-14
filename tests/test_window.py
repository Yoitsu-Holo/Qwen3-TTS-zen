"""Regression: the decode loop must stay inside the talker window.

`ZenTTS(max_frames=N)` sizes the RoPE table, the flat KV cache and the mask
buffer, and covers prompt + generated frames. The decode loop begins at position
`S` (prompt length), so it may emit at most `N - S` frames.

Before this was enforced, `pos = S + frame` ran past the table and tripped
`indexSelectSmallIndex: Assertion srcIndex < srcSelectDimSize`, killing the
worker with a device-side assert. Greedy decoding (which never stops early) walks
straight to the limit, so it is the natural stress case.

run:  python tests/test_window.py
"""
import os, sys
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/zen-test-window")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

TEXT = "你好，这是一段非常长的测试文本，用来寻找生成时长的上限。" * 12
INSTRUCT = "平静温和清晰的女声"


def main():
    import zen

    dev = os.environ.get("ZEN_DEV", "cuda:0")
    model_path = os.environ.get(
        "ZEN_MODEL", "/home/yoitsuholo/Code/Model/Qwen-TTS/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ok = True

    # --- host-side guard: a prompt that fills the window must raise, not assert
    m = zen.load_model(model_path, device_map=dev, dtype=torch.bfloat16)
    z = zen.install(m, compile=False, max_frames=300)
    tstep = getattr(z.talker_step, "_orig_mod", z.talker_step)
    try:
        tstep._mask(tstep.max_len)
        print("FAIL: mask past the window did not raise")
        ok = False
    except IndexError:
        print(f"ok  mask guard raises IndexError at pos={tstep.max_len}")
    del m, z
    torch.cuda.empty_cache()

    # --- greedy to the limit, eager and compiled, must not fault
    for tag, comp in (("eager", False), ("compiled", True)):
        m = zen.load_model(model_path, device_map=dev, dtype=torch.bfloat16)
        zen.install(m, compile=comp, max_frames=1400,
                    do_sample=False, subtalker_do_sample=False)
        torch.manual_seed(1234)
        wavs, fs = m.generate_voice_design(
            text=TEXT, instruct=INSTRUCT, language="Chinese",
            do_sample=False, subtalker_dosample=False)
        print(f"ok  {tag:9s} greedy to the limit: {len(wavs[0])/fs:.1f}s audio, no fault")
        del m
        torch.cuda.empty_cache()

    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
