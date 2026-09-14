"""Parity: zen's code-predictor decode vs the stock `_subtalker_decode`.

Compares the *logits* of every one of the 15 sub-steps, not the sampled ids:
both paths sample with `torch.multinomial`, so the drawn ids differ whenever the
RNG stream differs even for identical inputs. Logits are the meaningful
comparison — if the argmax agrees at every step the implementations are
numerically equivalent and any id difference is sampling noise.

run:  python tests/test_code_predictor.py
"""
import os, sys
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/zen-test-cp")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

import zen


def stock_logits(cp, a, b, ng, d):
    o = cp(input_ids=None, inputs_embeds=torch.cat((a, b), 1), past_key_values=None,
           cache_position=torch.arange(0, 2, device=d), use_cache=True,
           generation_steps=0, attention_mask=None)
    c = o.past_key_values
    out = [o.logits[:, -1, :].float().clone()]
    prev = o.logits[:, -1, :].float().argmax(-1, keepdim=True)
    for i in range(ng - 2):
        o = cp(input_ids=prev, inputs_embeds=None, past_key_values=c,
               cache_position=torch.tensor([2 + i], device=d), use_cache=True,
               generation_steps=1 + i, attention_mask=None)
        c = o.past_key_values
        out.append(o.logits[:, -1, :].float().clone())
        prev = o.logits[:, -1, :].float().argmax(-1, keepdim=True)
    return out


def main():
    from zen.code_predictor import CodePredictorDecode

    dev = os.environ.get("ZEN_DEV", "cuda:1")
    d = torch.device(dev)
    model_path = os.environ.get(
        "ZEN_MODEL", "/home/yoitsuholo/Code/Model/Qwen-TTS/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    m = zen.load_model(model_path, device_map=dev, dtype=torch.bfloat16)
    talker = m.model.talker
    cfg = talker.config
    ng = cfg.num_code_groups

    torch.manual_seed(7)
    a = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device=d)
    b = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device=d)

    sl = stock_logits(talker.code_predictor, a, b, ng, d)

    z = CodePredictorDecode(talker.code_predictor, ng, cfg.hidden_size, do_sample=False)
    zl = []
    orig = z._sample

    def spy(lg):
        zl.append(lg.float().clone())
        return orig(lg)

    z._sample = spy
    z(a, b)

    assert len(sl) == len(zl) == ng - 1, (len(sl), len(zl))
    ok = True
    worst = 0.0
    for i, (s, q) in enumerate(zip(sl, zl)):
        md = (s - q).abs().max().item()
        worst = max(worst, md)
        am_s, am_q = s.argmax(-1).item(), q.argmax(-1).item()
        if am_s != am_q:
            ok = False
            print(f"  step {i}: argmax {am_s} != {am_q}", flush=True)
    print(f"steps={len(sl)} argmax_all_equal={ok} max_logit_diff={worst:.4f}", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
