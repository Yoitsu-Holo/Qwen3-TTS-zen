# Vendored `qwen_tts`

This directory is an **unmodified copy** of the upstream `qwen_tts` package, taken
from the site-packages install of the Qwen-TTS venv:

    venv/lib/python3.14/site-packages/qwen_tts

It is vendored so that `zen` depends on no external checkout of that library. The
`Qwen3-TTS-opt` fork is *not* the source: its only functional delta from upstream
was a hand-rolled `_subtalker_decode` reached from `talker.forward()`, and zen
replaces `talker.generate` itself, so that code is never executed here.

To refresh from a newer upstream, re-copy and diff against this tree — it should be
byte-identical so the diff shows only genuine upstream changes.

Trimmed relative to the installed wheel (not needed as a library):

    __main__.py      console entry point
    cli/demo.py      interactive demo

Everything else, including the 25Hz tokenizer stack, is kept verbatim: the 12Hz
inference wrapper (`inference/qwen3_tts_tokenizer.py`) imports the V1 classes from
`..core` and registers both model types at import time, so removing them would
require editing vendored code.
