"""Run a MiniMind trainer with torch.compile(dynamic=True) injected; trainer file unmodified.

Option #2 from EXP-008: symbolic shapes from the first trace, so varying expert token
counts should not burn through dynamo's recompile_limit. The trainer's own
`model = torch.compile(model)` call is intercepted and given dynamic=True.

Usage (inside the container, FROM THE trainer/ DIRECTORY so relative paths match the
normal launch commands):
    cd /workspace/minimind/trainer
    python ../scripts/run_dynamic_compile_probe.py train_pretrain.py \
        --use_moe 1 --max_seq_len 768 --use_compile 1 ... (normal trainer flags)
"""
import sys
import runpy
import torch

_orig_compile = torch.compile


def _dynamic_compile(model, *args, **kwargs):
    kwargs.setdefault('dynamic', True)
    return _orig_compile(model, *args, **kwargs)


torch.compile = _dynamic_compile

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
print(f'[dynamic-compile probe] running {target} with torch.compile(dynamic=True)', flush=True)
runpy.run_path(target, run_name='__main__')
