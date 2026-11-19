"""Summarize speed-probe TensorBoard runs (scripts/run_speed_probe.sh).

Reads wall-clock timestamps of the 'loss' scalar per probe run, computes steady-state
seconds/step (median, warmup points skipped), throughput, projected full-stage ETA,
and the torch.compile speedup. Run inside the container from the repo root:
    python scripts/summarize_speed_probe.py
"""
import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = os.environ.get('TB_LOGDIR', 'tb_logs')
LOG_INTERVAL = 5  # must match --log_interval in run_speed_probe.sh
# project, label, positions per microbatch step, total microbatch steps in the real run
# (pretrain: 4344 optimizer steps x acc 8; SFT: 42,004 steps at acc 1; both pad to seq 768)
CFGS = [
    ('probe-pt-b32',          'pretrain b32',          32 * 768, 4344 * 8),
    ('probe-pt-b32-compile',  'pretrain b32 compile',  32 * 768, 4344 * 8),
    ('probe-sft-b16',         'sft b16',               16 * 768, 42004),
    ('probe-sft-b16-compile', 'sft b16 compile',       16 * 768, 42004),
]

results = {}
for proj, label, tok, total in CFGS:
    dirs = sorted(glob.glob(os.path.join(TB, proj, '*')))
    if not dirs:
        print(f'{label:24s} MISSING - run failed? check experiments/probe/{proj.replace("probe-", "")}.log')
        continue
    ea = EventAccumulator(dirs[-1])
    ea.Reload()
    try:
        ev = ea.Scalars('loss')
    except KeyError:
        print(f'{label:24s} NO SCALARS - run died before first log? check experiments/probe/')
        continue
    wt = np.array([e.wall_time for e in ev])
    st = np.array([e.step for e in ev])
    if len(wt) < 6:
        print(f'{label:24s} too few points ({len(wt)}) - crashed early? check experiments/probe/')
        continue
    # st is the tracker's LOG COUNTER (+1 per log call), not the training step:
    # each log interval spans LOG_INTERVAL training steps.
    per_step = np.diff(wt) / np.diff(st) / LOG_INTERVAL
    warm = per_step[4:]  # skip compile/warmup points
    med = float(np.median(warm))
    spikes = int((warm > 3 * med).sum())  # recompile or save spikes
    results[proj] = med
    span_avg = (wt[-1] - wt[0]) / ((st[-1] - st[0]) * LOG_INTERVAL)  # independent cross-check
    print(f'{label:24s} {med:7.3f} s/step  {tok / med:10,.0f} pos/s  '
          f'projected stage ETA {total * med / 3600:5.1f} h  '
          f'(points {len(wt)}, spikes {spikes}, span-avg {span_avg:.3f} s/step)')

for a, b, stage in [('probe-pt-b32', 'probe-pt-b32-compile', 'pretrain'),
                    ('probe-sft-b16', 'probe-sft-b16-compile', 'sft')]:
    if a in results and b in results:
        print(f'{stage} compile speedup: {results[a] / results[b]:.2f}x')

print('note: ETA assumes probe steady state holds at full dataset scale; '
      'spikes > 0 with compile suggest recompiles (dynamic expert token counts).')
