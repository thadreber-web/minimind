"""Train the MiniMind tokenizer on a jsonl file and record time, CPU use and compression.

Wraps trainer/train_tokenizer.py without modifying it. Run inside the container.
"""
import argparse
import json
import os
import resource
import sys
import threading
import time

import psutil
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trainer'))
import train_tokenizer as tt


def chars_per_token(tok_path, texts):
    tok = Tokenizer.from_file(tok_path)
    chars = sum(len(t) for t in texts)
    tokens = sum(len(tok.encode(t).ids) for t in texts)
    return chars / tokens, tok.get_vocab_size()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True, help='tokenizer output dir')
    p.add_argument('--heldout', required=True, help='jsonl of unseen text for the compression check')
    p.add_argument('--results', required=True, help='results .json path')
    p.add_argument('--vocab-size', type=int, default=tt.VOCAB_SIZE)
    args = p.parse_args()

    # time spent inside the Python text generator (file read + json parse)
    gen_time = [0.0]
    orig_get_texts = tt.get_texts

    def timed_get_texts(path, *a, **k):
        it = orig_get_texts(path, *a, **k)
        while True:
            t0 = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                gen_time[0] += time.perf_counter() - t0
                return
            gen_time[0] += time.perf_counter() - t0
            yield item
    tt.get_texts = timed_get_texts

    # sample CPU% and thread count once a second
    proc = psutil.Process()
    samples = []
    stop = threading.Event()

    def sampler():
        proc.cpu_percent(None)
        while not stop.wait(1.0):
            samples.append((proc.cpu_percent(None), proc.num_threads()))
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    t0 = time.perf_counter()
    tt.train_tokenizer(args.data, args.out, args.vocab_size)
    wall = time.perf_counter() - t0
    stop.set()
    th.join()

    ru = resource.getrusage(resource.RUSAGE_SELF)
    cpu = ru.ru_utime + ru.ru_stime
    heldout = [json.loads(l)['text'] for l in open(args.heldout, encoding='utf-8')]
    new_cpt, new_vocab = chars_per_token(os.path.join(args.out, 'tokenizer.json'), heldout)
    base_cpt, base_vocab = chars_per_token(os.path.join(os.path.dirname(tt.__file__), '..', 'model', 'tokenizer.json'), heldout)
    n_lines = sum(1 for _ in open(args.data, encoding='utf-8'))
    res = {
        'data': args.data, 'lines': n_lines, 'vocab_size_requested': args.vocab_size,
        'wall_s': round(wall, 2), 'cpu_user_s': round(ru.ru_utime, 2), 'cpu_sys_s': round(ru.ru_stime, 2),
        'cpu_over_wall': round(cpu / wall, 2), 'max_rss_mb': round(ru.ru_maxrss / 1024),
        'text_generator_s': round(gen_time[0], 2),
        'cpu_count': os.cpu_count(), 'RAYON_NUM_THREADS': os.environ.get('RAYON_NUM_THREADS'),
        'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM'),
        'peak_cpu_percent': max((s[0] for s in samples), default=None),
        'mean_cpu_percent': round(sum(s[0] for s in samples) / len(samples), 1) if samples else None,
        'peak_threads': max((s[1] for s in samples), default=None),
        'heldout_texts': len(heldout),
        'new_tokenizer': {'vocab': new_vocab, 'chars_per_token': round(new_cpt, 3)},
        'repo_tokenizer': {'vocab': base_vocab, 'chars_per_token': round(base_cpt, 3)},
    }
    with open(args.results, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
