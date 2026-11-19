"""Find held-out documents that have a near-duplicate in the training corpus (MinHash + LSH) and write cleaned held-out files.

Shingles are 5-grams of lowercase word tokens (first 2,000 tokens of each document). Run inside the container.
"""
import argparse
import json
import os
import re
from multiprocessing import Pool

import numpy as np
from datasketch import MinHash, MinHashLSH

TOKEN = re.compile(r'\w+')
NUM_PERM = 64
MAX_TOKENS = 2000
SHINGLE = 5


def signature(text):
    toks = TOKEN.findall(text.lower())[:MAX_TOKENS]
    m = MinHash(num_perm=NUM_PERM)
    if len(toks) < SHINGLE:
        m.update(' '.join(toks).encode())
    else:
        m.update_batch([' '.join(toks[i:i + SHINGLE]).encode() for i in range(len(toks) - SHINGLE + 1)])
    return m.hashvalues


def read(path):
    return [json.loads(l)['text'] for l in open(path, encoding='utf-8')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--heldout', nargs='+', required=True, help='name=path.jsonl')
    p.add_argument('--outdir', required=True)
    p.add_argument('--threshold', type=float, default=0.7)
    p.add_argument('--workers', type=int, default=18)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    corpus = read(args.corpus)
    held = {n: read(path) for n, path in (a.split('=', 1) for a in args.heldout)}
    # positive controls: 10 corpus documents (>= 200 words) with every 40th word replaced must be found
    rng = np.random.RandomState(0)
    long_ids = [i for i, t in enumerate(corpus) if len(t.split()) >= 200]
    ctl_ids = [int(i) for i in rng.choice(long_ids, 10, replace=False)]
    controls = [' '.join('zzqx' if j % 40 == 0 else w for j, w in enumerate(corpus[i].split())) for i in ctl_ids]
    unrelated = 'The quick brown fox jumps over the lazy dog near the quiet river bank today and then goes home.'

    with Pool(args.workers) as pool:
        corpus_sigs = pool.map(signature, corpus, chunksize=500)
        held_sigs = {n: pool.map(signature, t, chunksize=50) for n, t in held.items()}
        control_sig = pool.map(signature, controls + [unrelated])

    lsh = MinHashLSH(threshold=args.threshold, num_perm=NUM_PERM)
    for i, h in enumerate(corpus_sigs):
        lsh.insert(str(i), MinHash(num_perm=NUM_PERM, hashvalues=h))

    def has_dup(h):
        return bool(lsh.query(MinHash(num_perm=NUM_PERM, hashvalues=h)))

    report = {'corpus_docs': len(corpus), 'threshold': args.threshold, 'shingle_words': SHINGLE,
              'controls_modified_copies_found': f'{sum(has_dup(h) for h in control_sig[:-1])}/{len(controls)}', 'control_unrelated_found': has_dup(control_sig[-1]), 'heldout': {}}
    for n, sigs in held_sigs.items():
        dup = [has_dup(h) for h in sigs]
        keep = [t for t, d in zip(held[n], dup) if not d]
        with open(os.path.join(args.outdir, f'heldout_{n}_dedup.jsonl'), 'w', encoding='utf-8') as f:
            for t in keep:
                f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')
        report['heldout'][n] = {'docs': len(dup), 'near_duplicates': sum(dup), 'kept': len(keep)}
    json.dump(report, open(os.path.join(args.outdir, 'near_duplicates.json'), 'w'), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
