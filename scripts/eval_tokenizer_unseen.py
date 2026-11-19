"""Compare tokenizers on unseen text: compression, round-trip, overlap with the training corpus, chat template.

Run inside the container from /workspace/minimind.
"""
import argparse
import hashlib
import json
import statistics

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def load_texts(path):
    return [json.loads(l)['text'] for l in open(path, encoding='utf-8')]


def stats(tok, texts):
    ids = [tok.encode(t).ids for t in texts]
    chars = sum(len(t) for t in texts)
    tokens = sum(len(i) for i in ids)
    per_doc = [len(t) / len(i) for t, i in zip(texts, ids) if i]
    exact = sum(tok.decode(i) == t for t, i in zip(texts, ids))
    return {
        'chars_per_token': round(chars / tokens, 3),
        'median_doc_chars_per_token': round(statistics.median(per_doc), 3),
        'roundtrip_exact': f'{exact}/{len(texts)}',
        'total_tokens': tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizers', nargs='+', required=True, help='name=path/to/tokenizer.json')
    p.add_argument('--texts', nargs='+', required=True, help='name=path/to/file.jsonl')
    p.add_argument('--corpus', required=True, help='training corpus jsonl, for the overlap check')
    p.add_argument('--chat-dir', required=True, help='tokenizer dir (with tokenizer_config.json) for the chat template check')
    p.add_argument('--results', required=True)
    args = p.parse_args()

    toks = {n: Tokenizer.from_file(path) for n, path in (a.split('=', 1) for a in args.tokenizers)}
    sets = {n: load_texts(path) for n, path in (a.split('=', 1) for a in args.texts)}

    corpus_hashes = {hashlib.md5(json.loads(l)['text'].encode()).hexdigest() for l in open(args.corpus, encoding='utf-8')}
    res = {'overlap_with_training_corpus': {}, 'compression': {}}
    for sname, texts in sets.items():
        res['overlap_with_training_corpus'][sname] = f"{sum(hashlib.md5(t.encode()).hexdigest() in corpus_hashes for t in texts)}/{len(texts)}"
        res['compression'][sname] = {tname: stats(tok, texts) for tname, tok in toks.items()}

    zh = '人工智能是计算机科学的一个分支。你好，世界！'
    res['chinese_roundtrip'] = {n: {'tokens': len(t.encode(zh).ids), 'exact': t.decode(t.encode(zh).ids) == zh} for n, t in toks.items()}

    hf = AutoTokenizer.from_pretrained(args.chat_dir)
    msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Why is the sky blue?'},
            {'role': 'assistant', 'content': 'Because of Rayleigh scattering.'}]
    prompt = hf.apply_chat_template(msgs, tokenize=False)
    ids = hf(prompt)['input_ids']
    res['chat_template'] = {
        'prompt': prompt,
        'n_tokens': len(ids),
        'decode_matches_prompt': hf.decode(ids, skip_special_tokens=False) == prompt,
        'special_tokens_single_id': {s: hf.convert_tokens_to_ids(s) for s in ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>']},
        'len_tokenizer': len(hf),
    }
    json.dump(res, open(args.results, 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
