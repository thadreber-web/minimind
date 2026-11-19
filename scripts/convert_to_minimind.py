"""Convert a Hugging Face dataset into MiniMind jsonl (pretrain or sft format)."""
import argparse
import json
import os

from datasets import load_dataset

ROLE_MAP = {'human': 'user', 'user': 'user', 'gpt': 'assistant', 'assistant': 'assistant', 'system': 'system'}


def to_pretrain(row, args):
    text = str(row.get(args.text_field) or '').strip()
    return {'text': text} if len(text) >= args.min_chars else None


def to_sft(row, args):
    if args.messages_field:
        # Already-chat rows: [{role, content}] or ShareGPT [{from, value}]
        conversations = []
        for m in row.get(args.messages_field) or []:
            role = ROLE_MAP.get(str(m.get('role') or m.get('from') or '').lower())
            content = str(m.get('content') or m.get('value') or '').strip()
            if role and content:
                conversations.append({'role': role, 'content': content})
        if not any(c['role'] == 'assistant' for c in conversations):
            return None
    else:
        instruction = str(row.get(args.instruction_field) or '').strip()
        extra = str(row.get(args.input_field) or '').strip() if args.input_field else ''
        response = str(row.get(args.response_field) or '').strip()
        if not instruction or not response:
            return None
        conversations = [
            {'role': 'user', 'content': f'{instruction}\n{extra}' if extra else instruction},
            {'role': 'assistant', 'content': response},
        ]
    if sum(len(c['content']) for c in conversations) < args.min_chars:
        return None
    return {'conversations': conversations}


def main():
    p = argparse.ArgumentParser(description='Convert a Hugging Face dataset to MiniMind jsonl.')
    p.add_argument('--dataset', required=True, help='HF dataset id or local path')
    p.add_argument('--config', default=None, help='dataset config name')
    p.add_argument('--split', default='train', help='dataset split')
    p.add_argument('--streaming', action='store_true', help='stream rows instead of downloading the full dataset')
    p.add_argument('--format', required=True, choices=['pretrain', 'sft'], help='output format')
    p.add_argument('--out', required=True, help='output .jsonl path')
    p.add_argument('--text-field', default='text', help='pretrain: column holding the text')
    p.add_argument('--messages-field', default=None, help='sft: column holding a list of turns')
    p.add_argument('--instruction-field', default=None, help='sft single-turn: user prompt column')
    p.add_argument('--input-field', default=None, help='sft single-turn: optional context appended to the prompt')
    p.add_argument('--response-field', default=None, help='sft single-turn: assistant answer column')
    p.add_argument('--max-samples', default=0, type=int, help='stop after N written rows (0 = no limit)')
    p.add_argument('--min-chars', default=1, type=int, help='skip samples shorter than this many characters')
    args = p.parse_args()
    if args.format == 'sft' and not args.messages_field and not (args.instruction_field and args.response_field):
        p.error('sft needs --messages-field, or both --instruction-field and --response-field')

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    convert = to_pretrain if args.format == 'pretrain' else to_sft
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written = skipped = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in ds:
            if args.max_samples and written >= args.max_samples:
                break
            item = convert(row, args)
            if item is None:
                skipped += 1
                continue
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            written += 1
    print(f'wrote {written} rows to {args.out} (skipped {skipped})')


if __name__ == '__main__':
    main()
