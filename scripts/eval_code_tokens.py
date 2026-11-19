"""Token counts for code-specific strings (indentation, operators, snippets) across tokenizers.

Run inside the container from /workspace/minimind: python scripts/eval_code_tokens.py name=path ...
"""
import json
import sys

from tokenizers import Tokenizer

CASES = {
    'indent 4 spaces': '    ',
    'indent 8 spaces': '        ',
    'indent 12 spaces': '            ',
    'tab': '\t',
    'newline+indent': '\n    ',
    'def line': 'def add(a, b):',
    'return': '    return a + b',
    'class line': 'class Foo(object):',
    'import line': 'import numpy as np',
    'for loop': 'for i in range(len(xs)):',
    'f-string': 'print(f"{name}: {value:.2f}")',
    'docstring': '"""Return the sum."""',
    'operators': '== != <= >= -> ** // += -= :=',
    'brackets': '()[]{}',
    'comment': '# TODO: fix this',
    'json': '{"key": [1, 2, 3], "ok": true}',
    'snippet': 'def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n',
}


def main():
    toks = {n: Tokenizer.from_file(p) for n, p in (a.split('=', 1) for a in sys.argv[1:])}
    res = {case: {n: len(t.encode(s).ids) for n, t in toks.items()} for case, s in CASES.items()}
    ok = {n: all(t.decode(t.encode(s).ids) == s for s in CASES.values()) for n, t in toks.items()}
    print(json.dumps({'tokens_per_case': res, 'roundtrip_all_exact': ok}, indent=2))


if __name__ == '__main__':
    main()
