"""Audited build-launcher-only source equivalence for retained worker recovery.

No source POM, compiled source or artifact receipt is rewritten. The two exact
revisions differ only in tokenizer reactor selection and its offline regression.
This does NOT allow arbitrary different source commits to be combined.
"""
import argparse

CUDA_SOURCE = 'ecc4a7c4210b65ee098a58563671cd47896ab4be'
CPU_SOURCE = '131fa37e366eff6f7c8888cb2adc21b84a9360bf'


def check(expected, actual):
    if expected == actual:
        return
    if {expected, actual} != {CUDA_SOURCE, CPU_SOURCE}:
        raise ValueError(f'Unaudited worker source difference: {expected} / {actual}')
    print(f'AUDITED launcher-only source equivalence: {expected} / {actual}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected')
    parser.add_argument('actual')
    args = parser.parse_args()
    check(args.expected, args.actual)
