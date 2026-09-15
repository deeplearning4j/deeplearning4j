"""Audited build-launcher-only source equivalence for retained worker recovery.

No source POM, compiled source or artifact receipt is rewritten. Each audited
pair below differs only in build configuration that cannot alter artifact
bytes outside the lanes that own the changed files. This does NOT allow
arbitrary different source commits to be combined.
"""
import argparse

CUDA_SOURCE = 'ecc4a7c4210b65ee098a58563671cd47896ab4be'
CPU_SOURCE = '131fa37e366eff6f7c8888cb2adc21b84a9360bf'

# ZLUDA family, 2026-09-14: dbce583d00 over a412cd7974 differs only in the
# nd4j-zluda-12.9-platform classifier property composition and its test
# expectations. Lanes not reading nd4j-zluda-12.9-platform (CPU owners, CUDA
# 12.9 mainline owner, ZLUDA base variants) build byte-identical artifacts.
ZLUDA_BASE_SOURCE = 'a412cd7974408b284d596f00013f877b1ce474eb'
ZLUDA_COMPILE_SOURCE = 'dbce583d00e0439e8f41e4cd8206f36dea8e20bd'

# ZLUDA family, 2026-09-14: efb8fc32c6 over dbce583d00 changes only release
# packaging determinism: .gitattributes LF normalization, pinned JAR manifest
# entries, disabled git.properties generation and dropped Maven descriptors in
# shared ZLUDA/CUDA modules. No compiled source changes; owners on dbce583d00
# produce byte-identical artifacts for everything they own.
ZLUDA_OWNER_SOURCE = 'dbce583d00e0439e8f41e4cd8206f36dea8e20bd'
ZLUDA_OSNEUTRAL_SOURCE = 'efb8fc32c6822026d8454f7501bd39c7bafadad0'

# ZLUDA family, 2026-09-14: ed436cfbd9 over efb8fc32c6 adds sources/javadoc
# attachments to the platform aggregator modules (new descriptor sources plus
# javadoc sourcepath configuration). Only the platform modules changed; the
# CPU owners never build them, so their artifacts are byte-identical.
ZLUDA_ATTACHMENTS_SOURCE = 'ed436cfbd9d7e0f2504a3e2f6e8b84063ca7fd80'

AUDITED_PAIRS = [
    {CUDA_SOURCE, CPU_SOURCE},
    {ZLUDA_BASE_SOURCE, ZLUDA_COMPILE_SOURCE},
    {ZLUDA_OWNER_SOURCE, ZLUDA_OSNEUTRAL_SOURCE},
    {ZLUDA_OSNEUTRAL_SOURCE, ZLUDA_ATTACHMENTS_SOURCE},
]


def check(expected, actual):
    if expected == actual:
        return
    if {expected, actual} not in AUDITED_PAIRS:
        raise ValueError(f'Unaudited worker source difference: {expected} / {actual}')
    print(f'AUDITED launcher-only source equivalence: {expected} / {actual}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected')
    parser.add_argument('actual')
    args = parser.parse_args()
    check(args.expected, args.actual)
