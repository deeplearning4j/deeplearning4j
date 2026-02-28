#!/usr/bin/env python3
r"""Smart ccache wrapper - cross-platform (Windows + Linux/macOS).

Thin wrapper that sets ccache environment variables and execs ccache.
Kept minimal to avoid per-compilation overhead (Python startup + I/O).

Usage (via CMAKE_C/CXX/CUDA_COMPILER_LAUNCHER):
    python smart_ccache.py --ccache=<path> [--debug] [--] <compiler> [args...]
"""

import sys
import os
import subprocess


def main():
    # Parse our own flags (minimal — no regex, no imports beyond stdlib)
    ccache_path = None
    debug = False
    compiler_args_start = 1

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--':
            compiler_args_start = i + 1
            break
        elif arg.startswith('--ccache='):
            ccache_path = arg[9:]
        elif arg.startswith('--hash-dir=') or arg.startswith('--build-dir='):
            pass  # Ignored — kept for CLI compat with SmartCcache.cmake
        elif arg == '--debug':
            debug = True
        else:
            compiler_args_start = i
            break
        i += 1

    if not ccache_path or compiler_args_start >= len(sys.argv):
        print("Usage: smart_ccache.py --ccache=<path> [--debug] [--] <compiler> [args...]",
              file=sys.stderr)
        sys.exit(1)

    compiler = sys.argv[compiler_args_start]
    args = sys.argv[compiler_args_start + 1:]

    # Set sloppiness for ccache
    os.environ['CCACHE_SLOPPINESS'] = (
        'file_macro,include_file_ctime,include_file_mtime,pch_defines,'
        'time_macros,file_stat_matches,clang_index_store,locale,random_seed'
    )

    # Auto-detect compiler type to help ccache
    if 'CCACHE_COMPILERTYPE' not in os.environ:
        base = os.path.basename(compiler).lower()
        if 'g++' in base or 'gcc' in base or 'mingw' in base:
            os.environ['CCACHE_COMPILERTYPE'] = 'gcc'
        elif 'clang' in base:
            os.environ['CCACHE_COMPILERTYPE'] = 'clang'
        elif 'nvcc' in base:
            os.environ['CCACHE_COMPILERTYPE'] = 'nvcc'

    # Build command: ccache compiler args...
    cmd = [ccache_path, compiler] + args

    if debug:
        print(f"[smart_ccache] {' '.join(cmd[:8])}...", file=sys.stderr)

    # Exec directly — no subprocess overhead on Unix
    if sys.platform != 'win32' and os.name != 'nt':
        os.execvp(cmd[0], cmd)
    else:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
