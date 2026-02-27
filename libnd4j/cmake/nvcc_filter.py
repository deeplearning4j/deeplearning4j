#!/usr/bin/env python3
"""NVCC wrapper that filters out MSVC -Fd/-FS flags from response files and arguments.

CMake + Ninja on Windows generates -Xcompiler=-Fd<path>\,-FS in CUDA response files.
The backslash-comma causes nvcc to misparse this as two arguments, with -FS interpreted
as an input file, triggering: "A single input file is required".

This wrapper:
1. Intercepts --options-file arguments
2. Reads and filters the response file content
3. Writes a cleaned response file
4. Passes all other args through, filtering direct -Fd/-FS flags
5. Invokes the real nvcc with cleaned arguments

Usage (via CMAKE_CUDA_COMPILER_LAUNCHER):
    python nvcc_filter.py nvcc.exe [args...]
"""

import sys
import os
import re
import subprocess
import tempfile

def should_filter_arg(arg):
    """Return True if this argument should be stripped."""
    # Filter -Xcompiler args containing -Fd or -FS (PDB debug info flags)
    if re.match(r'-Xcompiler[=:].*-Fd', arg):
        return True
    if re.match(r'-Xcompiler[=:].*-FS', arg):
        return True
    if arg == '-Xcompiler=-FS':
        return True
    # Also filter standalone -Fd... args that might appear
    if arg.startswith('-Fd'):
        return True
    if arg == '-FS' or arg == '/FS':
        return True
    if arg.startswith('/Fd'):
        return True
    return False


def filter_response_file(rsp_path):
    """Read a response file, filter problematic flags, write cleaned version.

    Returns path to the cleaned response file (may be a new temp file).
    """
    if not os.path.isfile(rsp_path):
        return rsp_path

    with open(rsp_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    original = content

    # Filter lines/tokens containing -Fd or -FS in -Xcompiler context
    # Response file can have args on separate lines or space-separated
    lines = content.splitlines()
    filtered_lines = []
    for line in lines:
        # Split line into tokens (respecting quotes)
        tokens = []
        current = []
        in_quote = False
        for ch in line:
            if ch == '"':
                in_quote = not in_quote
                current.append(ch)
            elif ch == ' ' and not in_quote:
                if current:
                    tokens.append(''.join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            tokens.append(''.join(current))

        filtered_tokens = [t for t in tokens if not should_filter_arg(t)]
        if filtered_tokens:
            filtered_lines.append(' '.join(filtered_tokens))

    new_content = '\n'.join(filtered_lines) + '\n'

    if new_content != original:
        # Write filtered content to a new temp file in the same directory
        rsp_dir = os.path.dirname(rsp_path) or '.'
        fd, tmp_path = tempfile.mkstemp(suffix='.rsp', dir=rsp_dir, prefix='nvcc_filtered_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return tmp_path

    return rsp_path


def main():
    if len(sys.argv) < 2:
        print("Usage: nvcc_filter.py <nvcc_path> [args...]", file=sys.stderr)
        sys.exit(1)

    nvcc = sys.argv[1]
    args = sys.argv[2:]

    filtered_args = []
    temp_files = []

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle --options-file <path> (two separate args)
        if arg == '--options-file' and i + 1 < len(args):
            rsp_path = args[i + 1]
            new_rsp = filter_response_file(rsp_path)
            filtered_args.append('--options-file')
            filtered_args.append(new_rsp)
            if new_rsp != rsp_path:
                temp_files.append(new_rsp)
            i += 2
            continue

        # Handle --options-file=<path> (combined)
        if arg.startswith('--options-file='):
            rsp_path = arg[len('--options-file='):]
            new_rsp = filter_response_file(rsp_path)
            filtered_args.append('--options-file=' + new_rsp)
            if new_rsp != rsp_path:
                temp_files.append(new_rsp)
            i += 1
            continue

        # Handle @<path> response file syntax
        if arg.startswith('@') and len(arg) > 1:
            rsp_path = arg[1:]
            new_rsp = filter_response_file(rsp_path)
            filtered_args.append('@' + new_rsp)
            if new_rsp != rsp_path:
                temp_files.append(new_rsp)
            i += 1
            continue

        # Filter direct args
        if not should_filter_arg(arg):
            filtered_args.append(arg)

        i += 1

    try:
        result = subprocess.run([nvcc] + filtered_args)
        sys.exit(result.returncode)
    finally:
        # Cleanup temp files
        for tf in temp_files:
            try:
                os.unlink(tf)
            except OSError:
                pass


if __name__ == '__main__':
    main()
