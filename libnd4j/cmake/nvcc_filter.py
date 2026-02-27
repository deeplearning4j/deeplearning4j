#!/usr/bin/env python3
r"""NVCC wrapper that filters out MSVC -Fd/-FS flags from response files and arguments.

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

# Log counter for diagnostics (limit verbose output to first few files)
_log_count = 0
_MAX_VERBOSE_LOGS = 5


def should_filter_arg(arg):
    """Return True if this argument should be stripped."""
    # Filter -Xcompiler args containing -Fd or -FS (PDB debug info flags)
    if re.search(r'-Xcompiler.*-Fd', arg):
        return True
    if re.search(r'-Xcompiler.*-FS', arg):
        return True
    if re.search(r'-Xcompiler.*/Fd', arg):
        return True
    if re.search(r'-Xcompiler.*/FS', arg):
        return True
    if arg == '-Xcompiler=-FS' or arg == '-Xcompiler=/FS':
        return True
    # Also filter standalone -Fd... args that might appear
    if arg.startswith('-Fd') or arg.startswith('/Fd'):
        return True
    if arg in ('-FS', '/FS'):
        return True
    return False


def filter_response_file(rsp_path):
    """Read a response file, filter problematic flags, write cleaned version.

    Returns path to the cleaned response file (may be a new temp file).
    """
    global _log_count

    if not os.path.isfile(rsp_path):
        return rsp_path

    # Read in binary to preserve exact content for comparison
    with open(rsp_path, 'rb') as f:
        raw_content = f.read()

    # Decode and normalize line endings to \n for processing
    content = raw_content.decode('utf-8', errors='replace')
    # Detect original line ending style
    has_crlf = b'\r\n' in raw_content
    line_ending = '\r\n' if has_crlf else '\n'

    lines = content.splitlines()

    # Dump first few response files for diagnostics
    _log_count += 1
    if _log_count <= _MAX_VERBOSE_LOGS:
        print(f"[nvcc_filter] === Response file: {rsp_path} ===", file=sys.stderr)
        print(f"[nvcc_filter] Line count: {len(lines)}, CRLF: {has_crlf}", file=sys.stderr)
        for i, line in enumerate(lines[:30]):
            print(f"[nvcc_filter]   L{i}: {line[:300]}", file=sys.stderr)
        if len(lines) > 30:
            print(f"[nvcc_filter]   ... ({len(lines) - 30} more lines)", file=sys.stderr)

    filtered_lines = []
    removed_lines = []
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
        removed_tokens = [t for t in tokens if should_filter_arg(t)]

        if removed_tokens:
            removed_lines.append((line, removed_tokens))

        if filtered_tokens:
            filtered_lines.append(' '.join(filtered_tokens))

    if removed_lines:
        new_content = line_ending.join(filtered_lines) + line_ending
        if _log_count <= _MAX_VERBOSE_LOGS:
            print(f"[nvcc_filter] FILTERED {len(removed_lines)} lines:", file=sys.stderr)
            for orig_line, removed_toks in removed_lines:
                print(f"[nvcc_filter]   REMOVED tokens: {removed_toks}", file=sys.stderr)

        # Write filtered content to a new temp file in the same directory
        rsp_dir = os.path.dirname(rsp_path) or '.'
        fd, tmp_path = tempfile.mkstemp(suffix='.rsp', dir=rsp_dir, prefix='nvcc_filtered_')
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(new_content)
        return tmp_path
    else:
        if _log_count <= _MAX_VERBOSE_LOGS:
            print(f"[nvcc_filter] No PDB flags found to filter", file=sys.stderr)
        return rsp_path


def main():
    if len(sys.argv) < 2:
        print("Usage: nvcc_filter.py <nvcc_path> [args...]", file=sys.stderr)
        sys.exit(1)

    nvcc = sys.argv[1]
    args = sys.argv[2:]

    # Dump all args for first invocation
    global _log_count
    if _log_count == 0:
        print(f"[nvcc_filter] nvcc: {nvcc}", file=sys.stderr)
        print(f"[nvcc_filter] args ({len(args)}): {args[:20]}", file=sys.stderr)

    filtered_args = []
    temp_files = []
    direct_filtered = []

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
        if should_filter_arg(arg):
            direct_filtered.append(arg)
        else:
            filtered_args.append(arg)

        i += 1

    if direct_filtered and _log_count <= _MAX_VERBOSE_LOGS:
        print(f"[nvcc_filter] Direct args filtered: {direct_filtered}", file=sys.stderr)

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
