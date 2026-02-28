#!/usr/bin/env python3
r"""NVCC wrapper that filters out MSVC -Fd/-FS flags from response files and arguments.

CMake + Ninja on Windows generates -Xcompiler=-Fd<path>\,-FS in CUDA response files.
The backslash-comma causes nvcc to misparse this as two arguments, with -FS interpreted
as an input file, triggering: "A single input file is required".

This wrapper:
1. Intercepts --options-file arguments
2. Reads and filters the response file content
3. When ccache is enabled, expands response files into direct args (ccache can't parse --options-file)
4. When ccache is disabled, writes a cleaned response file
5. Passes all other args through, filtering direct -Fd/-FS flags
6. Invokes the real nvcc (optionally via ccache) with cleaned arguments

Usage (via CMAKE_CUDA_COMPILER_LAUNCHER):
    python nvcc_filter.py [--ccache=<path>] nvcc.exe [args...]
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


def fix_xcompiler_quoting(arg):
    """Fix -Xcompiler args with quoted space-separated values.

    CMake generates -Xcompiler=" /GR /EHsc" which breaks in Ninja .rsp files
    because the space splits it into multiple tokens. Convert to comma-separated:
    -Xcompiler=" /GR /EHsc" -> -Xcompiler=/GR,/EHsc
    """
    # Match -Xcompiler= followed by a quoted value with spaces
    m = re.match(r'^(-Xcompiler[=:])["\']\s*(.*?)\s*["\']$', arg)
    if m:
        prefix = m.group(1)
        value = m.group(2).strip()
        # Replace spaces with commas
        fixed = prefix + value.replace(' ', ',')
        return fixed
    return arg


def parse_response_file(rsp_path):
    """Read a response file and return filtered tokens.

    Returns (filtered_tokens, was_modified) tuple.
    """
    global _log_count

    if not os.path.isfile(rsp_path):
        return [], False

    with open(rsp_path, 'rb') as f:
        raw_content = f.read()

    content = raw_content.decode('utf-8', errors='replace')
    lines = content.splitlines()

    _log_count += 1
    if _log_count <= _MAX_VERBOSE_LOGS:
        print(f"[nvcc_filter] === Response file: {rsp_path} ===", file=sys.stderr)
        print(f"[nvcc_filter] Line count: {len(lines)}", file=sys.stderr)
        for i, line in enumerate(lines[:30]):
            print(f"[nvcc_filter]   L{i}: {line[:300]}", file=sys.stderr)
        if len(lines) > 30:
            print(f"[nvcc_filter]   ... ({len(lines) - 30} more lines)", file=sys.stderr)

    all_tokens = []
    removed_count = 0
    modified = False

    for line in lines:
        # Fix -Xcompiler quoting at the raw line level
        orig_line = line
        line = re.sub(
            r'-Xcompiler[=:]"[ ]*([^"]*)"',
            lambda m: '-Xcompiler=' + m.group(1).strip().replace(' ', ','),
            line
        )
        if line != orig_line:
            modified = True

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

        for t in tokens:
            if should_filter_arg(t):
                removed_count += 1
            else:
                all_tokens.append(fix_xcompiler_quoting(t))

    if _log_count <= _MAX_VERBOSE_LOGS:
        if removed_count:
            print(f"[nvcc_filter] Filtered {removed_count} tokens from {rsp_path}", file=sys.stderr)
        if modified:
            print(f"[nvcc_filter] Fixed -Xcompiler quoting in {rsp_path}", file=sys.stderr)

    return all_tokens, (removed_count > 0 or modified)


def filter_response_file(rsp_path):
    """Read a response file, filter problematic flags, write cleaned version.

    Returns path to the cleaned response file (may be a new temp file).
    Used when ccache is NOT enabled (response files are kept as-is).
    """
    tokens, was_modified = parse_response_file(rsp_path)

    if was_modified:
        # Detect original line ending style
        with open(rsp_path, 'rb') as f:
            raw = f.read()
        line_ending = '\r\n' if b'\r\n' in raw else '\n'

        new_content = ' '.join(tokens) + line_ending
        rsp_dir = os.path.dirname(rsp_path) or '.'
        fd, tmp_path = tempfile.mkstemp(suffix='.rsp', dir=rsp_dir, prefix='nvcc_filtered_')
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(new_content)
        return tmp_path
    else:
        return rsp_path


def main():
    if len(sys.argv) < 2:
        print("Usage: nvcc_filter.py [--ccache=<path>] <nvcc_path> [args...]", file=sys.stderr)
        sys.exit(1)

    # Parse optional --ccache=<path> to chain ccache with nvcc.
    ccache_path = None
    arg_start = 1
    if sys.argv[1].startswith('--ccache='):
        ccache_path = sys.argv[1][len('--ccache='):]
        if not os.path.isfile(ccache_path):
            print(f"[nvcc_filter] WARNING: ccache not found at {ccache_path}, proceeding without caching", file=sys.stderr)
            ccache_path = None
        arg_start = 2

    if arg_start >= len(sys.argv):
        print("Usage: nvcc_filter.py [--ccache=<path>] <nvcc_path> [args...]", file=sys.stderr)
        sys.exit(1)

    nvcc = sys.argv[arg_start]
    args = sys.argv[arg_start + 1:]

    global _log_count
    if _log_count == 0:
        print(f"[nvcc_filter] nvcc: {nvcc}", file=sys.stderr)
        if ccache_path:
            print(f"[nvcc_filter] ccache: {ccache_path}", file=sys.stderr)
        print(f"[nvcc_filter] args ({len(args)}): {args[:20]}", file=sys.stderr)

    filtered_args = []
    temp_files = []
    direct_filtered = []

    # When ccache is enabled, we EXPAND response files into direct args.
    # ccache can't parse nvcc's --options-file format, so it fails to find
    # the source file and flags. Expanding gives ccache direct visibility.
    use_expanded = ccache_path is not None

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle --options-file <path> (two separate args)
        if arg == '--options-file' and i + 1 < len(args):
            rsp_path = args[i + 1]
            if use_expanded:
                # Expand response file into direct args for ccache
                tokens, _ = parse_response_file(rsp_path)
                filtered_args.extend(tokens)
            else:
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
            if use_expanded:
                tokens, _ = parse_response_file(rsp_path)
                filtered_args.extend(tokens)
            else:
                new_rsp = filter_response_file(rsp_path)
                filtered_args.append('--options-file=' + new_rsp)
                if new_rsp != rsp_path:
                    temp_files.append(new_rsp)
            i += 1
            continue

        # Handle @<path> response file syntax
        if arg.startswith('@') and len(arg) > 1:
            rsp_path = arg[1:]
            if use_expanded:
                tokens, _ = parse_response_file(rsp_path)
                filtered_args.extend(tokens)
            else:
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

    if use_expanded and _log_count <= _MAX_VERBOSE_LOGS:
        print(f"[nvcc_filter] Expanded response files for ccache ({len(filtered_args)} args)", file=sys.stderr)

    try:
        cmd = [nvcc] + filtered_args
        if ccache_path:
            cmd = [ccache_path] + cmd
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[nvcc_filter] === NVCC FAILED (exit {result.returncode}) ===", file=sys.stderr)
            print(f"[nvcc_filter] Command ({len(cmd)} args): {' '.join(cmd[:15])}{'...' if len(cmd) > 15 else ''}", file=sys.stderr)
            if ccache_path:
                print(f"[nvcc_filter] ccache was: {ccache_path}", file=sys.stderr)
            for tf in temp_files:
                if os.path.isfile(tf):
                    with open(tf, 'r', encoding='utf-8', errors='replace') as f:
                        rsp_content = f.read()
                    print(f"[nvcc_filter] Filtered RSP ({tf}):", file=sys.stderr)
                    for i, line in enumerate(rsp_content.splitlines()[:50]):
                        print(f"[nvcc_filter]   {i}: {line[:300]}", file=sys.stderr)
            print(f"[nvcc_filter] === END FAILURE DUMP ===\n", file=sys.stderr)
        sys.exit(result.returncode)
    finally:
        for tf in temp_files:
            try:
                os.unlink(tf)
            except OSError:
                pass


if __name__ == '__main__':
    main()
