#!/usr/bin/env python3
r"""NVCC wrapper that filters out MSVC -Fd/-FS flags from response files and arguments.

CMake + Ninja on Windows generates -Xcompiler=-Fd<path>\,-FS in CUDA response files.
The backslash-comma causes nvcc to misparse this as two arguments, with -FS interpreted
as an input file, triggering: "A single input file is required".

This wrapper:
1. Intercepts --options-file arguments
2. Reads and filters the response file content
3. When ccache/sccache is enabled, expands response files into direct args
   (ccache can't parse --options-file; sccache's nvcc handler breaks tmpxft_* temp
   file handling on Windows when wrapping nvcc directly)
4. When no cache tool is used, writes a cleaned response file
5. Passes all other args through, filtering direct -Fd/-FS flags
6. Invokes the real nvcc (optionally via ccache/sccache) with cleaned arguments

Usage (via CMAKE_CUDA_COMPILER_LAUNCHER):
    python nvcc_filter.py [--ccache=<path>] [--sccache=<path>] nvcc.exe [args...]
"""

import sys
import os
import re
import subprocess
import tempfile

_VERBOSE_ENV = "DL4J_NVCC_FILTER_VERBOSE"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def verbose_enabled():
    """Return whether per-invocation NVCC diagnostics were explicitly requested."""
    return os.environ.get(_VERBOSE_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


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
    # Filter ALL bare MSVC /Flag args that leak from CMAKE_CXX_FLAGS into CUDA
    # response files. nvcc uses -dash prefix for its own flags; any /Flag is an MSVC
    # flag that nvcc misinterprets as an input file path, causing:
    #   "A single input file is required for a non-link phase"
    # Note: -Xcompiler=/Flag args are valid (forwarded to host compiler) and kept.
    if re.match(r'^/[A-Za-z]', arg):
        return True
    # Filter GCC flags that are not valid for nvcc
    if arg.startswith('-ftemplate-depth='):
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
    if not os.path.isfile(rsp_path):
        return [], False

    with open(rsp_path, 'rb') as f:
        raw_content = f.read()

    content = raw_content.decode('utf-8', errors='replace')
    lines = content.splitlines()

    if verbose_enabled():
        print(f"[nvcc_filter] === Response file: {rsp_path} ===", file=sys.stderr)
        print(f"[nvcc_filter] Line count: {len(lines)}", file=sys.stderr)
        for i, line in enumerate(lines[:30]):
            print(f"[nvcc_filter]   L{i}: {line[:2000]}", file=sys.stderr)
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

        # Split line into tokens (respecting quotes, stripping them)
        # Quotes are shell-level artifacts; when expanding args for direct
        # subprocess invocation, they must be removed to avoid literal quotes
        # confusing nvcc (e.g., -I"path" -> -Ipath as a direct arg).
        tokens = []
        current = []
        in_quote = False
        for ch in line:
            if ch == '"':
                in_quote = not in_quote
                # Don't append the quote character — strip it
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

    if verbose_enabled():
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

    # Parse optional --ccache=<path> or --sccache=<path> to chain a cache tool with nvcc.
    cache_path = None
    cache_tool = None  # "ccache" or "sccache"
    arg_start = 1
    while arg_start < len(sys.argv):
        if sys.argv[arg_start].startswith('--ccache='):
            cache_path = sys.argv[arg_start][len('--ccache='):]
            cache_tool = "ccache"
            arg_start += 1
        elif sys.argv[arg_start].startswith('--sccache='):
            cache_path = sys.argv[arg_start][len('--sccache='):]
            cache_tool = "sccache"
            arg_start += 1
        else:
            break

    if cache_path and not os.path.isfile(cache_path):
        print(f"[nvcc_filter] WARNING: {cache_tool} not found at {cache_path}, proceeding without caching", file=sys.stderr)
        cache_path = None
        cache_tool = None

    if arg_start >= len(sys.argv):
        print("Usage: nvcc_filter.py [--ccache=<path>] [--sccache=<path>] <nvcc_path> [args...]", file=sys.stderr)
        sys.exit(1)

    nvcc = sys.argv[arg_start]
    args = sys.argv[arg_start + 1:]

    if verbose_enabled():
        print(f"[nvcc_filter] nvcc: {nvcc}", file=sys.stderr)
        if cache_path:
            print(f"[nvcc_filter] {cache_tool}: {cache_path}", file=sys.stderr)
        print(f"[nvcc_filter] args ({len(args)}): {args[:20]}", file=sys.stderr)

    filtered_args = []
    temp_files = []
    direct_filtered = []

    # When a cache tool is enabled, we EXPAND response files into direct args.
    # ccache can't parse nvcc's --options-file format; sccache's direct nvcc
    # wrapping breaks tmpxft_* temp files on Windows. Expanding gives the cache
    # tool direct visibility into args and avoids the temp file issue.
    use_expanded = cache_path is not None

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

    if direct_filtered and verbose_enabled():
        print(f"[nvcc_filter] Direct args filtered: {direct_filtered}", file=sys.stderr)

    if use_expanded and verbose_enabled():
        print(f"[nvcc_filter] Expanded response files for {cache_tool} ({len(filtered_args)} args)", file=sys.stderr)

    try:
        cmd = [nvcc] + filtered_args
        if cache_path:
            cmd = [cache_path] + cmd
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[nvcc_filter] === NVCC FAILED (exit {result.returncode}) ===", file=sys.stderr)
            print(f"[nvcc_filter] Full command ({len(cmd)} args):", file=sys.stderr)
            for ci, ca in enumerate(cmd):
                print(f"[nvcc_filter]   [{ci}] {ca}", file=sys.stderr)
            if cache_path:
                print(f"[nvcc_filter] {cache_tool} was: {cache_path}", file=sys.stderr)
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
