#!/usr/bin/env python3
r"""Smart ccache wrapper - cross-platform (Windows + Linux/macOS).

Replaces smart_ccache.sh for environments where bash is not available via
CreateProcess (e.g. MSYS2/MinGW CMake on Windows).

Strategy:
- If file is NEW (not in hash cache): let ccache use content-based caching (may hit!)
- If file is KNOWN and UNCHANGED: let ccache use content-based caching (will hit)
- If file is KNOWN and CHANGED: force recompile with CCACHE_RECACHE=1

Also expands nvcc --options-file response files inline so ccache can see the flags.

Usage (via CMAKE_C/CXX/CUDA_COMPILER_LAUNCHER):
    python smart_ccache.py --ccache=<path> --hash-dir=<dir> --build-dir=<dir>
        [--debug] [--] <compiler> [args...]
"""

import sys
import os
import re
import subprocess
import zlib

# Portable file locking
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None


def _lock_file(f, exclusive=True):
    """Acquire a file lock (shared or exclusive), cross-platform."""
    if _fcntl is not None:
        op = _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH
        _fcntl.flock(f.fileno(), op)
    elif _msvcrt is not None:
        import struct
        # Lock 1 byte at position 0
        if exclusive:
            _msvcrt.locking(f.fileno(), _msvcrt.LK_LOCK, 1)
        else:
            # msvcrt doesn't have shared locks; use exclusive
            _msvcrt.locking(f.fileno(), _msvcrt.LK_LOCK, 1)


def _unlock_file(f):
    """Release a file lock, cross-platform."""
    if _fcntl is not None:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
    elif _msvcrt is not None:
        try:
            _msvcrt.locking(f.fileno(), _msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


_SOURCE_EXTENSIONS = {'.cpp', '.c', '.cc', '.cxx', '.cu'}
_MAX_VERBOSE_LOGS = 2
_log_count = 0

# Detect if we're on Windows (MSYS2 environment)
_IS_WINDOWS = sys.platform == 'win32' or os.name == 'nt'


def msys_to_windows_path(path):
    """Convert MSYS2-style paths (/c/...) to Windows paths (C:/...).

    When MSYS sh.exe invokes a native Windows Python, it translates some
    argument paths but not all. This ensures any remaining MSYS paths are
    converted to Windows format that native Windows tools can resolve.
    """
    if not _IS_WINDOWS or not path:
        return path
    # /c/... -> C:/...
    m = re.match(r'^/([a-zA-Z])(/.*)', path)
    if m:
        return m.group(1).upper() + ':' + m.group(2)
    # /mingw64/... -> C:/msys64/mingw64/... (common MSYS2 prefix)
    if path.startswith('/mingw64/') or path.startswith('/mingw32/'):
        msys_root = os.environ.get('MSYSTEM_PREFIX', 'C:/msys64' + path[:path.index('/', 1)])
        # Try to find the MSYS root
        for root in ['C:/msys64', 'D:/msys64', os.environ.get('MSYS2_ROOT', '')]:
            if root and os.path.isdir(root + path):
                return root + path
        return path  # Can't resolve, return as-is
    return path


def compute_file_hash(path):
    """Fast content hash using CRC32 + file size (mirrors cksum behavior)."""
    try:
        data = open(path, 'rb').read()
        return f"{zlib.crc32(data) & 0xffffffff} {len(data)}"
    except OSError:
        return None


def extract_cuda_arch(args):
    """Extract CUDA compute capabilities from compiler arguments."""
    arch_list = []
    prev_arg = ""
    for arg in args:
        # -arch=sm_XX or --gpu-architecture=sm_XX
        m = re.match(r'^(?:-arch|--gpu-architecture)=sm_(\d+)', arg)
        if m:
            sm = m.group(1)
            if sm not in arch_list:
                arch_list.append(sm)

        # -gencode arch=compute_XX,code=sm_XX
        m = re.search(r'code=sm_(\d+)', arg)
        if m:
            sm = m.group(1)
            if sm not in arch_list:
                arch_list.append(sm)

        # Separated form: -arch sm_XX
        if prev_arg == '-arch':
            m = re.match(r'^sm_(\d+)', arg)
            if m:
                sm = m.group(1)
                if sm not in arch_list:
                    arch_list.append(sm)

        prev_arg = arg

    return '_'.join(arch_list) if arch_list else ''


def find_source_file(args):
    """Find the source file in the argument list."""
    for arg in args:
        ext = os.path.splitext(arg)[1].lower()
        if ext in _SOURCE_EXTENSIONS:
            # Try as-is
            if os.path.isfile(arg):
                return os.path.abspath(arg)
            # Try with MSYS path conversion
            converted = msys_to_windows_path(arg)
            if converted != arg and os.path.isfile(converted):
                return os.path.abspath(converted)
            # Try relative to cwd
            full = os.path.join(os.getcwd(), arg)
            if os.path.isfile(full):
                return os.path.abspath(full)
    return None


def read_hash_cache(cache_path, lock_path):
    """Read the hash cache file, returning a dict of {filepath: hash}."""
    cache = {}
    if not os.path.isfile(cache_path):
        return cache
    try:
        with open(lock_path, 'a+') as lf:
            _lock_file(lf, exclusive=False)
            try:
                with open(cache_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        idx = line.find(':')
                        if idx > 0:
                            cache[line[:idx]] = line[idx + 1:]
            finally:
                _unlock_file(lf)
    except OSError:
        pass
    return cache


def update_hash_cache(cache_path, lock_path, filepath, new_hash):
    """Update a single entry in the hash cache file."""
    try:
        with open(lock_path, 'a+') as lf:
            _lock_file(lf, exclusive=True)
            try:
                lines = []
                if os.path.isfile(cache_path):
                    with open(cache_path, 'r', encoding='utf-8', errors='replace') as f:
                        lines = [l for l in f.readlines() if not l.startswith(filepath + ':')]
                lines.append(f"{filepath}:{new_hash}\n")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            finally:
                _unlock_file(lf)
    except OSError as e:
        print(f"[smart_ccache] WARNING: failed to update hash cache: {e}", file=sys.stderr)


def expand_response_file(rsp_path):
    """Read an nvcc response file and return tokens."""
    if not os.path.isfile(rsp_path):
        return []

    with open(rsp_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    tokens = []
    for line in content.splitlines():
        # Fix -Xcompiler quoting: -Xcompiler=" /GR /EHsc" -> -Xcompiler=/GR,/EHsc
        line = re.sub(
            r'-Xcompiler[=:]"[ ]*([^"]*)"',
            lambda m: '-Xcompiler=' + m.group(1).strip().replace(' ', ','),
            line
        )
        # Simple tokenizer respecting quotes (strips quotes — they're
        # shell-level artifacts that must be removed for direct subprocess args)
        current = []
        in_quote = False
        for ch in line:
            if ch == '"':
                in_quote = not in_quote
            elif ch == ' ' and not in_quote:
                if current:
                    tokens.append(''.join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            tokens.append(''.join(current))

    # Filter -Fd/-FS flags (same as nvcc_filter.py)
    filtered = []
    for t in tokens:
        if re.search(r'-Xcompiler.*-Fd', t):
            continue
        if re.search(r'-Xcompiler.*-FS', t):
            continue
        if re.search(r'-Xcompiler.*/Fd', t):
            continue
        if re.search(r'-Xcompiler.*/FS', t):
            continue
        if t in ('-Xcompiler=-FS', '-Xcompiler=/FS', '-FS', '/FS'):
            continue
        if t.startswith('-Fd') or t.startswith('/Fd'):
            continue
        filtered.append(t)

    return filtered


def main():
    # Parse our own flags
    ccache_path = None
    hash_dir = None
    build_dir = None
    debug = False
    compiler_args_start = 1

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--':
            compiler_args_start = i + 1
            break
        elif arg.startswith('--ccache='):
            ccache_path = arg[len('--ccache='):]
        elif arg.startswith('--hash-dir='):
            hash_dir = arg[len('--hash-dir='):]
        elif arg.startswith('--build-dir='):
            build_dir = arg[len('--build-dir='):]
        elif arg == '--debug':
            debug = True
        else:
            compiler_args_start = i
            break
        i += 1

    if not ccache_path or compiler_args_start >= len(sys.argv):
        print("Usage: smart_ccache.py --ccache=<path> --hash-dir=<dir> --build-dir=<dir> "
              "[--debug] [--] <compiler> [args...]", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(ccache_path):
        print(f"[smart_ccache] WARNING: ccache not found at {ccache_path}", file=sys.stderr)
        # Fall through without ccache
        ccache_path = None

    compiler = sys.argv[compiler_args_start]
    args = sys.argv[compiler_args_start + 1:]

    if not hash_dir:
        hash_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ccache_hashes')
    if not build_dir:
        build_dir = os.getcwd()

    lock_path = os.path.join(os.path.dirname(hash_dir), '.ccache_file_hashes.lock')
    os.makedirs(hash_dir, exist_ok=True)

    # Set sloppiness (must match ccache.conf from setup-ccache-windows action)
    os.environ['CCACHE_SLOPPINESS'] = (
        'file_macro,include_file_ctime,include_file_mtime,pch_defines,'
        'time_macros,file_stat_matches,clang_index_store,locale,random_seed'
    )

    # Auto-detect compiler type from compiler basename to help ccache.
    # When ccache is invoked from Python (not MSYS sh.exe), auto-detection
    # may fail since the execution context is different.
    if 'CCACHE_COMPILERTYPE' not in os.environ:
        compiler_base = os.path.basename(compiler).lower()
        if 'g++' in compiler_base or 'gcc' in compiler_base or 'mingw' in compiler_base:
            os.environ['CCACHE_COMPILERTYPE'] = 'gcc'
        elif 'clang' in compiler_base:
            os.environ['CCACHE_COMPILERTYPE'] = 'clang'
        elif 'nvcc' in compiler_base:
            os.environ['CCACHE_COMPILERTYPE'] = 'nvcc'

    # Note: Do NOT set CCACHE_LOGFILE here — parallel compilations writing to the
    # same log file causes contention and deadlocks. Use SD_CCACHE_DEBUG to enable
    # per-invocation stderr logging instead.

    # Determine architecture for cache partitioning
    cuda_arch = extract_cuda_arch(args)

    # Segment/shape keys from environment
    segment_key = re.sub(r'[^A-Za-z0-9_.\-]', '_',
                         os.environ.get('SD_SMART_CCACHE_SEGMENT', 'default'))
    shape_key = re.sub(r'[^A-Za-z0-9_.\-]', '_',
                       os.environ.get('SD_SMART_CCACHE_SHAPE_KEY', 'base'))

    if cuda_arch:
        cache_file = os.path.join(hash_dir, f"{segment_key}__{shape_key}__cuda_sm_{cuda_arch}")
    else:
        cache_file = os.path.join(hash_dir, f"{segment_key}__{shape_key}__default")

    if debug:
        print(f"[smart_ccache] cache_file={cache_file}", file=sys.stderr)
        if cuda_arch:
            print(f"[smart_ccache] CUDA arch: sm_{cuda_arch}", file=sys.stderr)

    # Find source file and check for changes
    source_file = find_source_file(args)
    if source_file:
        # Skip generated files (in build dir)
        is_generated = source_file.startswith(os.path.abspath(build_dir))
        if is_generated:
            if debug:
                print(f"[smart_ccache] Generated file, skip hash check: {source_file}",
                      file=sys.stderr)
        else:
            current_hash = compute_file_hash(source_file)
            if current_hash:
                cache = read_hash_cache(cache_file, lock_path)
                stored_hash = cache.get(source_file, '')

                if stored_hash and current_hash != stored_hash:
                    # File CHANGED — force recompile
                    if debug:
                        print(f"[smart_ccache] CHANGED, forcing recache: {source_file}",
                              file=sys.stderr)
                    os.environ['CCACHE_RECACHE'] = '1'
                    update_hash_cache(cache_file, lock_path, source_file, current_hash)
                elif not stored_hash:
                    # NEW file — let ccache try, record hash
                    if debug:
                        print(f"[smart_ccache] New file: {source_file}", file=sys.stderr)
                    update_hash_cache(cache_file, lock_path, source_file, current_hash)
                else:
                    # Unchanged
                    if debug:
                        print(f"[smart_ccache] Unchanged: {source_file}", file=sys.stderr)

    # Expand response files (for nvcc/ccache compatibility)
    # For GCC @file args: ccache handles these natively, so we pass them through.
    # We only expand --options-file (nvcc-specific) since ccache doesn't understand those.
    expanded_args = []
    j = 0
    while j < len(args):
        arg = args[j]
        if arg == '--options-file' and j + 1 < len(args):
            rsp_path = args[j + 1]
            if _IS_WINDOWS:
                rsp_path = msys_to_windows_path(rsp_path)
            expanded_args.extend(expand_response_file(rsp_path))
            j += 2
            continue
        if arg.startswith('--options-file='):
            rsp_path = arg[len('--options-file='):]
            if _IS_WINDOWS:
                rsp_path = msys_to_windows_path(rsp_path)
            expanded_args.extend(expand_response_file(rsp_path))
            j += 1
            continue
        expanded_args.append(arg)
        j += 1

    # Convert MSYS paths to Windows paths for native Windows tools
    if _IS_WINDOWS:
        ccache_path = msys_to_windows_path(ccache_path) if ccache_path else ccache_path
        compiler = msys_to_windows_path(compiler)
        converted_args = []
        for a in expanded_args:
            # Convert @file response file paths
            if a.startswith('@') and len(a) > 1:
                rsp_path = a[1:]
                converted_rsp = msys_to_windows_path(rsp_path)
                a = '@' + converted_rsp
            # Convert paths in -I, -L, -isystem args
            for prefix in ('-I', '-L', '-isystem'):
                if a.startswith(prefix + '/'):
                    a = prefix + msys_to_windows_path(a[len(prefix):])
                    break
            # Convert standalone paths
            if a.startswith('/') and not a.startswith('//'):
                a = msys_to_windows_path(a)
            converted_args.append(a)
        expanded_args = converted_args

    # Build command
    cmd = []
    if ccache_path:
        cmd.append(ccache_path)
    cmd.append(compiler)
    cmd.extend(expanded_args)

    if debug:
        print(f"[smart_ccache] cmd ({len(cmd)} args): {' '.join(cmd[:10])}...", file=sys.stderr)

    # On first invocation, log key details for diagnostics
    global _log_count
    _log_count += 1
    if _log_count <= _MAX_VERBOSE_LOGS:
        print(f"[smart_ccache] compiler: {compiler}", file=sys.stderr)
        print(f"[smart_ccache] ccache: {ccache_path}", file=sys.stderr)
        print(f"[smart_ccache] cwd: {os.getcwd()}", file=sys.stderr)
        print(f"[smart_ccache] args ({len(expanded_args)}): {expanded_args[:8]}...", file=sys.stderr)
        # Show @file args specifically
        at_files = [a for a in expanded_args if a.startswith('@')]
        if at_files:
            print(f"[smart_ccache] @files: {at_files}", file=sys.stderr)
            for af in at_files:
                rsp = af[1:]
                exists = os.path.isfile(rsp)
                print(f"[smart_ccache]   {af} exists={exists}", file=sys.stderr)
                if exists:
                    try:
                        with open(rsp, 'r') as rf:
                            content = rf.read(500)
                        print(f"[smart_ccache]   content: {content[:200]}", file=sys.stderr)
                    except Exception as e:
                        print(f"[smart_ccache]   read error: {e}", file=sys.stderr)
        # Show source file
        src = [a for a in expanded_args if os.path.splitext(a)[1].lower() in _SOURCE_EXTENSIONS]
        if src:
            print(f"[smart_ccache] source: {src}", file=sys.stderr)
        if _IS_WINDOWS:
            print(f"[smart_ccache] PATH (first 3): {os.environ.get('PATH', '').split(os.pathsep)[:3]}", file=sys.stderr)
            print(f"[smart_ccache] CCACHE_COMPILERTYPE: {os.environ.get('CCACHE_COMPILERTYPE', 'unset')}", file=sys.stderr)

    # Use os.execvp on Unix (replaces this process, like bash exec) or
    # subprocess.run on Windows (execvp doesn't work reliably on Windows).
    if _IS_WINDOWS:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
