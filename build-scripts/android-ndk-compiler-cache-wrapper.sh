#!/usr/bin/env bash
# Route Android NDK object compilation through sccache/ccache while keeping links
# on the real NDK compiler. Source-bearing compile-and-link commands are split.
set -euo pipefail

: "${DL4J_COMPILER_CACHE:?Set DL4J_COMPILER_CACHE to sccache or ccache}"
: "${DL4J_ANDROID_REAL_COMPILER:?Set DL4J_ANDROID_REAL_COMPILER to the NDK compiler}"
[[ -x "${DL4J_COMPILER_CACHE}" ]] || {
    echo "Android compiler cache is not executable: ${DL4J_COMPILER_CACHE}" >&2
    exit 2
}
[[ -x "${DL4J_ANDROID_REAL_COMPILER}" ]] || {
    echo "Android NDK compiler is not executable: ${DL4J_ANDROID_REAL_COMPILER}" >&2
    exit 2
}

args=("$@")
source_indexes=()
sources=()
compile_only=false
non_link_output=false
output=""
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        -c) compile_only=true ;;
        -E|-S|-M|-MM) non_link_output=true ;;
        -o)
            if ((i + 1 < ${#args[@]})); then
                output="${args[$((i + 1))]}"
            fi
            ;;
        -o?*) output="${args[$i]#-o}" ;;
        *.c|*.cc|*.cpp|*.cxx|*.C|*.m|*.mm|*.i|*.ii|*.s|*.S)
            source_indexes+=("$i")
            sources+=("${args[$i]}")
            ;;
    esac
done

record_cached_compile() {
    if [[ -n "${DL4J_ANDROID_CACHE_ATTESTATION:-}" ]]; then
        mkdir -p "$(dirname "${DL4J_ANDROID_CACHE_ATTESTATION}")"
        printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DL4J_ANDROID_REAL_COMPILER}" \
            >> "${DL4J_ANDROID_CACHE_ATTESTATION}"
    fi
}

if [[ "${compile_only}" == true ]]; then
    record_cached_compile
    exec "${DL4J_COMPILER_CACHE}" "${DL4J_ANDROID_REAL_COMPILER}" "$@"
fi
if [[ "${non_link_output}" == true || "${#sources[@]}" -eq 0 ]]; then
    exec "${DL4J_ANDROID_REAL_COMPILER}" "$@"
fi

is_source_index() {
    local candidate="$1"
    local source_index=""
    for source_index in "${source_indexes[@]}"; do
        [[ "${candidate}" == "${source_index}" ]] && return 0
    done
    return 1
}

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/dl4j-android-cache.XXXXXX")"
cleanup_objects() {
    rm -rf -- "${temporary_dir}"
}
trap cleanup_objects EXIT

temporary_objects=()
for ((source_number = 0; source_number < ${#sources[@]}; source_number++)); do
    source_file="${sources[$source_number]}"
    object_file="${temporary_dir}/source-${source_number}.o"
    temporary_objects+=("${object_file}")
    compile_args=()
    for ((i = 0; i < ${#args[@]}; i++)); do
        if is_source_index "$i"; then
            continue
        fi
        case "${args[$i]}" in
            -o|-Xlinker|-Wl|-L|-l|-u|-T|-z)
                ((i += 1))
                ;;
            -o?*|-Wl,*|-Xlinker=*|-L*|-l*|-static|-shared|-pie|-rdynamic|-s|-nostdlib|-nostartfiles|-nodefaultlibs)
                ;;
            *.o|*.obj|*.a|*.so|*.so.*|*.dylib|*.dll|*.lib)
                ;;
            *)
                compile_args+=("${args[$i]}")
                ;;
        esac
    done
    record_cached_compile
    "${DL4J_COMPILER_CACHE}" "${DL4J_ANDROID_REAL_COMPILER}" \
        "${compile_args[@]}" -c "${source_file}" -o "${object_file}"
done

rewritten=()
source_number=0
for ((i = 0; i < ${#args[@]}; i++)); do
    if is_source_index "$i"; then
        rewritten+=("${temporary_objects[$source_number]}")
        source_number=$((source_number + 1))
    else
        rewritten+=("${args[$i]}")
    fi
done
"${DL4J_ANDROID_REAL_COMPILER}" "${rewritten[@]}"
