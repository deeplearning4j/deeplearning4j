#!/usr/bin/env bash
# Shared, fail-closed Android compiler-cache helpers.
# Source this file from build drivers; do not execute it directly.

if [[ -n "${_DL4J_ANDROID_COMPILER_CACHE_HELPERS_LOADED:-}" ]]; then
    return 0
fi
_DL4J_ANDROID_COMPILER_CACHE_HELPERS_LOADED=1

_DL4J_ANDROID_CACHE_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DL4J_ANDROID_NDK_CACHE_WRAPPER="${_DL4J_ANDROID_CACHE_HELPER_DIR}/android-ndk-compiler-cache-wrapper.sh"

dl4j_resolve_android_compiler_cache() {
    local purpose="${1:-Android builds}"
    local candidate="${DL4J_COMPILER_CACHE:-}"

    if [[ -z "${candidate}" ]]; then
        candidate="$(command -v ccache 2>/dev/null || command -v sccache 2>/dev/null || true)"
    elif [[ "${candidate}" != */* ]]; then
        candidate="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
        echo "${purpose} require DL4J_COMPILER_CACHE, sccache, or ccache" >&2
        return 1
    fi
    case "$(basename "${candidate}")" in
        sccache*|ccache*) ;;
        *)
            echo "${purpose} require an sccache or ccache executable, found ${candidate}" >&2
            return 1
            ;;
    esac
    if ! "${candidate}" --version >/dev/null 2>&1; then
        echo "Android compiler cache is not runnable: ${candidate}" >&2
        return 1
    fi
    printf '%s\n' "${candidate}"
}

dl4j_enable_android_compiler_cache_environment() {
    local purpose="${1:-Android builds}"

    DL4J_COMPILER_CACHE="$(dl4j_resolve_android_compiler_cache "${purpose}")" || return 1
    export DL4J_COMPILER_CACHE
    case "$(basename "${DL4J_COMPILER_CACHE}")" in
        sccache*) export SD_USE_SCCACHE=1 ;;
        ccache*) unset SD_USE_SCCACHE ;;
    esac
    case ":${PATH}:" in
        *":$(dirname "${DL4J_COMPILER_CACHE}"):"*) ;;
        *) export PATH="$(dirname "${DL4J_COMPILER_CACHE}"):${PATH}" ;;
    esac
}

dl4j_android_ndk_host_tag() {
    case "$(uname -s)" in
        Linux) printf '%s\n' linux-x86_64 ;;
        Darwin) printf '%s\n' darwin-x86_64 ;;
        MINGW*|MSYS*|CYGWIN*) printf '%s\n' windows-x86_64 ;;
        *)
            echo "Unsupported Android NDK build host: $(uname -s)" >&2
            return 1
            ;;
    esac
}

dl4j_write_android_compiler_cache_wrapper() {
    local output="${1:?wrapper output path is required}"
    local compiler="${2:?real Android compiler path is required}"

    : "${DL4J_COMPILER_CACHE:?resolve the Android compiler cache first}"
    if [[ ! -x "${compiler}" ]]; then
        echo "Android NDK compiler is not executable: ${compiler}" >&2
        return 1
    fi
    if [[ ! -x "${DL4J_ANDROID_NDK_CACHE_WRAPPER}" ]]; then
        echo "Shared Android compiler-cache wrapper is not executable: ${DL4J_ANDROID_NDK_CACHE_WRAPPER}" >&2
        return 1
    fi
    mkdir -p "$(dirname "${output}")"
    printf '#!/usr/bin/env bash\nexport DL4J_COMPILER_CACHE=%q\nexport DL4J_ANDROID_REAL_COMPILER=%q\nexec %q "$@"\n' \
        "${DL4J_COMPILER_CACHE}" "${compiler}" "${DL4J_ANDROID_NDK_CACHE_WRAPPER}" > "${output}"
    chmod +x "${output}"
}

dl4j_create_android_ndk_cache_overlay() {
    local ndk_root="${1:?Android NDK root is required}"
    local host_tag="${2:?Android NDK host tag is required}"
    local overlay_root="${3:?overlay output path is required}"
    local real_prebuilt="${ndk_root}/toolchains/llvm/prebuilt/${host_tag}"
    local real_bin="${real_prebuilt}/bin"
    local entry=""
    local name=""
    local wrapped=0

    : "${DL4J_COMPILER_CACHE:?resolve the Android compiler cache first}"
    if [[ ! -d "${ndk_root}" || ! -d "${real_bin}" ]]; then
        echo "Android NDK LLVM toolchain is missing under ${ndk_root}" >&2
        return 1
    fi
    if [[ -z "${overlay_root}" || "${overlay_root}" == "/" || "${overlay_root}" == "${ndk_root}" ]]; then
        echo "Unsafe Android NDK cache-overlay path: '${overlay_root}'" >&2
        return 1
    fi

    rm -rf -- "${overlay_root}"
    mkdir -p "${overlay_root}/toolchains/llvm/prebuilt/${host_tag}/bin"

    for entry in "${ndk_root}"/* "${ndk_root}"/.[!.]* "${ndk_root}"/..?*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        [[ "${name}" == toolchains ]] && continue
        ln -s "${entry}" "${overlay_root}/${name}"
    done
    for entry in "${ndk_root}/toolchains"/*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        [[ "${name}" == llvm ]] && continue
        ln -s "${entry}" "${overlay_root}/toolchains/${name}"
    done
    for entry in "${ndk_root}/toolchains/llvm"/*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        [[ "${name}" == prebuilt ]] && continue
        ln -s "${entry}" "${overlay_root}/toolchains/llvm/${name}"
    done
    for entry in "${ndk_root}/toolchains/llvm/prebuilt"/*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        [[ "${name}" == "${host_tag}" ]] && continue
        ln -s "${entry}" "${overlay_root}/toolchains/llvm/prebuilt/${name}"
    done
    for entry in "${real_prebuilt}"/*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        [[ "${name}" == bin ]] && continue
        ln -s "${entry}" "${overlay_root}/toolchains/llvm/prebuilt/${host_tag}/${name}"
    done
    for entry in "${real_bin}"/*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        case "${name}" in
            clang|clang++|clang-[0-9]*|*-linux-android*-clang|*-linux-android*-clang++)
                if [[ -x "${entry}" ]]; then
                    dl4j_write_android_compiler_cache_wrapper \
                        "${overlay_root}/toolchains/llvm/prebuilt/${host_tag}/bin/${name}" \
                        "${entry}"
                    wrapped=$((wrapped + 1))
                else
                    ln -s "${entry}" "${overlay_root}/toolchains/llvm/prebuilt/${host_tag}/bin/${name}"
                fi
                ;;
            *)
                ln -s "${entry}" "${overlay_root}/toolchains/llvm/prebuilt/${host_tag}/bin/${name}"
                ;;
        esac
    done
    if [[ "${wrapped}" -eq 0 ]]; then
        echo "Android NDK cache overlay did not wrap any Clang drivers" >&2
        return 1
    fi
    DL4J_ANDROID_NDK_CACHE_OVERLAY="${overlay_root}"
    export DL4J_ANDROID_NDK_CACHE_OVERLAY
}
