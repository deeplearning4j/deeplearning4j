#!/usr/bin/env bash
# Shared zero-configuration discovery for the Android mobile build entry points.
# This file is sourced; callers retain ownership of shell options and error text.

sdx_android_fail() {
    printf 'android-build-env: %s\n' "$*" >&2
    return 3
}

sdx_android_resolve_executable() {
    local label="$1"
    local explicit="$2"
    shift 2
    local candidate resolved

    if [[ -n "$explicit" ]]; then
        if [[ "$explicit" == */* ]]; then
            [[ -x "$explicit" ]] || sdx_android_fail "$label is not executable: $explicit"
            realpath -e -- "$explicit"
        else
            resolved="$(command -v -- "$explicit" 2>/dev/null || true)"
            [[ -n "$resolved" ]] || sdx_android_fail "$label is not on PATH: $explicit"
            realpath -e -- "$resolved"
        fi
        return
    fi

    for candidate in "$@"; do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        realpath -e -- "$candidate"
        return
    done
    sdx_android_fail "could not discover $label; use an SDX_* override"
}

sdx_android_resolve_ndk() {
    local explicit="${1:-}"
    local sdk_root candidate
    local candidates=()

    for candidate in "$explicit" "${SDX_ANDROID_NDK:-}" \
        "${ANDROID_NDK:-}" "${ANDROID_NDK_ROOT:-}" "${ANDROID_NDK_HOME:-}"; do
        [[ -n "$candidate" ]] || continue
        candidates+=("$candidate")
    done
    for sdk_root in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}" \
        "${HOME:-}/Android/Sdk" "${HOME:-}/Library/Android/sdk" \
        "${HOME:-}/dev-apps/android-sdk"; do
        [[ -n "$sdk_root" ]] || continue
        candidates+=("$sdk_root/ndk/28.1.13356709")
    done

    shopt -s nullglob
    for sdk_root in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}" \
        "${HOME:-}/Android/Sdk" "${HOME:-}/Library/Android/sdk" \
        "${HOME:-}/dev-apps/android-sdk"; do
        [[ -n "$sdk_root" ]] || continue
        for candidate in "$sdk_root"/ndk/28.*; do
            candidates+=("$candidate")
        done
    done
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        [[ -f "$candidate/build/cmake/android.toolchain.cmake" &&
           -s "$candidate/source.properties" ]] || continue
        realpath -e -- "$candidate"
        return
    done
    sdx_android_fail "could not discover Android NDK r28; set SDX_ANDROID_NDK"
}

sdx_android_java_major_matches() {
    local candidate="$1"
    local major="$2"
    local version
    [[ -x "$candidate/bin/java" && -x "$candidate/bin/javac" ]] || return 1
    # JAVA_TOOL_OPTIONS may print a banner before the version line; select the actual version record.
    version="$("$candidate/bin/java" -version 2>&1 | sed -n '/ version "/p' | sed -n '1p')"
    [[ "$version" == *"\"$major."* ]]
}

sdx_android_resolve_java17() {
    local explicit="${1:-}"
    local sdkman_root="${SDKMAN_CANDIDATES_DIR:-${HOME:-}/.sdkman/candidates}/java"
    local candidate
    local candidates=(
        "$explicit"
        "${SDX_JAVA17_HOME:-}"
        "${JAVA_HOME:-}"
        "$sdkman_root/current"
        "${HOME:-}/dev-apps/jdk-17"
    )

    shopt -s nullglob
    candidates+=("$sdkman_root"/17*)
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" && -d "$candidate" ]] || continue
        sdx_android_java_major_matches "$candidate" 17 || continue
        realpath -e -- "$candidate"
        return
    done
    sdx_android_fail "could not discover JDK 17; set SDX_JAVA17_HOME"
}

sdx_android_resolve_maven() {
    local explicit="${1:-}"
    local repo_root="$2"
    local candidates=()
    [[ -z "${MAVEN_HOME:-}" ]] || candidates+=("$MAVEN_HOME/bin/mvn")
    candidates+=(
        "$repo_root/mvnw"
        "${HOME:-}/dev-apps/mvn/bin/mvn"
        "$(command -v mvn 2>/dev/null || true)"
    )
    sdx_android_resolve_executable Maven \
        "${explicit:-${SDX_MAVEN:-${MAVEN_CMD:-${MVN_CMD:-}}}}" \
        "${candidates[@]}"
}

sdx_android_resolve_profile() {
    local profile_dir="$1"
    local requested="${2:-${SDX_ANDROID_PROFILE:-tensor-g3-nnapi}}"
    local candidate

    if [[ "$requested" == */* || "$requested" == *.env ]]; then
        candidate="$requested"
    else
        candidate="$profile_dir/$requested.env"
    fi
    [[ -f "$candidate" && ! -L "$candidate" && -s "$candidate" ]] ||
        sdx_android_fail "Android accelerator profile not found: $requested"
    realpath -e -- "$candidate"
}

sdx_android_default_jobs() {
    local jobs
    jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')"
    if (( jobs > 8 )); then
        printf '8\n'
    else
        printf '%s\n' "$jobs"
    fi
}

sdx_android_default_build_root() {
    printf '%s\n' "${SDX_ANDROID_BUILD_ROOT:-${TMPDIR:-/tmp}/sdx-android-build}"
}
