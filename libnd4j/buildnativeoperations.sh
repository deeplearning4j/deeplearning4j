#!/usr/bin/env bash
#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  * See the NOTICE file distributed with this work for additional
#  * information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#
# PREFLIGHT CHECK OPTIONS:
#   --dry-run              Run preflight checks and CMake configuration, then exit
#                          Validates prerequisites, configuration, and binary size
#
# BUILD MODE OPTIONS:
#   --cmake-only ON        Run CMake configuration only, skip build
#   --link-only ON         Skip compilation, only relink (requires prior build)
#   --triton ON|OFF        Enable/disable Triton GPU compiler integration (default: OFF)
#
# OOM KILLER OPTIONS (cross-platform: Linux, macOS, Windows/MSYS2):
#   --oom-killer ON|OFF              Enable/disable OOM killer (default: ON)
#   --oom-memory-threshold <1-99>    System memory usage % to trigger kill (default: 80)
#   --oom-critical-threshold <1-99>  CRITICAL threshold: immediate SIGKILL, no grace (default: 92)
#   --oom-monitor-interval <seconds> Check interval in seconds (default: 1)
#   --oom-process-max-mb <MB>        Max process tree memory in MB before kill (default: 0=disabled)
#   --oom-velocity-threshold <1-50>  Kill if memory grows faster than this %/5sec (default: 8)
#
#   The OOM killer monitors FOUR conditions (any triggers a kill):
#   0. CRITICAL threshold exceeded - IMMEDIATE SIGKILL, no grace period (prevents OS freeze)
#   1. System memory threshold exceeded (--oom-memory-threshold) - graceful then force kill
#   2. Process tree memory limit exceeded (--oom-process-max-mb) - tracks build + all child compilers
#   3. Rapid memory growth detected (--oom-velocity-threshold) - predictive kill before OOM
#
#   Note: On Windows, uses PowerShell/wmic for memory detection and taskkill for termination
#
# EXAMPLES:
#   # Validate build configuration before building
#   ./buildnativeoperations.sh --dry-run -a cpu -c ON --compiler clang
#
#   # Normal build
#   ./buildnativeoperations.sh -a cpu -c ON --compiler clang -j 14
#
#   # Build with OOM killer enabled (basic - system memory threshold only)
#   ./buildnativeoperations.sh -a cpu -j 8 --oom-killer ON --oom-memory-threshold 85
#
#   # Build with full OOM protection (recommended for large builds)
#   ./buildnativeoperations.sh -a cuda -j 8 --oom-killer ON \
#       --oom-memory-threshold 85 \
#       --oom-process-max-mb 32000 \
#       --oom-velocity-threshold 10
#
set -eu

# cd to the directory containing this script
DIR="$( builtin cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
builtin cd "$DIR"


# Check bash version and set compatibility mode
BASH_MAJOR_VERSION=${BASH_VERSION%%.*}
USE_ASSOCIATIVE_ARRAYS=false

if [[ $BASH_MAJOR_VERSION -ge 4 ]]; then
    USE_ASSOCIATIVE_ARRAYS=true
    echo "✅ Using bash $BASH_VERSION with associative array support"
else
    echo "⚠️  Using bash $BASH_VERSION - falling back to compatibility mode"
fi




# =============================================================================
# TYPE VALIDATION SYSTEM
# =============================================================================


    # Fallback for older bash - use delimited strings
# Format: "key:value|key:value|..."
ALL_SUPPORTED_TYPES_STR="bool:Boolean type|int8:8-bit signed integer|uint8:8-bit unsigned integer|int16:16-bit signed integer|uint16:16-bit unsigned integer|int32:32-bit signed integer|uint32:32-bit unsigned integer|int64:64-bit signed integer|uint64:64-bit unsigned integer|float16:16-bit floating point (half precision)|bfloat16:16-bit brain floating point|float32:32-bit floating point (single precision)|double:64-bit floating point (double precision)|float:Alias for float32|half:Alias for float16|long:Alias for int64|unsignedlong:Alias for uint64|int:Alias for int32|bfloat:Alias for bfloat16|float64:Alias for double|utf8:UTF-8 string type|utf16:UTF-16 string type|utf32:UTF-32 string type"

TYPE_ALIASES_STR="float:float32|half:float16|long:int64|unsignedlong:uint64|int:int32|bfloat:bfloat16|float64:double"

DEBUG_TYPE_PROFILES_STR="MINIMAL_INDEXING:float32;double;int32;int64|ESSENTIAL:float32;double;int32;int64;int8;int16|FLOATS_ONLY:float32;double;float16|INTEGERS_ONLY:int8;int16;int32;int64;uint8;uint16;uint32;uint64|SINGLE_PRECISION:float32;int32;int64|DOUBLE_PRECISION:double;int32;int64"

# Bare minimum types required for basic functionality
MINIMUM_REQUIRED_TYPES=("int32" "int64" "float32")

# Essential types for most operations (recommended minimum)
ESSENTIAL_TYPES=("int32" "int64" "float32" "double")

# =============================================================================
# COMPATIBILITY HELPER FUNCTIONS
# =============================================================================

# Function to lookup value in delimited string (fallback for associative arrays)
lookup_in_string() {
    local key="$1"
    local data_string="$2"
    local default_value="${3:-}"

    # Extract value using parameter expansion and grep
    local result
    result=$(echo "$data_string" | grep -o "${key}:[^|]*" | cut -d: -f2- || echo "$default_value")
    echo "$result"
}

# Function to check if key exists in delimited string
key_exists_in_string() {
    local key="$1"
    local data_string="$2"

    echo "$data_string" | grep -q "${key}:" && return 0 || return 1
}

# Function to get all keys from delimited string
get_keys_from_string() {
    local data_string="$1"
    echo "$data_string" | grep -o '[^|]*:' | sed 's/:$//'
}


get_type_description() {
    case "$1" in
        "bool") echo "Boolean type" ;;
        "int8") echo "8-bit signed integer" ;;
        "uint8") echo "8-bit unsigned integer" ;;
        "int16") echo "16-bit signed integer" ;;
        "uint16") echo "16-bit unsigned integer" ;;
        "int32") echo "32-bit signed integer" ;;
        "uint32") echo "32-bit unsigned integer" ;;
        "int64") echo "64-bit signed integer" ;;
        "uint64") echo "64-bit unsigned integer" ;;
        "float16") echo "16-bit floating point (half precision)" ;;
        "bfloat16") echo "16-bit brain floating point" ;;
        "float32") echo "32-bit floating point (single precision)" ;;
        "double") echo "64-bit floating point (double precision)" ;;
        "float") echo "Alias for float32" ;;
        "half") echo "Alias for float16" ;;
        "long") echo "Alias for int64" ;;
        "unsignedlong") echo "Alias for uint64" ;;
        "int") echo "Alias for int32" ;;
        "bfloat") echo "Alias for bfloat16" ;;
        "float64") echo "Alias for double" ;;
        "utf8") echo "UTF-8 string type" ;;
        "utf16") echo "UTF-16 string type" ;;
        "utf32") echo "UTF-32 string type" ;;
        *) echo "Unknown type" ;;
    esac
}

is_supported_type() {
    local normalized_type
    normalized_type=$(normalize_type "$1")
    case "$normalized_type" in
        bool|int8|uint8|int16|uint16|int32|uint32|int64|uint64|float16|bfloat16|float32|double|utf8|utf16|utf32)
            return 0 ;;
        *)
            return 1 ;;
    esac
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to print colored output with status indicators
print_colored() {
    local color="$1"
    local message="$2"

    # Check if stdout is a terminal and colors are supported
    if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
        # Terminal supports colors
        case "$color" in
            "red")    echo -e "\033[1;31m❌ $message\033[0m" ;;
            "green")  echo -e "\033[1;32m✅ $message\033[0m" ;;
            "yellow") echo -e "\033[1;33m⚠️  $message\033[0m" ;;
            "blue")   echo -e "\033[1;34mℹ️  $message\033[0m" ;;
            "cyan")   echo -e "\033[1;36m🔍 $message\033[0m" ;;
            "magenta"|"purple") echo -e "\033[1;35m🔧 $message\033[0m" ;;
            "white")  echo -e "\033[1;37m💬 $message\033[0m" ;;
            "gray"|"grey") echo -e "\033[1;90m📝 $message\033[0m" ;;
            "bold")   echo -e "\033[1m$message\033[0m" ;;
            *)        echo "$message" ;;
        esac
    else
        # No color support or not a terminal - use plain text with indicators
        case "$color" in
            "red")    echo "❌ ERROR: $message" ;;
            "green")  echo "✅ SUCCESS: $message" ;;
            "yellow") echo "⚠️  WARNING: $message" ;;
            "blue")   echo "ℹ️  INFO: $message" ;;
            "cyan")   echo "🔍 DEBUG: $message" ;;
            "magenta"|"purple") echo "🔧 CONFIG: $message" ;;
            "white")  echo "💬 MESSAGE: $message" ;;
            "gray"|"grey") echo "📝 NOTE: $message" ;;
            *)        echo "$message" ;;
        esac
    fi
}

# Function to normalize type names (handle aliases)
normalize_type() {
    case "$1" in
        "float") echo "float32" ;;
        "half") echo "float16" ;;
        "long") echo "int64" ;;
        "unsignedlong") echo "uint64" ;;
        "int") echo "int32" ;;
        "bfloat") echo "bfloat16" ;;
        "float64") echo "double" ;;
        *) echo "$1" ;;
    esac
}

get_debug_type_profile() {
    case "$1" in
        "MINIMAL_INDEXING") echo "float32;double;int32;int64" ;;
        "ESSENTIAL") echo "float32;double;int32;int64;int8;int16" ;;
        "FLOATS_ONLY") echo "float32;double;float16" ;;
        "INTEGERS_ONLY") echo "int8;int16;int32;int64;uint8;uint16;uint32;uint64" ;;
        "SINGLE_PRECISION") echo "float32;int32;int64" ;;
        "DOUBLE_PRECISION") echo "double;int32;int64" ;;
        *) echo "float32;double;int32;int64" ;;  # default to MINIMAL_INDEXING
    esac
}


# Function to validate a single type
validate_single_type() {
    local type="$1"
    is_supported_type "$type"
}
# Function to display available types
show_available_types() {
    print_colored "cyan" "\n=== AVAILABLE DATA TYPES ==="
    echo
    print_colored "yellow" "Core Types:"
    for type in bool int8 uint8 int16 uint16 int32 uint32 int64 uint64; do
        printf "  %-12s - %s\n" "$type" "$(get_type_description "$type")"
    done

    echo
    print_colored "yellow" "Floating Point Types:"
    for type in float16 bfloat16 float32 double; do
        printf "  %-12s - %s\n" "$type" "$(get_type_description "$type")"
    done

    echo
    print_colored "yellow" "String Types:"
    for type in utf8 utf16 utf32; do
        printf "  %-12s - %s\n" "$type" "$(get_type_description "$type")"
    done

    echo
    print_colored "yellow" "Type Aliases:"
    for alias in float half long unsignedlong int bfloat float64; do
        printf "  %-12s -> %s\n" "$alias" "$(normalize_type "$alias")"
    done

    echo
    print_colored "yellow" "Debug Type Profiles:"
    for profile in MINIMAL_INDEXING ESSENTIAL FLOATS_ONLY INTEGERS_ONLY SINGLE_PRECISION DOUBLE_PRECISION; do
        printf "  %-16s: %s\n" "$profile" "$(get_debug_type_profile "$profile")"
    done
}


# Function to validate types list with comprehensive checking
validate_types() {
    local types_string="$1"
    local mode="${2:-normal}"  # normal, debug, or strict

    if [[ -z "$types_string" ]]; then
        if [[ "$mode" == "strict" ]]; then
            print_colored "red" "ERROR: No data types specified and strict mode enabled!"
            return 1
        else
            print_colored "yellow" "WARNING: No data types specified, using ALL types"
            return 0
        fi
    fi

    # Handle special keywords
    if [[ "$types_string" == "all" || "$types_string" == "ALL" ]]; then
        print_colored "green" "✓ Using ALL data types"
        return 0
    fi

    # Parse semicolon-separated types
    IFS=';' read -ra TYPES_ARRAY <<< "$types_string"
    local invalid_types=()
    local valid_types=()
    local normalized_types=()

    print_colored "blue" "\n=== TYPE VALIDATION ==="

    for type in "${TYPES_ARRAY[@]}"; do
        # Trim whitespace
        type=$(echo "$type" | tr -d ' ')

        if [[ -z "$type" ]]; then
            continue
        fi

        if validate_single_type "$type"; then
            local normalized_type
            normalized_type=$(normalize_type "$type")
            valid_types+=("$type")
            normalized_types+=("$normalized_type")

            if [[ "$type" != "$normalized_type" ]]; then
                print_colored "cyan" "  ✓ $type (normalized to: $normalized_type)"
            else
                print_colored "green" "  ✓ $type"
            fi
        else
            invalid_types+=("$type")
            print_colored "red" "  ✗ $type (INVALID)"
        fi
    done

    # Report results
    echo
    if [[ ${#invalid_types[@]} -gt 0 ]]; then
        print_colored "red" "ERROR: Found ${#invalid_types[@]} invalid type(s): ${invalid_types[*]}"
        echo
        show_available_types
        return 1
    fi

    if [[ ${#valid_types[@]} -eq 0 ]]; then
        print_colored "red" "ERROR: No valid types found!"
        show_available_types
        return 1
    fi

    # Check for minimum required types
    local missing_essential=()
    for req_type in "${MINIMUM_REQUIRED_TYPES[@]}"; do
        local found=false
        for norm_type in "${normalized_types[@]}"; do
            if [[ "$norm_type" == "$req_type" ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" == false ]]; then
            missing_essential+=("$req_type")
        fi
    done

    if [[ ${#missing_essential[@]} -gt 0 ]]; then
        print_colored "yellow" "WARNING: Missing recommended essential types: ${missing_essential[*]}"
        print_colored "yellow" "         Array indexing and basic operations may fail at runtime!"

        if [[ "$mode" == "strict" ]]; then
            print_colored "red" "ERROR: Strict mode requires essential types: ${MINIMUM_REQUIRED_TYPES[*]}"
            return 1
        fi
    fi

    # Check for excessive type combinations in debug builds
    if [[ "$mode" == "debug" && ${#normalized_types[@]} -gt 6 ]]; then
        local estimated_combinations=$((${#normalized_types[@]} * ${#normalized_types[@]} * ${#normalized_types[@]}))
        print_colored "yellow" "WARNING: Debug build with ${#normalized_types[@]} types may generate ~$estimated_combinations combinations"
        print_colored "yellow" "         This could result in very large binaries and long compile times!"
        print_colored "yellow" "         Consider using a debug type profile: --debug-type-profile MINIMAL_INDEXING"
    fi

    print_colored "green" "✓ Type validation passed: ${#valid_types[@]} valid types"
    print_colored "cyan" "Selected types: ${normalized_types[*]}"

    return 0
}


# Function to resolve debug type profile
resolve_debug_types() {
    local profile="$1"
    local custom_types="$2"

    if [[ "$profile" == "CUSTOM" ]]; then
        if [[ -n "$custom_types" ]]; then
            # Ensure minimum indexing types are included in custom
            local minimum_types="int32;int64;float32"
            local combined_types="$minimum_types"

            # Add custom types, avoiding duplicates
            IFS=';' read -ra CUSTOM_ARRAY <<< "$custom_types"
            for type in "${CUSTOM_ARRAY[@]}"; do
                if [[ "$combined_types" != *"$type"* ]]; then
                    combined_types="$combined_types;$type"
                fi
            done
            echo "$combined_types"
        else
            print_colored "red" "ERROR: CUSTOM profile specified but no custom types provided!"
            return 1
        fi
    else
        local resolved_types
        resolved_types=$(get_debug_type_profile "$profile")
        if [[ -z "$resolved_types" ]]; then
            print_colored "yellow" "Warning: Unknown debug profile '$profile', using MINIMAL_INDEXING"
            resolved_types=$(get_debug_type_profile "MINIMAL_INDEXING")
        fi
        echo "$resolved_types"
    fi
}
# Function to estimate build impact
estimate_build_impact() {
    local types_string="$1"
    local build_type="${2:-release}"

    if [[ -z "$types_string" || "$types_string" == "all" || "$types_string" == "ALL" ]]; then
        print_colored "cyan" "\n=== BUILD IMPACT ESTIMATION ==="
        print_colored "yellow" "Using ALL types - expect full compilation with all template instantiations"
        return 0
    fi

    IFS=';' read -ra TYPES_ARRAY <<< "$types_string"
    local type_count=${#TYPES_ARRAY[@]}

    if [[ $type_count -gt 0 ]]; then
        local est_2_combinations=$((type_count * type_count))
        local est_3_combinations=$((type_count * type_count * type_count))
        local est_binary_size_mb=$((est_3_combinations * 10 / 27))  # Rough estimate

        print_colored "cyan" "\n=== BUILD IMPACT ESTIMATION ==="
        print_colored "yellow" "Type count: $type_count"
        print_colored "yellow" "Estimated 2-type combinations: $est_2_combinations"
        print_colored "yellow" "Estimated 3-type combinations: $est_3_combinations"
        print_colored "yellow" "Estimated binary size: ~${est_binary_size_mb}MB"

        if [[ "$build_type" == "debug" && $est_3_combinations -gt 125 ]]; then
            print_colored "red" "⚠️  HIGH COMBINATION COUNT WARNING!"
            print_colored "red" "   Debug build with $est_3_combinations combinations may cause:"
            print_colored "red" "   - Binary size >2GB (x86-64 limit exceeded)"
            print_colored "red" "   - Compilation failure due to PLT overflow"
            print_colored "red" "   - Very long build times"
            echo
            print_colored "yellow" "Consider using fewer types for debug builds:"
            print_colored "yellow" "  --debug-type-profile MINIMAL_INDEXING"
            print_colored "yellow" "  --datatypes \"float32;double;int32;int64\""
        elif [[ $est_binary_size_mb -gt 1000 ]]; then
            print_colored "yellow" "⚠️  Large binary size warning: ~${est_binary_size_mb}MB"
        fi
    fi
}

# Function to show type validation summary
show_type_summary() {
    local types_string="$1"
    local build_type="${2:-release}"
    local profile="${3:-}"

    print_colored "cyan" "\n=== TYPE CONFIGURATION SUMMARY ==="

    if [[ -n "$profile" ]]; then
        print_colored "yellow" "Debug Type Profile: $profile"
        if [[ "$profile" != "CUSTOM" ]]; then
            print_colored "cyan" "Profile Types: $(get_debug_type_profile "$profile")"
        fi
    fi

    if [[ -z "$types_string" || "$types_string" == "all" || "$types_string" == "ALL" ]]; then
        print_colored "green" "Type Selection: ALL (default)"
        print_colored "cyan" "Building with all supported data types"
    else
        print_colored "green" "Type Selection: SELECTIVE"
        print_colored "cyan" "Building with types: $types_string"
    fi

    print_colored "yellow" "Build Type: $build_type"
    echo
}


# =============================================================================
# PLATFORM HELPER FUNCTIONS
# =============================================================================

setwindows_msys() {
  if [[ $KERNEL == *"windows"* ]]; then
    export CMAKE_COMMAND="$CMAKE_COMMAND -G \"MSYS Makefiles\""
  fi
}

setandroid_defaults() {
  if [[ -z ${ANDROID_NDK:-} ]]; then
    export ANDROID_NDK=$HOME/Android/android-ndk/
    echo "No ANDROID_NDK variable set. Setting to default of $ANDROID_NDK"
  else
    echo "USING ANDROID NDK $ANDROID_NDK"
  fi

  if [[ -z ${ANDROID_VERSION:-} ]]; then
    export ANDROID_VERSION=21
    echo "No ANDROID_VERSION variable set. Setting to default of $ANDROID_VERSION"
  else
    echo "USING ANDROID VERSION $ANDROID_VERSION"
  fi
}

# =============================================================================
# VARIABLE INITIALIZATION
# =============================================================================

export CMAKE_COMMAND="cmake"
if which cmake3 &> /dev/null; then
    export CMAKE_COMMAND="cmake3"
fi

[[ -z ${MAKEJ:-} ]] && MAKEJ=4

# Add load average limiting to prevent memory exhaustion
# -l flag: only start new job if load average < limit
# Load limit = 75% of available cores (24 for 32 cores)
CPU_COUNT=1
if command -v nproc >/dev/null 2>&1; then
    CPU_COUNT="$(nproc)"
elif [[ "$(uname -s)" == "Darwin" ]] && command -v sysctl >/dev/null 2>&1; then
    CPU_COUNT="$(sysctl -n hw.ncpu 2>/dev/null || echo 1)"
fi

if ! [[ "${CPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${CPU_COUNT}" -lt 1 ]]; then
    CPU_COUNT=1
fi

# Use MAKEJ + 1 as load limit when explicitly set (CI), otherwise 75% of CPU count
if [[ -n "${LIBND4J_LOAD_LIMIT:-}" ]]; then
    LOAD_LIMIT="${LIBND4J_LOAD_LIMIT}"
elif [[ "${MAKEJ}" -gt 1 ]]; then
    # In CI with explicit -j, trust the caller and set load limit = MAKEJ + 1
    LOAD_LIMIT=$((MAKEJ + 1))
else
    LOAD_LIMIT=$((CPU_COUNT * 3 / 4))
fi
if [[ "${LOAD_LIMIT}" -lt 1 ]]; then
    LOAD_LIMIT=1
fi
export LOAD_LIMIT
export MAKE_COMMAND="make -j${MAKEJ} -l${LOAD_LIMIT}"
export MAKE_ARGUMENTS=
echo eval $CMAKE_COMMAND


# Initialize all script variables to prevent unbound variable errors
PARALLEL="${PARALLEL:-true}"
OS="${OS:-}"
CHIP="${CHIP:-}"
BUILD="${BUILD:-}"
COMPUTE="${COMPUTE:-}"
ARCH="${ARCH:-}"
LIBTYPE="${LIBTYPE:-}"
PACKAGING="${PACKAGING:-}"
CHIP_EXTENSION="${CHIP_EXTENSION:-}"
CHIP_VERSION="${CHIP_VERSION:-}"
EXPERIMENTAL="${EXPERIMENTAL:-}"
OPERATIONS="${OPERATIONS:-}"
DATATYPES="${DATATYPES:-}"
CLEAN="${CLEAN:-false}"
CLEAN_ALL="${CLEAN_ALL:-false}"
MINIFIER="${MINIFIER:-false}"
TESTS="${TESTS:-false}"
PRINT_INDICES="${PRINT_INDICES:-OFF}"
VERBOSE="${VERBOSE:-true}"
VERBOSE_ARG="${VERBOSE_ARG:-VERBOSE=1}"
HELPER="${HELPER:-}"
# Multi-helper support: comma-separated list of helpers
HELPERS="${HELPERS:-}"
# BLAS implementation selection: 'openblas', 'mkl', or empty for auto-detect
# When using oneDNN, defaults to auto-detect (prefers MKL if available)
# Set to 'openblas' to force OpenBLAS even with oneDNN
BLAS_IMPL="${BLAS_IMPL:-}"
# Dynamic kernel selection configuration
DYNAMIC_KERNEL_SELECTION="${DYNAMIC_KERNEL_SELECTION:-ON}"
KERNEL_STRATEGY="${KERNEL_STRATEGY:-fastest}"
KERNEL_AUTOTUNING="${KERNEL_AUTOTUNING:-OFF}"
KERNEL_CACHING="${KERNEL_CACHING:-ON}"
HELPER_PRIORITY="${HELPER_PRIORITY:-}"
CHECK_VECTORIZATION="${CHECK_VECTORIZATION:-OFF}"
NAME="${NAME:-}"
OP_OUTPUT_FILE="${OP_OUTPUT_FILE:-include/generated/include_ops.h}"
USE_LTO="${USE_LTO:-}"
# CUDA parallel compilation options (CUDA 11.2+)
CUDA_LTO="${CUDA_LTO:-OFF}"           # Device LTO for better runtime performance
CUDA_THREADS="${CUDA_THREADS:-0}"     # NVCC --threads (0=auto)
CUDA_SPLIT_COMPILE="${CUDA_SPLIT_COMPILE:-0}"  # NVCC --split-compile (0=auto)
SANITIZE="${SANITIZE:-OFF}"
OPTIMIZATION_LEVEL="${OPTIMIZATION_LEVEL:-}"
SANITIZERS="${SANITIZERS:-address,undefined,float-divide-by-zero,float-cast-overflow}"
FUNC_TRACE="${FUNC_TRACE:-OFF}"
LOG_OUTPUT="${LOG_OUTPUT:-none}"
PRINT_MATH="${PRINT_MATH:-OFF}"
KEEP_NVCC="${KEEP_NVCC:-OFF}"
PREPROCESS="${PREPROCESS:-ON}"
CMAKE_ARGUMENTS="${CMAKE_ARGUMENTS:-}"
PTXAS_INFO="${PTXAS_INFO:-OFF}"
BUILD_PPSTEP="${BUILD_PPSTEP:-OFF}"
EXTRACT_INSTANTIATIONS="${EXTRACT_INSTANTIATIONS:-OFF}"
CMAKE_ONLY="${CMAKE_ONLY:-OFF}"
LINK_ONLY="${LINK_ONLY:-OFF}"
COMPILER="${COMPILER:-}"
BUILD_WITH_JAVA="${BUILD_WITH_JAVA:-ON}"


# Type validation specific variables
DEBUG_TYPE_PROFILE="${DEBUG_TYPE_PROFILE:-}"
DEBUG_CUSTOM_TYPES="${DEBUG_CUSTOM_TYPES:-}"
DEBUG_AUTO_REDUCE="${DEBUG_AUTO_REDUCE:-true}"
STRICT_TYPE_VALIDATION="${STRICT_TYPE_VALIDATION:-false}"
SD_TYPES_VALIDATED="${SD_TYPES_VALIDATED:-false}"
SD_VALIDATED_TYPES="${SD_VALIDATED_TYPES:-}"

# MLIR JIT compilation variables
MLIR="${MLIR:-OFF}"
MLIR_VERSION="${MLIR_VERSION:-18}"
MLIR_GPU="${MLIR_GPU:-OFF}"
TRITON="${TRITON:-OFF}"

# Dependency cache - persists built dependencies across 'mvn clean'
DEP_CACHE="${DEP_CACHE:-ON}"
DEP_CACHE_DIR="${DEP_CACHE_DIR:-}"
DEP_CACHE_CLEAR="${DEP_CACHE_CLEAR:-OFF}"
DEP_CACHE_CLEAR_DEP="${DEP_CACHE_CLEAR_DEP:-}"

# Unity build - combines source files for faster compilation
UNITY_BUILD="${UNITY_BUILD:-OFF}"

# OOM Killer configuration
# Monitors memory usage during build and kills processes if threshold is exceeded
OOM_KILLER_ENABLED="${OOM_KILLER_ENABLED:-ON}"
OOM_MEMORY_THRESHOLD="${OOM_MEMORY_THRESHOLD:-80}"  # Percentage of system memory (1-99) - lowered from 85%
OOM_CRITICAL_THRESHOLD="${OOM_CRITICAL_THRESHOLD:-92}"  # Immediate SIGKILL threshold, no grace period
OOM_MONITOR_INTERVAL="${OOM_MONITOR_INTERVAL:-1}"   # Check interval in seconds (1s for fast response)
OOM_PROCESS_MAX_MB="${OOM_PROCESS_MAX_MB:-0}"       # Process tree memory limit in MB (0=disabled)
OOM_VELOCITY_THRESHOLD="${OOM_VELOCITY_THRESHOLD:-8}" # Kill if memory grows faster than this %/sec
OOM_MONITOR_PID=""  # Will hold the PID of the background monitor

# Velocity tracking arrays (simulated with temp file for subshell compatibility)
OOM_VELOCITY_FILE=""

# =============================================================================
# OOM KILLER FUNCTIONS
# =============================================================================

# Detect platform for OOM killer (cached for performance)
OOM_PLATFORM=""
detect_oom_platform() {
    if [[ -n "$OOM_PLATFORM" ]]; then
        echo "$OOM_PLATFORM"
        return
    fi

    local uname_out
    uname_out="$(uname -s 2>/dev/null || echo "Unknown")"

    case "$uname_out" in
        Linux*)
            OOM_PLATFORM="linux"
            ;;
        Darwin*)
            OOM_PLATFORM="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*|Windows_NT*)
            OOM_PLATFORM="windows"
            ;;
        *)
            # Check for Windows environment variables as fallback
            if [[ -n "${WINDIR:-}" ]] || [[ -n "${SystemRoot:-}" ]]; then
                OOM_PLATFORM="windows"
            else
                OOM_PLATFORM="unknown"
            fi
            ;;
    esac

    echo "$OOM_PLATFORM"
}

# Get current memory usage percentage
get_memory_usage_percent() {
    local platform
    platform=$(detect_oom_platform)

    case "$platform" in
        linux)
            # Linux: use /proc/meminfo
            if [[ -f /proc/meminfo ]]; then
                local mem_total mem_available
                mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
                mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
                if [[ -n "$mem_total" && -n "$mem_available" && "$mem_total" -gt 0 ]]; then
                    echo $(( (mem_total - mem_available) * 100 / mem_total ))
                    return
                fi
            fi
            ;;
        macos)
            # macOS: use vm_stat
            if command -v vm_stat &>/dev/null; then
                local pages_free pages_active pages_inactive pages_speculative pages_wired
                pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
                pages_active=$(vm_stat | grep "Pages active" | awk '{print $3}' | tr -d '.')
                pages_inactive=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
                pages_speculative=$(vm_stat | grep "Pages speculative" | awk '{print $4}' | tr -d '.')
                pages_wired=$(vm_stat | grep "Pages wired" | awk '{print $4}' | tr -d '.')
                local total_pages=$((pages_free + pages_active + pages_inactive + pages_speculative + pages_wired))
                local used_pages=$((pages_active + pages_wired))
                if [[ "$total_pages" -gt 0 ]]; then
                    echo $(( used_pages * 100 / total_pages ))
                    return
                fi
            fi
            ;;
        windows)
            # Windows/MSYS2/MinGW/Cygwin: multiple detection methods

            # Method 1: Try PowerShell (most reliable on Windows)
            if command -v powershell.exe &>/dev/null; then
                local ps_result
                ps_result=$(powershell.exe -NoProfile -NonInteractive -Command \
                    '$os = Get-CimInstance Win32_OperatingSystem; [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100)' 2>/dev/null | tr -d '\r\n')
                if [[ "$ps_result" =~ ^[0-9]+$ ]]; then
                    echo "$ps_result"
                    return
                fi
            fi

            # Method 2: Try wmic (older Windows, deprecated but still works)
            if command -v wmic.exe &>/dev/null; then
                local total_mem free_mem
                total_mem=$(wmic.exe OS get TotalVisibleMemorySize /value 2>/dev/null | grep '=' | cut -d'=' -f2 | tr -d '\r\n')
                free_mem=$(wmic.exe OS get FreePhysicalMemory /value 2>/dev/null | grep '=' | cut -d'=' -f2 | tr -d '\r\n')
                if [[ -n "$total_mem" && -n "$free_mem" && "$total_mem" -gt 0 ]]; then
                    echo $(( (total_mem - free_mem) * 100 / total_mem ))
                    return
                fi
            fi

            # Method 3: Try MSYS2/Cygwin /proc/meminfo (some MSYS2 builds support this)
            if [[ -f /proc/meminfo ]]; then
                local mem_total mem_available mem_free
                mem_total=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
                mem_available=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}')
                mem_free=$(grep MemFree /proc/meminfo 2>/dev/null | awk '{print $2}')
                if [[ -n "$mem_total" && "$mem_total" -gt 0 ]]; then
                    if [[ -n "$mem_available" ]]; then
                        echo $(( (mem_total - mem_available) * 100 / mem_total ))
                        return
                    elif [[ -n "$mem_free" ]]; then
                        echo $(( (mem_total - mem_free) * 100 / mem_total ))
                        return
                    fi
                fi
            fi

            # Method 4: Try systeminfo (slowest, last resort)
            if command -v systeminfo.exe &>/dev/null; then
                local sysinfo_output total_phys avail_phys
                sysinfo_output=$(systeminfo.exe 2>/dev/null)
                # Parse "Total Physical Memory" and "Available Physical Memory"
                total_phys=$(echo "$sysinfo_output" | grep -i "Total Physical Memory" | grep -oE '[0-9,]+' | tr -d ',')
                avail_phys=$(echo "$sysinfo_output" | grep -i "Available Physical Memory" | grep -oE '[0-9,]+' | tr -d ',')
                if [[ -n "$total_phys" && -n "$avail_phys" && "$total_phys" -gt 0 ]]; then
                    echo $(( (total_phys - avail_phys) * 100 / total_phys ))
                    return
                fi
            fi
            ;;
    esac

    # Fallback: return 0 (no memory info available)
    echo "0"
}

# Get memory usage in MB for a single process (from /proc/[pid]/status VmRSS)
get_process_memory_mb() {
    local pid="$1"
    local platform
    platform=$(detect_oom_platform)

    case "$platform" in
        linux)
            if [[ -f "/proc/$pid/status" ]]; then
                local vmrss
                vmrss=$(grep VmRSS "/proc/$pid/status" 2>/dev/null | awk '{print $2}')
                if [[ -n "$vmrss" && "$vmrss" =~ ^[0-9]+$ ]]; then
                    # VmRSS is in kB, convert to MB
                    echo $(( vmrss / 1024 ))
                    return
                fi
            fi
            ;;
        macos)
            # macOS: use ps to get RSS (resident set size) in KB
            if command -v ps &>/dev/null; then
                local rss
                rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
                if [[ -n "$rss" && "$rss" =~ ^[0-9]+$ ]]; then
                    echo $(( rss / 1024 ))
                    return
                fi
            fi
            ;;
        windows)
            # Windows: use PowerShell to get WorkingSet
            if command -v powershell.exe &>/dev/null; then
                local ws_bytes
                ws_bytes=$(powershell.exe -NoProfile -NonInteractive -Command \
                    "(Get-Process -Id $pid -ErrorAction SilentlyContinue).WorkingSet64" 2>/dev/null | tr -d '\r\n')
                if [[ -n "$ws_bytes" && "$ws_bytes" =~ ^[0-9]+$ ]]; then
                    echo $(( ws_bytes / 1048576 ))  # bytes to MB
                    return
                fi
            fi
            ;;
    esac

    echo "0"
}

# Get total memory usage in MB for a process tree (process + all descendants)
# Build processes spawn many children (make -> gcc/g++/nvcc/ld).
# We must monitor the ENTIRE process tree to catch memory-hungry compilers/linkers.
get_process_tree_memory_mb() {
    local root_pid="$1"
    local platform
    platform=$(detect_oom_platform)
    local total_mb=0

    case "$platform" in
        linux)
            # Get root process memory
            local root_mem
            root_mem=$(get_process_memory_mb "$root_pid")
            total_mb=$root_mem

            # Find all descendant processes using /proc/[pid]/task/[tid]/children or pgrep
            if command -v pgrep &>/dev/null; then
                # Get all descendants recursively
                local descendants
                descendants=$(pgrep -P "$root_pid" 2>/dev/null || true)
                for child_pid in $descendants; do
                    if [[ -n "$child_pid" ]]; then
                        # Recursively get child tree memory
                        local child_tree_mem
                        child_tree_mem=$(get_process_tree_memory_mb "$child_pid")
                        total_mb=$((total_mb + child_tree_mem))
                    fi
                done
            else
                # Fallback: scan /proc for child processes
                for child_dir in /proc/[0-9]*/; do
                    local child_pid
                    child_pid=$(basename "$child_dir" 2>/dev/null || true)
                    if [[ -f "/proc/$child_pid/stat" ]]; then
                        local ppid
                        ppid=$(awk '{print $4}' "/proc/$child_pid/stat" 2>/dev/null || true)
                        if [[ "$ppid" == "$root_pid" ]]; then
                            local child_mem
                            child_mem=$(get_process_memory_mb "$child_pid")
                            total_mb=$((total_mb + child_mem))
                        fi
                    fi
                done
            fi
            ;;
        macos)
            # macOS: use ps to get all processes in the tree
            local root_mem
            root_mem=$(get_process_memory_mb "$root_pid")
            total_mb=$root_mem

            if command -v pgrep &>/dev/null; then
                local descendants
                descendants=$(pgrep -P "$root_pid" 2>/dev/null || true)
                for child_pid in $descendants; do
                    if [[ -n "$child_pid" ]]; then
                        local child_tree_mem
                        child_tree_mem=$(get_process_tree_memory_mb "$child_pid")
                        total_mb=$((total_mb + child_tree_mem))
                    fi
                done
            fi
            ;;
        windows)
            # Windows: use PowerShell to get process tree memory
            if command -v powershell.exe &>/dev/null; then
                local tree_mem
                tree_mem=$(powershell.exe -NoProfile -NonInteractive -Command "
                    function Get-ProcessTreeMemory {
                        param(\$pid)
                        \$total = 0
                        \$proc = Get-Process -Id \$pid -ErrorAction SilentlyContinue
                        if (\$proc) { \$total = \$proc.WorkingSet64 }
                        \$children = Get-CimInstance Win32_Process | Where-Object { \$_.ParentProcessId -eq \$pid }
                        foreach (\$child in \$children) {
                            \$total += Get-ProcessTreeMemory -pid \$child.ProcessId
                        }
                        return \$total
                    }
                    [math]::Round((Get-ProcessTreeMemory -pid $root_pid) / 1MB)
                " 2>/dev/null | tr -d '\r\n')
                if [[ -n "$tree_mem" && "$tree_mem" =~ ^[0-9]+$ ]]; then
                    echo "$tree_mem"
                    return
                fi
            fi
            # Fallback to single process
            total_mb=$(get_process_memory_mb "$root_pid")
            ;;
    esac

    echo "$total_mb"
}

# Get all descendant PIDs of a process (recursive)
get_descendant_pids() {
    local parent_pid="$1"
    local descendants=""
    local children

    # Get direct children
    children=$(pgrep -P "$parent_pid" 2>/dev/null) || true

    for child in $children; do
        descendants="$descendants $child"
        # Recursively get descendants of each child
        local child_descendants
        child_descendants=$(get_descendant_pids "$child")
        descendants="$descendants $child_descendants"
    done

    echo "$descendants"
}

# Kill a process tree (cross-platform)
# This kills all descendants of the given PID, not just direct children
kill_process_tree() {
    local pid="$1"
    local signal="${2:-TERM}"
    local platform
    platform=$(detect_oom_platform)

    case "$platform" in
        windows)
            # Windows: use taskkill to kill process tree (//T flag handles descendants)
            if command -v taskkill.exe &>/dev/null; then
                case "$signal" in
                    TERM|SIGTERM|15)
                        # Graceful termination - taskkill without /F
                        taskkill.exe //PID "$pid" //T 2>/dev/null || true
                        ;;
                    KILL|SIGKILL|9)
                        # Force kill
                        taskkill.exe //F //PID "$pid" //T 2>/dev/null || true
                        ;;
                    *)
                        taskkill.exe //PID "$pid" //T 2>/dev/null || true
                        ;;
                esac
            else
                # Fallback to standard kill
                kill -"$signal" "$pid" 2>/dev/null || true
            fi
            ;;
        *)
            # Unix-like systems (Linux, macOS)
            # Get all descendants first (children, grandchildren, etc.)
            local descendants
            descendants=$(get_descendant_pids "$pid")

            # Kill descendants first (bottom-up to avoid orphans re-parenting)
            for desc_pid in $descendants; do
                kill -"$signal" "$desc_pid" 2>/dev/null || true
            done

            # Kill the parent process
            kill -"$signal" "$pid" 2>/dev/null || true
            ;;
    esac
}

# Check if a process is running (cross-platform)
is_process_running() {
    local pid="$1"
    local platform
    platform=$(detect_oom_platform)

    case "$platform" in
        windows)
            # Windows: try multiple methods
            if command -v tasklist.exe &>/dev/null; then
                tasklist.exe //FI "PID eq $pid" 2>/dev/null | grep -q "$pid" && return 0
            fi
            # Fallback to kill -0
            kill -0 "$pid" 2>/dev/null && return 0
            return 1
            ;;
        *)
            # Unix-like systems
            kill -0 "$pid" 2>/dev/null && return 0
            return 1
            ;;
    esac
}

# Start the OOM monitor in the background
# This is the critical watchdog that ACTIVELY KILLS the build when memory is exceeded
start_oom_monitor() {
    local threshold="$1"
    local interval="$2"
    local build_pid="$3"
    local process_max_mb="${4:-0}"
    local velocity_threshold="${5:-8}"
    local critical_threshold="${6:-92}"  # NEW: Critical threshold for immediate kill
    local platform
    platform=$(detect_oom_platform)

    # Create temp file for velocity tracking (subshell can't share variables)
    local velocity_file
    velocity_file=$(mktemp /tmp/oom_velocity.XXXXXX 2>/dev/null || echo "/tmp/oom_velocity.$$")
    echo "0 0 0 0 0" > "$velocity_file"  # Initialize with 5 readings

    (
        # Set up signal handlers and cleanup
        cleanup_velocity_file() {
            rm -f "$velocity_file" 2>/dev/null || true
        }

        case "$platform" in
            windows)
                trap 'cleanup_velocity_file; exit 0' TERM INT EXIT
                ;;
            *)
                trap 'cleanup_velocity_file; exit 0' TERM INT HUP QUIT EXIT
                ;;
        esac

        local check_count=0
        local first_check=true

        while true; do
            # Check memory BEFORE sleeping on first iteration
            # to catch OOM conditions immediately when monitor starts
            if [[ "$first_check" == "true" ]]; then
                first_check=false
            else
                sleep "$interval"
            fi

            # Check if build process is still running
            if ! is_process_running "$build_pid"; then
                exit 0
            fi

            check_count=$((check_count + 1))

            # Get current system memory usage
            local mem_usage
            mem_usage=$(get_memory_usage_percent)

            # Handle case where memory detection fails
            if [[ "$mem_usage" == "0" ]]; then
                continue
            fi

            # Get process tree memory if limit is set
            local process_tree_mb=0
            if [[ "$process_max_mb" -gt 0 ]]; then
                process_tree_mb=$(get_process_tree_memory_mb "$build_pid")
            fi

            # Calculate memory velocity (change per second)
            local velocity=0
            if [[ -f "$velocity_file" ]]; then
                # Read last 5 readings
                local readings
                readings=$(cat "$velocity_file" 2>/dev/null || echo "0 0 0 0 0")
                local oldest newest
                oldest=$(echo "$readings" | awk '{print $1}')
                newest=$mem_usage

                # Shift readings and add new one
                echo "$readings $mem_usage" | awk '{print $2, $3, $4, $5, $6}' > "$velocity_file"

                # Calculate velocity over 5 readings (5 seconds at 1s interval)
                if [[ "$oldest" -gt 0 ]]; then
                    velocity=$((newest - oldest))
                    # Divide by 5 to get per-second rate (since we have 5 readings over 5 seconds)
                    # But keep as integer for comparison
                fi
            fi

            # Log periodically (every 30 checks = 30 seconds at 1s interval)
            if [[ $((check_count % 30)) -eq 0 ]]; then
                local status_msg="OOM Monitor: System ${mem_usage}%"
                if [[ "$process_max_mb" -gt 0 ]]; then
                    status_msg="${status_msg}, Process tree ${process_tree_mb}MB/${process_max_mb}MB"
                fi
                if [[ "$velocity" -ne 0 ]]; then
                    status_msg="${status_msg}, Velocity ${velocity}%/5s"
                fi
                echo "$status_msg"
            fi

            local kill_reason=""
            local kill_type=""
            local is_critical=false

            # CHECK 0: CRITICAL threshold - immediate kill, no grace period
            if [[ "$mem_usage" -ge "$critical_threshold" ]]; then
                kill_reason="CRITICAL MEMORY: ${mem_usage}% exceeds CRITICAL threshold of ${critical_threshold}%"
                kill_type="CRITICAL"
                is_critical=true
            # CHECK 1: Process tree memory limit (most specific - kills only if OUR process is the problem)
            elif [[ "$process_max_mb" -gt 0 && "$process_tree_mb" -ge "$process_max_mb" ]]; then
                kill_reason="PROCESS MEMORY LIMIT: Build process tree using ${process_tree_mb}MB exceeds limit of ${process_max_mb}MB"
                kill_type="PROCESS_LIMIT"
            # CHECK 2: Rapid memory growth (predictive kill - memory growing too fast)
            elif [[ "$velocity" -ge "$velocity_threshold" && "$mem_usage" -ge 70 ]]; then
                kill_reason="RAPID MEMORY GROWTH: Memory growing at ${velocity}% per 5 seconds (threshold: ${velocity_threshold}%) at ${mem_usage}% usage"
                kill_type="VELOCITY"
            # CHECK 3: System memory threshold (traditional OOM killer)
            elif [[ "$mem_usage" -ge "$threshold" ]]; then
                kill_reason="SYSTEM MEMORY THRESHOLD: ${mem_usage}% exceeds threshold of ${threshold}%"
                kill_type="THRESHOLD"
            fi

            if [[ -n "$kill_reason" ]]; then
                echo ""
                print_colored "red" "═══════════════════════════════════════════════════════════"
                if [[ "$is_critical" == "true" ]]; then
                    print_colored "red" "🚨🚨🚨 CRITICAL OOM - IMMEDIATE KILL! 🚨🚨🚨"
                else
                    print_colored "red" "🚨 OOM KILLER TRIGGERED! ($kill_type)"
                fi
                print_colored "red" "═══════════════════════════════════════════════════════════"
                print_colored "red" "$kill_reason"
                print_colored "yellow" ""
                print_colored "yellow" "Status at kill time:"
                print_colored "cyan" "  • System memory: ${mem_usage}%"
                print_colored "cyan" "  • Process tree memory: ${process_tree_mb}MB"
                print_colored "cyan" "  • Memory velocity: ${velocity}% per 5 seconds"
                print_colored "cyan" "  • Build PID: $build_pid"
                print_colored "yellow" ""

                if [[ "$is_critical" == "true" ]]; then
                    # Immediate SIGKILL - no time for graceful shutdown
                    print_colored "red" "Sending immediate SIGKILL (no grace period)..."
                    kill_process_tree "$build_pid" "KILL"
                    sleep 0.5
                else
                    # Normal threshold: Try graceful first, but with shorter timeout
                    print_colored "yellow" "Terminating build process tree..."
                    kill_process_tree "$build_pid" "TERM"
                    sleep 1  # Reduced from 2 seconds

                    # Force kill if still running
                    if is_process_running "$build_pid"; then
                        print_colored "yellow" "Process still running, forcing termination..."
                        kill_process_tree "$build_pid" "KILL"
                        sleep 0.5
                    fi
                fi

                # Kill any remaining child processes of the build (safer than blanket pkill)
                # Only kill processes that are descendants of our build_pid
                if [[ "$platform" == "windows" ]]; then
                    # Windows taskkill with /T already kills the process tree, no extra action needed
                    :
                else
                    # Linux/macOS: kill remaining descendants of the build process only
                    if command -v pkill &>/dev/null; then
                        # Use -P to kill only children of build_pid (not all matching processes system-wide)
                        pkill -9 -P "$build_pid" 2>/dev/null || true
                    fi
                fi

                print_colored "yellow" ""
                print_colored "yellow" "SUGGESTIONS TO REDUCE MEMORY USAGE:"
                print_colored "cyan" "  1. Reduce parallel jobs: -j 2 (currently: ${MAKEJ})"
                print_colored "cyan" "  2. Reduce data types: --datatypes \"float32;int32;int64\""
                print_colored "cyan" "  3. Increase swap/page file size"
                print_colored "cyan" "  4. Close other memory-intensive applications"
                print_colored "cyan" "  5. Increase thresholds: --oom-memory-threshold 95 --oom-process-max-mb 16384"
                print_colored "yellow" ""
                print_colored "yellow" "CMAKE BUILD CONFIGURATION OPTIONS:"
                print_colored "cyan" "  6. Reduce type instantiation chunk size:"
                print_colored "cyan" "     Edit cmake/TemplateProcessing.cmake: set(_chunk_target 2)"
                print_colored "cyan" "  7. Reduce CUDA architectures: -DCMAKE_CUDA_ARCHITECTURES=\"86\""
                if [[ "$platform" == "windows" ]]; then
                    print_colored "cyan" "  8. Windows: Check Task Manager for memory-heavy processes"
                    print_colored "cyan" "  9. Windows: Increase virtual memory in System Properties"
                fi
                print_colored "yellow" ""

                cleanup_velocity_file
                exit 137  # Standard OOM exit code
            fi
        done
    ) &

    OOM_MONITOR_PID=$!
    local monitor_info="threshold: ${threshold}%, CRITICAL: ${critical_threshold}%, interval: ${interval}s"
    if [[ "$process_max_mb" -gt 0 ]]; then
        monitor_info="${monitor_info}, process limit: ${process_max_mb}MB"
    fi
    monitor_info="${monitor_info}, velocity: ${velocity_threshold}%/5s"
    print_colored "green" "✓ OOM monitor ACTIVE (PID: $OOM_MONITOR_PID, $monitor_info, platform: $platform)"
    print_colored "yellow" "  ⚠️  Memory thresholds: ${threshold}%=graceful kill, ${critical_threshold}%=IMMEDIATE SIGKILL"
}

# Stop the OOM monitor
stop_oom_monitor() {
    if [[ -n "$OOM_MONITOR_PID" ]]; then
        if is_process_running "$OOM_MONITOR_PID"; then
            kill_process_tree "$OOM_MONITOR_PID" "TERM"
            # Wait for the monitor to exit (with timeout for Windows compatibility)
            local wait_count=0
            while is_process_running "$OOM_MONITOR_PID" && [[ $wait_count -lt 10 ]]; do
                sleep 0.1 2>/dev/null || sleep 1
                wait_count=$((wait_count + 1))
            done
            # Force kill if still running
            if is_process_running "$OOM_MONITOR_PID"; then
                kill_process_tree "$OOM_MONITOR_PID" "KILL"
            fi
        fi
        wait "$OOM_MONITOR_PID" 2>/dev/null || true
        OOM_MONITOR_PID=""
    fi
}

# Wrapper to run a command with OOM monitoring
run_with_oom_monitor() {
    local cmd="$1"

    # Auto-disable OOM monitor on Windows (see comment near main build section)
    if [[ "$OOM_KILLER_ENABLED" == "ON" ]]; then
        local _oom_platform
        _oom_platform=$(detect_oom_platform)
        if [[ "$_oom_platform" == "windows" ]]; then
            OOM_KILLER_ENABLED="OFF"
        fi
    fi

    if [[ "$OOM_KILLER_ENABLED" == "ON" ]]; then
        # Validate threshold
        if [[ ! "$OOM_MEMORY_THRESHOLD" =~ ^[0-9]+$ ]] || \
           [[ "$OOM_MEMORY_THRESHOLD" -lt 1 ]] || \
           [[ "$OOM_MEMORY_THRESHOLD" -gt 99 ]]; then
            print_colored "red" "❌ ERROR: Invalid OOM memory threshold: $OOM_MEMORY_THRESHOLD"
            print_colored "yellow" "   Must be a number between 1 and 99"
            exit 1
        fi

        # Validate interval
        if [[ ! "$OOM_MONITOR_INTERVAL" =~ ^[0-9]+$ ]] || \
           [[ "$OOM_MONITOR_INTERVAL" -lt 1 ]]; then
            print_colored "red" "❌ ERROR: Invalid OOM monitor interval: $OOM_MONITOR_INTERVAL"
            print_colored "yellow" "   Must be a positive integer (seconds)"
            exit 1
        fi

        print_colored "cyan" "═══════════════════════════════════════════════════════════"
        print_colored "cyan" "🛡️  OOM KILLER ENABLED - ACTIVE MEMORY PROTECTION"
        print_colored "cyan" "═══════════════════════════════════════════════════════════"
        print_colored "blue" "System memory threshold: ${OOM_MEMORY_THRESHOLD}%"
        print_colored "blue" "Monitor interval: ${OOM_MONITOR_INTERVAL}s"
        if [[ "$OOM_PROCESS_MAX_MB" -gt 0 ]]; then
            print_colored "blue" "Process tree limit: ${OOM_PROCESS_MAX_MB}MB"
        else
            print_colored "yellow" "Process tree limit: DISABLED (use --oom-process-max-mb to enable)"
        fi
        print_colored "blue" "Velocity threshold: ${OOM_VELOCITY_THRESHOLD}% per 5 seconds"

        local current_mem
        current_mem=$(get_memory_usage_percent)
        print_colored "blue" "Current memory usage: ${current_mem}%"
        echo ""

        # Run the build command in background
        eval "$cmd" &
        local build_pid=$!

        # Start the OOM monitor
        start_oom_monitor "$OOM_MEMORY_THRESHOLD" "$OOM_MONITOR_INTERVAL" "$build_pid" "$OOM_PROCESS_MAX_MB" "$OOM_VELOCITY_THRESHOLD" "$OOM_CRITICAL_THRESHOLD"

        # Wait for build to complete
        wait "$build_pid"
        local exit_code=$?

        # Stop the OOM monitor
        stop_oom_monitor

        return $exit_code
    else
        # No OOM monitoring, run command directly
        eval "$cmd"
        return $?
    fi
}

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]
do
    key="$1"
    value="${2:-}"

    case $key in
         --extract-instantiations)
        EXTRACT_INSTANTIATIONS="$value"
        if [[ "$value" == "ON" ]]; then
            print_colored "blue" "✓ Template instantiation extraction enabled"
            print_colored "yellow" "Build will extract instantiations and exit"
        fi
        shift # past argument
        ;;

        -dt|--datatypes)
            DATATYPES="$value"
            print_colored "green" "✓ Detected datatypes argument: $value"
            shift # past argument
            ;;
        --dataypes|--dataype|--datatype)
            print_colored "red" "❌ ERROR: Invalid argument '$key'"
            print_colored "red" "    Did you mean '--datatypes' or '-dt'?"
            print_colored "yellow" "    Correct usage: --datatypes \"float32;double;int32;int64\""
            exit 1
            ;;
        --debug-type-profile)
            DEBUG_TYPE_PROFILE="$value"
            print_colored "blue" "✓ Detected debug profile: $value"
            shift # past argument
            ;;
        --debug-custom-types)
            DEBUG_CUSTOM_TYPES="$value"
            shift # past argument
            ;;
        --debug-auto-reduce)
            DEBUG_AUTO_REDUCE="$value"
            shift # past argument
            ;;
        --strict-types)
            STRICT_TYPE_VALIDATION="true"
            shift # past argument
            ;;
        --show-types)
            show_available_types
            exit 0
            ;;
        --validate-types)
            if [[ -n "$value" ]]; then
                validate_types "$value" "strict"
                exit $?
            else
                print_colored "red" "ERROR: --validate-types requires a type list"
                exit 1
            fi
            ;;

        # Build configuration arguments
        --generate-flatc)
            export GENERATE_FLATC="ON"
            CMAKE_ARGUMENTS="$CMAKE_ARGUMENTS -DGENERATE_FLATC=ON"
            shift # past argument
            ;;
        -ptxas|--ptxas-info)
            PTXAS_INFO="$value"
            shift # past argument
            ;;
        -ol|--optimization-level)
            OPTIMIZATION_LEVEL="$value"
            shift # past argument
            ;;
        -pi|--print-indices)
            PRINT_INDICES="$value"
            shift # past argument
            ;;
        -pm|--print-math)
            PRINT_MATH="$value"
            shift # past argument
            ;;
        -h|--helper)
            HELPER="$value"
            shift # past argument
            ;;
        --helpers)
            HELPERS="$value"
            print_colored "green" "✓ Multi-helper mode: $value"
            shift # past argument
            ;;
        --dynamic-kernel-selection)
            DYNAMIC_KERNEL_SELECTION="$value"
            shift # past argument
            ;;
        --kernel-strategy)
            KERNEL_STRATEGY="$value"
            print_colored "blue" "✓ Kernel strategy: $value"
            shift # past argument
            ;;
        --kernel-autotuning)
            KERNEL_AUTOTUNING="$value"
            shift # past argument
            ;;
        --kernel-caching)
            KERNEL_CACHING="$value"
            shift # past argument
            ;;
        --helper-priority)
            HELPER_PRIORITY="$value"
            if [ -n "$value" ]; then
                print_colored "blue" "✓ Helper priority: $value"
            fi
            shift # past argument
            ;;
        --blas)
            BLAS_IMPL="$value"
            case "$value" in
                openblas)
                    print_colored "green" "✓ BLAS implementation: OpenBLAS (MKL disabled)"
                    ;;
                mkl)
                    print_colored "green" "✓ BLAS implementation: Intel MKL"
                    ;;
                ""|auto)
                    print_colored "blue" "✓ BLAS implementation: auto-detect"
                    BLAS_IMPL=""
                    ;;
                *)
                    print_colored "red" "❌ ERROR: Invalid BLAS implementation '$value'"
                    print_colored "yellow" "    Valid options: openblas, mkl, auto (or empty)"
                    exit 1
                    ;;
            esac
            shift # past argument
            ;;
        --mlir)
            MLIR="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "green" "✓ MLIR JIT compilation enabled"
            fi
            shift # past argument
            ;;
        --mlir-version)
            MLIR_VERSION="$value"
            print_colored "blue" "✓ MLIR/LLVM version: $value"
            shift # past argument
            ;;
        --mlir-gpu)
            MLIR_GPU="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "green" "✓ MLIR GPU backend enabled"
            fi
            shift # past argument
            ;;
        --triton)
            TRITON="$value"
            case "$value" in
                ON)
                    print_colored "green" "✓ Triton enabled"
                    ;;
                OFF)
                    print_colored "blue" "✓ Triton disabled"
                    ;;
                *)
                    print_colored "red" "❌ ERROR: Invalid Triton value '$value'"
                    print_colored "yellow" "    Valid options: ON, OFF"
                    exit 1
                    ;;
            esac
            shift # past argument
            ;;
        --dep-cache)
            DEP_CACHE="$value"
            shift # past argument
            ;;
        --dep-cache-dir)
            DEP_CACHE_DIR="$value"
            shift # past argument
            ;;
        --dep-cache-clear)
            DEP_CACHE_CLEAR="$value"
            shift # past argument
            ;;
        --dep-cache-clear-dep)
            DEP_CACHE_CLEAR_DEP="$value"
            shift # past argument
            ;;
        -o|-platform|--platform)
            OS="$value"
            shift # past argument
            ;;
        -ft|-functrace|--functrace)
            FUNC_TRACE="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "cyan" "✓ Function tracing enabled"
            fi
            shift # past argument
            ;;
        -b|--build-type)
            BUILD="$value"
            shift # past argument
            ;;
        --compiler|-compiler)
            COMPILER="$value"
            print_colored "blue" "✓ Compiler set to: $value"
            shift # past argument
            ;;
        -p|--packaging)
            PACKAGING="$value"
            shift # past argument
            ;;
        -kno|--keep-nvcc-output)
            KEEP_NVCC="$value"
            shift # past argument
            ;;
        -c|--chip)
            CHIP="$value"
            shift # past argument
            ;;
        -cc|--compute)
            COMPUTE="$value"
            echo COMPUTE="$value"
            shift # past argument
            ;;
        -a|--arch)
            ARCH="$value"
            shift # past argument
            ;;
        -l|--libtype)
            LIBTYPE="$value"
            shift # past argument
            ;;
        -e|--chip-extension)
            CHIP_EXTENSION="$value"
            shift # past argument
            ;;
        -v|--chip-version)
            CHIP_VERSION="$value"
            shift # past argument
            ;;
        -op|--operations)
            OPERATIONS="$value"
            shift # past argument
            ;;
        --use_lto)
            USE_LTO="-DSD_USE_LTO=$value"
            shift # past argument
            ;;
        --cuda-lto)
            CUDA_LTO="$value"
            shift # past argument
            ;;
        --cuda-threads)
            CUDA_THREADS="$value"
            shift # past argument
            ;;
        --cuda-split-compile)
            CUDA_SPLIT_COMPILE="$value"
            shift # past argument
            ;;
        --name)
            NAME="$value"
            shift # past argument
            ;;
        --check-vectorization)
            CHECK_VECTORIZATION="$value"
            shift # past argument
            ;;
        -j)
            MAKEJ="$value"
            shift # past argument
            ;;
        clean)
            CLEAN="true"
            shift # past argument
            ;;
        clean-all)
            CLEAN="true"
            CLEAN_ALL="true"
            shift # past argument
            ;;
        -m|--minifier)
            MINIFIER="true"
            shift # past argument
            ;;
        -t|--tests)
            TESTS="true"
            shift # past argument
            ;;
        -V|--verbose)
            VERBOSE="true"
            shift # past argument
            ;;
        -l|--log-output)
            LOG_OUTPUT="$value"
            shift # past argument
            ;;
        -sa|--sanitize)
            SANITIZE="$value"
            shift # past argument
            ;;
        -sar|--sanitizers)
            SANITIZERS="$value"
            shift # past argument
            ;;
        -of|--op-output-file)
            OP_OUTPUT_FILE="$value"
            shift # past argument
            ;;
          --preprocess)
            PREPROCESS="$value"
            shift # past argument
            ;;
        --ppstep|--build-ppstep)
            BUILD_PPSTEP="$value"
            shift # past argument
            ;;
        --cmake-only|--configure-only)
            CMAKE_ONLY="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "blue" "✓ CMake-only mode enabled - will exit after configuration"
            fi
            shift # past argument
            ;;
        --link-only)
            LINK_ONLY="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "blue" "✓ Link-only mode enabled - will skip compilation and only relink"
            fi
            shift # past argument
            ;;
        --unity-build)
            UNITY_BUILD="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "green" "✓ Unity build enabled - combining source files for faster compilation"
            fi
            shift # past argument
            ;;
        --oom-killer)
            OOM_KILLER_ENABLED="$value"
            if [[ "$value" == "ON" ]]; then
                print_colored "green" "✓ OOM killer enabled - will monitor memory during build"
            fi
            shift # past argument
            ;;
        --oom-memory-threshold)
            OOM_MEMORY_THRESHOLD="$value"
            print_colored "blue" "✓ OOM memory threshold set to: ${value}%"
            shift # past argument
            ;;
        --oom-monitor-interval)
            OOM_MONITOR_INTERVAL="$value"
            print_colored "blue" "✓ OOM monitor interval set to: ${value}s"
            shift # past argument
            ;;
        --oom-process-max-mb)
            OOM_PROCESS_MAX_MB="$value"
            print_colored "blue" "✓ OOM process memory limit set to: ${value}MB"
            shift # past argument
            ;;
        --oom-velocity-threshold)
            OOM_VELOCITY_THRESHOLD="$value"
            print_colored "blue" "✓ OOM velocity threshold set to: ${value}% per 5 seconds"
            shift # past argument
            ;;
        --oom-critical-threshold)
            OOM_CRITICAL_THRESHOLD="$value"
            print_colored "blue" "✓ OOM CRITICAL threshold set to: ${value}% (immediate SIGKILL)"
            shift # past argument
            ;;
        --dry-run)
            # Handle both "--dry-run" (no value) and "--dry-run ON" (with value)
            if [[ "$value" == "ON" ]] || [[ "$value" == "OFF" ]]; then
                DRY_RUN="$value"
                shift # past argument
            else
                DRY_RUN="ON"
            fi
            if [ "$DRY_RUN" == "ON" ]; then
                print_colored "blue" "✓ Dry-run mode enabled - will run preflight checks and CMake, then exit"
            fi
            shift # past argument
            ;;
        --default)
            DEFAULT=YES
            shift # past argument
            ;;
        *)
            # unknown option
            shift # past argument
            ;;
    esac
done

# =============================================================================
# POST-ARGUMENT PROCESSING AND TYPE VALIDATION
# =============================================================================

# Re-export MAKE_COMMAND with the parsed MAKEJ value (was set before arg parsing)
export MAKE_COMMAND="make -j${MAKEJ} -l${LOAD_LIMIT}"

print_colored "blue" "\n🔍 PROCESSING TYPE CONFIGURATION"

USER_EXPLICIT_DATATYPES="$DATATYPES"

# Handle debug builds with auto-reduction
if [[ "$FUNC_TRACE" == "ON" && "$DEBUG_AUTO_REDUCE" == "true" ]]; then
    print_colored "cyan" "=== DEBUG BUILD TYPE REDUCTION ACTIVE ==="

    if [[ -n "$DEBUG_TYPE_PROFILE" ]]; then
        RESOLVED_TYPES=$(resolve_debug_types "$DEBUG_TYPE_PROFILE" "$DEBUG_CUSTOM_TYPES")
        if [[ $? -eq 0 ]]; then
            # FIXED: Only override DATATYPES if it wasn't explicitly set by user
            if [[ -z "$USER_EXPLICIT_DATATYPES" ]]; then
                DATATYPES="$RESOLVED_TYPES"
                print_colored "green" "Auto-applied debug profile: $DEBUG_TYPE_PROFILE"
                print_colored "cyan" "Profile resolved to: $DATATYPES"
            else
                print_colored "yellow" "⚠️  Debug profile specified but datatypes already set"
                print_colored "yellow" "    Using explicit datatypes: $USER_EXPLICIT_DATATYPES"
                print_colored "yellow" "    Ignoring profile: $DEBUG_TYPE_PROFILE"
                DATATYPES="$USER_EXPLICIT_DATATYPES"
            fi
        else
            exit 1
        fi
    elif [[ -z "$USER_EXPLICIT_DATATYPES" ]]; then
        # No types specified and no profile - use minimal safe default
        DATATYPES=$(resolve_debug_types "MINIMAL_INDEXING" "")
        print_colored "yellow" "Auto-selected MINIMAL_INDEXING profile for debug build"
        print_colored "cyan" "Resolved to: $DATATYPES"
    else
        print_colored "green" "Using explicitly specified datatypes for debug build: $USER_EXPLICIT_DATATYPES"
        DATATYPES="$USER_EXPLICIT_DATATYPES"
    fi

    print_colored "cyan" "============================================="
fi


# Determine validation mode
validation_mode="normal"
if [[ "$FUNC_TRACE" == "ON" ]]; then
    validation_mode="debug"
fi
if [[ "$STRICT_TYPE_VALIDATION" == "true" ]]; then
    validation_mode="strict"
fi

# MANDATORY type validation
print_colored "blue" "\n🔍 MANDATORY TYPE VALIDATION"

if [[ -n "$DATATYPES" && "$DATATYPES" != "all" && "$DATATYPES" != "ALL" ]]; then
    print_colored "cyan" "Validating specified datatypes: $DATATYPES"

    if ! validate_types "$DATATYPES" "$validation_mode"; then
        print_colored "red" "\n💥 BUILD HALTED: Type validation failed!"
        print_colored "yellow" "Use --show-types to see available types"
        exit 1
    fi

    # Show build impact and summary
    estimate_build_impact "$DATATYPES" "$BUILD"
    show_type_summary "$DATATYPES" "$BUILD" "$DEBUG_TYPE_PROFILE"

    # Export validated types to CMake
    export SD_TYPES_VALIDATED="true"
    export SD_VALIDATED_TYPES="$DATATYPES"
    DATATYPES_ARG="-DSD_TYPES_LIST=\"$DATATYPES\" -DSD_TYPES_VALIDATED=true"

    print_colored "green" "✓ Passing validated types to CMake: $DATATYPES"

else
    # Building with ALL types
    print_colored "cyan" "No selective datatypes specified - building with ALL types"
    show_type_summary "" "$BUILD" "$DEBUG_TYPE_PROFILE"

    export SD_TYPES_VALIDATED="false"
    export SD_VALIDATED_TYPES=""
    DATATYPES_ARG=""

    print_colored "cyan" "✓ Building with ALL types (default)"
fi

# Ensure CMake uses our validated types
CMAKE_ARGUMENTS="$CMAKE_ARGUMENTS $DATATYPES_ARG"

# Add strict validation mode to CMake if enabled
if [[ "$STRICT_TYPE_VALIDATION" == "true" ]]; then
    CMAKE_ARGUMENTS="$CMAKE_ARGUMENTS -DSD_STRICT_TYPE_VALIDATION=ON"
fi

# Add debug configuration
if [[ "$FUNC_TRACE" == "ON" ]]; then
    CMAKE_ARGUMENTS="$CMAKE_ARGUMENTS -DSD_GCC_FUNCTRACE=ON"
fi

# Final validation check
print_colored "blue" "\n🔍 FINAL PRE-BUILD VALIDATION"

# Show the actual configuration being used
if [[ "${SD_TYPES_VALIDATED:-false}" == "true" ]]; then
    print_colored "green" "✓ Types have been validated: ${SD_VALIDATED_TYPES:-none}"
    print_colored "cyan" "✓ Building with SELECTIVE types"
else
    print_colored "cyan" "✓ Building with ALL types (default)"
fi

# Show what's being passed to CMake
if [[ -n "$DATATYPES_ARG" ]]; then
    print_colored "cyan" "CMake will receive validated types: $DATATYPES_ARG"
else
    print_colored "cyan" "CMake will receive: (all types - no type restrictions)"
fi

print_colored "green" "✓ All validations passed -"
print_colored "cyan" "=====================================\n"

print_colored "green" "\n✅ Type validation completed - proceeding with build"

if [ "$FUNC_TRACE" == "ON" ] && [ "$DEBUG_AUTO_REDUCE" == "true" ]; then
    echo "=== DEBUG BUILD TYPE REDUCTION ACTIVE ==="

    if [ -n "$DEBUG_TYPE_PROFILE" ]; then
        RESOLVED_TYPES=$(resolve_debug_types "$DEBUG_TYPE_PROFILE" "$DEBUG_CUSTOM_TYPES")
        # FIXED: Only use profile if no explicit datatypes were set
        if [ -z "$DATATYPES" ]; then
            DATATYPES="$RESOLVED_TYPES"
            echo "Debug Profile: $DEBUG_TYPE_PROFILE"
            echo "Resolved Types: $DATATYPES"
        else
            echo "Debug Profile: $DEBUG_TYPE_PROFILE (IGNORED - explicit datatypes set)"
            echo "Using explicit datatypes: $DATATYPES"
        fi
    elif [ -z "$DATATYPES" ]; then
        # No types specified and no profile - use minimal safe default
        DATATYPES=$(resolve_debug_types "MINIMAL_INDEXING" "")
        echo "Auto-selected MINIMAL_INDEXING profile for debug build"
        echo "Types: $DATATYPES"
    fi

    echo "============================================="
fi

HOST=$(uname -s | tr [A-Z] [a-z])
KERNEL=$HOST-$(uname -m | tr [A-Z] [a-z])

if [ "$(uname)" == "Darwin" ]; then
    HOST="macosx"
    KERNEL="darwin-x86_64"
    echo "RUNNING OSX CLANG"
elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ] || [ "$(expr substr $(uname -s) 1 4)" == "MSYS" ]; then
    HOST="windows"
    KERNEL="windows-x86_64"
    echo "Running windows"
elif [ "$(uname -m)" == "ppc64le" ]; then
    if [ -z "$ARCH" ]; then
        ARCH="power8"
    fi
    KERNEL="linux-ppc64le"
fi

if [ -z "$OS" ]; then
    OS="$HOST"
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
    export ANDROID_NDK=$HOME/Android/android-ndk/
fi

case "$OS" in
    linux-armhf)
        if [ -z "$ARCH" ]; then
            ARCH="armv7-a"
        fi
        if [ ! -z ${RPI_BIN+set} ]; then
            export CMAKE_COMMAND="$CMAKE_COMMAND -D CMAKE_TOOLCHAIN_FILE=cmake/rpi.cmake"
        fi
        export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_ARM_BUILD=true -DSD_SANITIZE=OFF "
        ;;
    linux-arm64)
        if [ -z "$ARCH" ]; then
            ARCH="armv8-a"
        fi
        if [ ! -z ${RPI_BIN+set} ]; then
            export CMAKE_COMMAND="$CMAKE_COMMAND -D CMAKE_TOOLCHAIN_FILE=cmake/rpi.cmake"
        fi
        export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_ARM_BUILD=true"
        ;;
    android-arm)
        if [ -z "$ARCH" ]; then
            ARCH="armv7-a"
        fi

        setandroid_defaults

        # Note here for android 32 bit prefix on the binutils is different
        # See https://developer.android.com/ndk/guides/other_build_systems
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-arm/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-arm.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;
    android-arm64)
        if [ -z "$ARCH" ]; then
            ARCH="armv8-a"
        fi

        setandroid_defaults

        echo "BUILDING ANDROID ARM with KERNEL $KERNEL"
        export ANDROID_BIN="$ANDROID_NDK/toolchains/aarch64-linux-android-4.9/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-arm64/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-arm64.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;
    android-x86)
        if [ -z "$ARCH" ]; then
            ARCH="i686"
        fi

        setandroid_defaults
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-x86/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-x86.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;
    android-x86_64)

        if [ -z "$ARCH" ]; then
            ARCH="x86-64"
        fi
        echo "BUILDING ANDROID x86_64"

        setandroid_defaults


        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL/"
        export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
        export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL/bin/clang"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-$ANDROID_VERSION/arch-x86_64/"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/android-x86_64.cmake -DSD_ANDROID_BUILD=true"
        setwindows_msys
        ;;

    ios-x86_64)
        LIBTYPE="static"
        ARCH="x86-64"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-x86_64.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-x86)
        LIBTYPE="static"
        ARCH="i386"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-x86.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-arm64)
        LIBTYPE="static"
        ARCH="arm64"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-arm64.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-arm)
        LIBTYPE="static"
        ARCH="armv7"
        if xcrun --sdk iphoneos --show-sdk-version &> /dev/null; then
            export IOS_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
        else
            export IOS_VERSION="10.3"
        fi
        XCODE_PATH="$(xcode-select --print-path)"
        export IOS_SDK="$XCODE_PATH/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS$IOS_VERSION.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-arm.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    ios-armv7)
        # change those 2 parameters and make sure the IOS_SDK exists
        export iPhoneOS="iPhoneOS"
        export IOS_VERSION="10.3"
        LIBTYPE="static"
        ARCH="armv7"
        export IOS_SDK="/Applications/Xcode.app/Contents/Developer/Platforms/${iPhoneOS}.platform/Developer/SDKs/${iPhoneOS}${IOS_VERSION}.sdk"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_TOOLCHAIN_FILE=cmake/ios-armv7.cmake --debug-trycompile -DSD_IOS_BUILD=true"
        ;;

    linux*)
        ;;

    macosx*)
        export CC=clang
        export CXX=clang++
        PARALLEL="true"
        export CMAKE_COMMAND="$CMAKE_COMMAND -DCMAKE_MACOSX_RPATH=ON -DSD_APPLE_BUILD=true"
        ;;

    windows*)
        # Do something under Windows NT platform
        if [ "$CHIP" == "cuda" ]; then
            export CMAKE_COMMAND="cmake -G \"Ninja\""
            export MAKE_COMMAND="ninja -j${MAKEJ}"
            export CC="cl.exe"
            export CXX="cl.exe"
            PARALLEL="true"
            VERBOSE_ARG="-v"
        else
            # NOTE when running windows, do NOT use the gcc compiler in usr/ always use the one in mingw/bin
            # Another note on this particular gcc: ensure the path is setup in msys2 with /c/msys64/mingw64/bin
            # prefixed. If the one in USR is used, errors like error: 'RTLD_LAZY' was not declared in this scope
            # may show up
            export CMAKE_COMMAND="cmake -G \"MSYS Makefiles\""
            export MAKE_COMMAND="make -j${MAKEJ} -l${LOAD_LIMIT}"
            export CC=gcc
            export CXX=g++
            PARALLEL="true"
        fi

        # Try some defaults for Visual Studio 2013 if user has not run vcvarsall.bat or something
        if [ -z "${VCINSTALLDIR:-}" ]; then
            echo "NEED TO SET DEFAULTS FOR VISUAL STUDIO, NO VCINSTALLDIR environment variable found"
            export VisualStudioVersion=12.0
            export VSINSTALLDIR="C:\\Program Files (x86)\\Microsoft Visual Studio $VisualStudioVersion"
            export VCINSTALLDIR="$VSINSTALLDIR\\VC"
            export WindowsSdkDir="C:\\Program Files (x86)\\Windows Kits\\8.1"
            export Platform=X64
            export INCLUDE="$VCINSTALLDIR\\INCLUDE;$WindowsSdkDir\\include\\shared;$WindowsSdkDir\\include\\um"
            export LIB="$VCINSTALLDIR\\LIB\\amd64;$WindowsSdkDir\\lib\\winv6.3\\um\\x64"
            export LIBPATH="$VCINSTALLDIR\\LIB\\amd64;$WindowsSdkDir\\References\\CommonConfiguration\\Neutral"
            export PATH="$PATH:$VCINSTALLDIR\\BIN\\amd64:$WindowsSdkDir\\bin\\x64:$WindowsSdkDir\\bin\\x86"
        fi
        # Make sure we are using 64-bit MinGW-w64
        export PATH=/mingw64/bin/:/mingw64/lib:$PATH
        # export GENERATOR="MSYS Makefiles"
        ;;
esac

if [ -v OPENBLAS_PATH ]; then
  echo "OPENBLAS_PATH is set."
else
  OPENBLAS_PATH=""
fi

# Fixed auto-detection for Android and ARM platforms
if [[ -z "$OPENBLAS_PATH" ]]; then
    JAVACPP_CACHE="$HOME/.javacpp/cache"
    if [[ -d "$JAVACPP_CACHE" ]]; then
        # Determine platform-specific JAR patterns based on OS variable
        case "$OS" in
            android-x86_64)
                JAR_PATTERNS=("openblas-*-android-x86_64.jar" "openblas-*-linux-x86_64.jar")
                OPENBLAS_SUBPATHS=("android-x86_64" "linux-x86_64")
                ;;
            android-arm64)
                JAR_PATTERNS=("openblas-*-android-arm64.jar" "openblas-*-linux-arm64.jar" "openblas-*-android-aarch64.jar" "openblas-*-linux-aarch64.jar")
                OPENBLAS_SUBPATHS=("android-arm64" "linux-arm64" "android-aarch64" "linux-aarch64")
                ;;
            linux-arm64)
                JAR_PATTERNS=("openblas-*-linux-arm64.jar" "openblas-*-linux-aarch64.jar")
                OPENBLAS_SUBPATHS=("linux-arm64" "linux-aarch64")
                ;;
            macosx-arm64)
                JAR_PATTERNS=("openblas-*-macosx-arm64.jar")
                OPENBLAS_SUBPATHS=("macosx-arm64")
                ;;
            macosx-x86_64)
                JAR_PATTERNS=("openblas-*-macosx-x86_64.jar")
                OPENBLAS_SUBPATHS=("macosx-x86_64")
                ;;
            *)
                JAR_PATTERNS=("openblas-*-linux-x86_64.jar")
                OPENBLAS_SUBPATHS=("linux-x86_64")
                ;;
        esac

        # Try each pattern until we find a working JAR
        for pattern in "${JAR_PATTERNS[@]}"; do
            OPENBLAS_JAR=$(find "$JAVACPP_CACHE" -name "$pattern" | head -1)
            if [[ -n "$OPENBLAS_JAR" ]]; then
                # Try each subpath until we find cblas.h
                for subpath in "${OPENBLAS_SUBPATHS[@]}"; do
                    if [[ -f "$OPENBLAS_JAR/org/bytedeco/openblas/$subpath/include/cblas.h" ]]; then
                        export OPENBLAS_PATH="$OPENBLAS_JAR/org/bytedeco/openblas/$subpath"
                        echo "✅ Auto-detected OPENBLAS_PATH for $OS: $OPENBLAS_PATH"
                        break 2
                    fi
                done
            fi
        done

        if [[ -z "$OPENBLAS_PATH" ]]; then
            echo "⚠️  Could not auto-detect OpenBLAS for $OS platform"
        fi
    fi
fi

# ============================================================================
# MKL PATH DETECTION (for VML - Vector Math Library)
# ============================================================================
# MKL provides vectorized math functions (vsErf, vdErf, etc.) when used with oneDNN
#
# BLAS_IMPL controls which BLAS library to use:
#   - "openblas": Force OpenBLAS, skip MKL detection entirely
#   - "mkl": Force MKL (will error if not found)
#   - "" or "auto": Auto-detect (prefer MKL if available, fallback to OpenBLAS)

if [ -v MKL_PATH ]; then
  echo "MKL_PATH is set: $MKL_PATH"
else
  MKL_PATH=""
fi

# Skip MKL detection if explicitly using OpenBLAS
if [[ "$BLAS_IMPL" == "openblas" ]]; then
    echo "ℹ️  BLAS implementation set to OpenBLAS - skipping MKL detection"
    MKL_PATH=""
# Auto-detect MKL path from JavaCPP cache when onednn helper is enabled
elif [[ -z "$MKL_PATH" ]]; then
    # Check if onednn is in HELPERS or HELPER
    if [[ "$HELPERS" == *"onednn"* ]] || [[ "$HELPER" == *"onednn"* ]]; then
        echo "Attempting to auto-detect MKL path for VML support..."

        # Platform-specific subpaths
        MKL_SUBPATHS=("linux-x86_64" "linux-aarch64" "macosx-x86_64" "macosx-arm64" "windows-x86_64")

        # Search for MKL in JavaCPP cache locations
        for JAVACPP_CACHE in "$HOME/.javacpp/cache" ".javacpp/cache" "../.javacpp/cache"; do
            if [[ -d "$JAVACPP_CACHE" ]]; then
                echo "  Searching in cache: $JAVACPP_CACHE"

                # New JavaCPP cache structure: org.bytedeco/mkl/<version>/
                if [[ -d "$JAVACPP_CACHE/org.bytedeco/mkl" ]]; then
                    for MKL_VERSION_DIR in "$JAVACPP_CACHE/org.bytedeco/mkl"/*; do
                        if [[ -d "$MKL_VERSION_DIR" ]]; then
                            for subpath in "${MKL_SUBPATHS[@]}"; do
                                # Check for include directory with mkl.h
                                if [[ -f "$MKL_VERSION_DIR/$subpath/include/mkl.h" ]]; then
                                    export MKL_PATH="$MKL_VERSION_DIR/$subpath"
                                    echo "✅ Auto-detected MKL_PATH (new structure): $MKL_PATH"
                                    break 3
                                fi
                                # Check nested org/bytedeco/mkl structure (from JAR extraction)
                                if [[ -f "$MKL_VERSION_DIR/org/bytedeco/mkl/$subpath/include/mkl.h" ]]; then
                                    export MKL_PATH="$MKL_VERSION_DIR/org/bytedeco/mkl/$subpath"
                                    echo "✅ Auto-detected MKL_PATH (JAR structure): $MKL_PATH"
                                    break 3
                                fi
                            done
                        fi
                    done
                fi

                # Old structure: mkl-<version>/ at top level
                MKL_JAR=$(find "$JAVACPP_CACHE" -maxdepth 1 -name "mkl-*" -type d 2>/dev/null | head -1)
                if [[ -n "$MKL_JAR" && -d "$MKL_JAR" ]]; then
                    for subpath in "${MKL_SUBPATHS[@]}"; do
                        if [[ -f "$MKL_JAR/org/bytedeco/mkl/$subpath/include/mkl.h" ]]; then
                            export MKL_PATH="$MKL_JAR/org/bytedeco/mkl/$subpath"
                            echo "✅ Auto-detected MKL_PATH (legacy structure): $MKL_PATH"
                            break 2
                        fi
                    done
                fi
            fi
        done

        if [[ -z "$MKL_PATH" ]]; then
            if [[ "$BLAS_IMPL" == "mkl" ]]; then
                print_colored "red" "❌ ERROR: MKL explicitly requested but not found in JavaCPP cache"
                print_colored "yellow" "    Ensure mkl artifact is in dependencies when using --blas mkl"
                exit 1
            else
                echo "ℹ️  MKL not found in JavaCPP cache - VML will use scalar fallback"
                echo "   To enable MKL VML, add -Dlibnd4j.blas=mkl to your build command"
            fi
        fi
    fi
fi

# If MKL_PATH not found in JavaCPP cache, try to extract from Maven repository
if [[ -z "$MKL_PATH" ]]; then
    M2_MKL_DIR="$HOME/.m2/repository/org/bytedeco/mkl"
    if [[ -d "$M2_MKL_DIR" ]]; then
        echo "  Searching for MKL redist JAR in Maven repository..."
        # Find the latest MKL version directory
        MKL_VERSION_DIR=$(find "$M2_MKL_DIR" -maxdepth 1 -type d -name "*-1.5.*" | sort -V | tail -1)
        if [[ -n "$MKL_VERSION_DIR" ]]; then
            # Platform-specific redist JAR
            case "$(uname -s)-$(uname -m)" in
                Linux-x86_64)  PLATFORM_SUFFIX="linux-x86_64" ;;
                Darwin-x86_64) PLATFORM_SUFFIX="macosx-x86_64" ;;
                Darwin-arm64)  PLATFORM_SUFFIX="macosx-arm64" ;;
                MINGW*|MSYS*|CYGWIN*) PLATFORM_SUFFIX="windows-x86_64" ;;
                *) PLATFORM_SUFFIX="" ;;
            esac

            if [[ -n "$PLATFORM_SUFFIX" ]]; then
                MKL_REDIST_JAR=$(find "$MKL_VERSION_DIR" -name "mkl-*-${PLATFORM_SUFFIX}-redist.jar" | head -1)
                if [[ -f "$MKL_REDIST_JAR" ]]; then
                    EXTRACT_DIR="${MKL_VERSION_DIR}/mkl-redist-extracted"
                    if [[ ! -d "$EXTRACT_DIR/org/bytedeco/mkl/${PLATFORM_SUFFIX}/include" ]]; then
                        echo "  Extracting MKL redist JAR: $MKL_REDIST_JAR"
                        mkdir -p "$EXTRACT_DIR"
                        unzip -q -o "$MKL_REDIST_JAR" -d "$EXTRACT_DIR" 2>/dev/null
                    fi
                    if [[ -f "$EXTRACT_DIR/org/bytedeco/mkl/${PLATFORM_SUFFIX}/include/mkl.h" ]]; then
                        export MKL_PATH="$EXTRACT_DIR/org/bytedeco/mkl/${PLATFORM_SUFFIX}"
                        echo "✅ Extracted MKL headers to: $MKL_PATH"
                    fi
                fi
            fi
        fi
    fi
fi

# Export MKL_PATH for CMake
if [[ -n "$MKL_PATH" ]]; then
    MKL_CMAKE="-DMKL_ROOT=$MKL_PATH"
    echo "✅ MKL VML enabled: $MKL_PATH"
else
    MKL_CMAKE=""
fi

# Also pass BLAS_IMPL to CMake for conditional compilation
if [[ -n "$BLAS_IMPL" ]]; then
    BLAS_CMAKE="-DBLAS_IMPL=$BLAS_IMPL"
else
    BLAS_CMAKE=""
fi

if [ ! -d "include/generated" ]; then
    mkdir -p "include/generated"
fi

if [ -f "$OP_OUTPUT_FILE" ]; then
    rm -f "${OP_OUTPUT_FILE}"
fi

if [ -z "$BUILD" ]; then
    BUILD="release"
fi

if [ -z "$CHIP" ]; then
    CHIP="cpu"
fi

if [ -z "$LIBTYPE" ]; then
    LIBTYPE="dynamic"
fi

if [ -z "$PACKAGING" ]; then
    PACKAGING="none"
fi

SOURCE_PATH="$DIR"
BUILD_DIR="$DIR/blasbuild/$CHIP"

# =============================================================================
# TMPDIR Configuration - Use build directory instead of /tmp (tmpfs)
# NVCC and GCC create large temporary files during compilation. By using
# a directory under the build tree, we use disk storage instead of RAM.
# This prevents /tmp from filling up during large CUDA/functrace builds.
# =============================================================================
COMPILER_TMPDIR="$BUILD_DIR/compiler_tmp"
mkdir -p "$COMPILER_TMPDIR"
export TMPDIR="$COMPILER_TMPDIR"
export TMP="$COMPILER_TMPDIR"
export TEMP="$COMPILER_TMPDIR"
print_colored "cyan" "📁 Compiler temp directory: $COMPILER_TMPDIR"

export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_SANITIZE=$SANITIZE -DSD_SANITIZERS=$SANITIZERS"

if [ "$CHIP_EXTENSION" == "avx512" ] || [ "$ARCH" == "avx512" ]; then
    CHIP_EXTENSION="avx512"
    ARCH="skylake-avx512"
elif [ "$CHIP_EXTENSION" == "avx2" ] || [ "$ARCH" == "avx2" ]; then
    CHIP_EXTENSION="avx2"
    ARCH="x86-64"
elif [ "$CHIP_EXTENSION" == "x86_64" ] || [ "$ARCH" == "x86_64" ]; then
    CHIP_EXTENSION="x86_64"
    ARCH="x86-64"
fi

if [ -z "$ARCH" ]; then
    ARCH="x86-64"
fi

if [ -z "$COMPUTE" ]; then
    COMPUTE="all"
fi

# Enable call stacking
if [ "$FUNC_TRACE" == "ON" ]; then
    export CMAKE_COMMAND="$CMAKE_COMMAND -DSD_GCC_FUNCTRACE=ON"
fi

OPERATIONS_ARG=

if [ -z "$OPERATIONS" ]; then
    OPERATIONS_ARG="-DSD_ALL_OPS=true"
else
    OPERATIONS_ARG="-DSD_OPS_LIST=\"$OPERATIONS\" -DSD_ALL_OPS=false"
fi

DATATYPES_ARG=

if [ -n "$DATATYPES" ]; then
    DATATYPES_ARG="-DSD_TYPES_LIST=\"$DATATYPES\""
fi

if [ -z "$EXPERIMENTAL" ]; then
    EXPERIMENTAL="no"
fi

if [ "$CHIP" == "cpu" ]; then
    BLAS_ARG="-DSD_CPU=true -DBLAS=TRUE"
elif [ "$CHIP" == "aurora" ]; then
    BLAS_ARG="-DSD_AURORA=true -DBLAS=TRUE"
elif [ "$CHIP" == "cuda" ]; then
    BLAS_ARG="-DSD_CUDA=true -DBLAS=TRUE"
elif [ "$CHIP" == "tpu" ]; then
    BLAS_ARG="-DSD_TPU=true -DBLAS=TRUE"
else
    # Handle CUDA version numbers like "12.9" - treat as CUDA build
    BLAS_ARG="-DSD_CUDA=true -DBLAS=TRUE"
fi

if [ -z "$NAME" ]; then
    if [ "$CHIP" == "cpu" ]; then
        NAME="nd4jcpu"
    elif [ "$CHIP" == "cuda" ]; then
        NAME="nd4jcuda"
    elif [ "$CHIP" == "aurora" ]; then
        NAME="nd4jaurora"
    elif [ "$CHIP" == "tpu" ]; then
        NAME="nd4jtpu"
    else
        # Handle CUDA version numbers like "12.9" - treat as CUDA build
        NAME="nd4jcuda"
    fi
fi

if [ "$LIBTYPE" == "dynamic" ]; then
    SHARED_LIBS_ARG="-DSD_SHARED_LIB=ON -DSD_STATIC_LIB=OFF"
else
    SHARED_LIBS_ARG="-DSD_SHARED_LIB=OFF -DSD_STATIC_LIB=ON"
fi

# Set build type
if [ "$FUNC_TRACE" == "ON" ]; then
    BUILD_TYPE="-DCMAKE_BUILD_TYPE=none"
elif [ "$BUILD" == "release" ]; then
    BUILD_TYPE="-DCMAKE_BUILD_TYPE=Release"
else
    BUILD_TYPE="-DCMAKE_BUILD_TYPE=Debug"
fi

if [ "$PACKAGING" == "none" ]; then
    PACKAGING_ARG="-DPACKAGING=none"
fi

if [ "$PACKAGING" == "rpm" ]; then
    PACKAGING_ARG="-DPACKAGING=rpm"
fi

if [ "$PACKAGING" == "deb" ]; then
    PACKAGING_ARG="-DPACKAGING=deb"
fi

if [ "$PACKAGING" == "msi" ]; then
    PACKAGING_ARG="-DPACKAGING=msi"
fi

# Use parent of output file to mean source include directory
OP_OUTPUT_FILE_ARG="-DOP_OUTPUT_FILE=../${OP_OUTPUT_FILE}"

EXPERIMENTAL_ARG=""
MINIFIER_ARG="-DSD_BUILD_MINIFIER=false"
TESTS_ARG="-DSD_BUILD_TESTS=OFF"
NAME_ARG="-DSD_LIBRARY_NAME=$NAME"

if [ "$EXPERIMENTAL" == "yes" ]; then
    EXPERIMENTAL_ARG="-DSD_EXPERIMENTAL=yes"
fi

if [ "$MINIFIER" == "true" ]; then
    MINIFIER_ARG="-DSD_BUILD_MINIFIER=true"
fi

if [ "$TESTS" == "true" ]; then
    MINIFIER_ARG="-DSD_BUILD_MINIFIER=true"
    TESTS_ARG="-DSD_BUILD_TESTS=ON"
fi

ARCH_ARG="-DSD_ARCH=$ARCH -DSD_EXTENSION=$CHIP_EXTENSION"

CUDA_COMPUTE="-DCOMPUTE=\"$COMPUTE\""

if [ "$CHIP" == "cuda" ] && [ -n "$CHIP_VERSION" ]; then
    case $OS in
        linux*)
            if [ "${CUDA_PATH-}" == "" ]; then
                export CUDA_PATH="/usr/local/cuda-$CHIP_VERSION/"
            fi
            ;;
        macosx*)
            export CUDA_PATH="/Developer/NVIDIA/CUDA-$CHIP_VERSION/"
            ;;
        windows*)
            export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$CHIP_VERSION/"
            ;;
    esac
fi



mkbuilddir() {
    if [ "$CLEAN" == "true" ]; then
        if [ "$CLEAN_ALL" == "true" ]; then
            echo "Removing ALL blasbuild directories (clean-all)"
            rm -Rf "$DIR/blasbuild"
        else
            echo "Removing blasbuild for $CHIP only (use 'clean-all' to clean all backends)"
            rm -Rf "$DIR/blasbuild/$CHIP"
        fi
    fi

    mkdir -p "$BUILD_DIR"

    # File locking to prevent concurrent builds of the same backend
    LOCK_FILE="$BUILD_DIR/.build.lock"
    exec 200>"$LOCK_FILE"
    LOCK_WAIT_SECONDS="${LOCK_WAIT_SECONDS:-1200}"

    if command -v flock >/dev/null 2>&1; then
        # Some environments (notably certain macOS runner setups) may have flock
        # installed but unable to lock file descriptors reliably.
        FLOCK_TEST_FILE="$BUILD_DIR/.flock.test"
        exec 201>"$FLOCK_TEST_FILE"
        FLOCK_SUPPORTED=true
        if ! flock -n 201 2>/dev/null; then
            FLOCK_SUPPORTED=false
        fi
        exec 201>&-
        rm -f "$FLOCK_TEST_FILE"

        if [ "$FLOCK_SUPPORTED" = "true" ]; then
            LOCK_START_TS=$(date +%s)
            while ! flock -n 200 2>/dev/null; do
                NOW_TS=$(date +%s)
                ELAPSED=$((NOW_TS - LOCK_START_TS))
                if [ "$ELAPSED" -ge "$LOCK_WAIT_SECONDS" ]; then
                    print_colored "red" "ERROR: Another build is already running for backend '$CHIP'"
                    print_colored "yellow" "Lock file: $LOCK_FILE"
                    print_colored "yellow" "Waited ${LOCK_WAIT_SECONDS}s for lock. If no other build is running, remove the lock file manually"
                    exit 1
                fi
                sleep 5
            done
        else
            print_colored "yellow" "WARNING: flock is unavailable for this environment; skipping backend lock"
        fi
    else
        print_colored "yellow" "WARNING: flock command not found; skipping backend lock"
    fi
    # Lock will be released when script exits (fd 200 closes)

    builtin cd "$BUILD_DIR"
}

HELPERS_CMAKE=""

# Process multi-helper configuration (--helpers flag)
if [ -n "$HELPERS" ]; then
    print_colored "blue" "=== Multi-Helper Configuration ==="
    # Convert comma-separated list to CMake semicolon-separated list
    HELPERS_LIST_CMAKE=$(echo "$HELPERS" | tr ',' ';')
    HELPERS_CMAKE="-DHELPERS_LIST=\"${HELPERS_LIST_CMAKE}\""

    # Also set individual helper flags for backwards compatibility
    IFS=','
    read -ra HLP <<< "$HELPERS"
    for i in "${HLP[@]}"; do
        # Trim whitespace
        i=$(echo "$i" | tr -d ' ')
        HELPERS_CMAKE="${HELPERS_CMAKE} -DHELPERS_$i=ON"
        print_colored "green" "   ✓ Enabling helper: $i"
    done
    IFS=' '
    print_colored "blue" "==================================="
# Fallback to legacy single helper mode (-h flag)
elif [ -n "$HELPER" ]; then
    print_colored "yellow" "Using legacy single-helper mode (consider using --helpers for multi-backend support)"
    IFS=','
    read -ra HLP <<< "$HELPER"
    for i in "${HLP[@]}"; do
        HELPERS_CMAKE="${HELPERS_CMAKE} -DHELPERS_$i=ON"
    done
    IFS=' '
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!                                                                                                           !!"
    echo "!!                                                 WARNING!                                                  !!"
    echo "!!                                      No helper packages configured!                                       !!"
    echo "!!         You can specify helpers using --helpers key. I.e. <--helpers onednn,cudnn>                        !!"
    echo "!!         Or use legacy mode with -h key. I.e. <-h onednn>                                                  !!"
    echo "!!                                                                                                           !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

# Dynamic kernel selection CMake arguments
KERNEL_CMAKE=""
if [ "$DYNAMIC_KERNEL_SELECTION" == "ON" ]; then
    KERNEL_CMAKE="-DSD_DYNAMIC_KERNEL_SELECTION=ON"
    KERNEL_CMAKE="${KERNEL_CMAKE} -DSD_KERNEL_STRATEGY=${KERNEL_STRATEGY}"

    if [ "$KERNEL_AUTOTUNING" == "ON" ]; then
        KERNEL_CMAKE="${KERNEL_CMAKE} -DSD_KERNEL_AUTOTUNING=ON"
    fi
    if [ "$KERNEL_CACHING" == "ON" ]; then
        KERNEL_CMAKE="${KERNEL_CMAKE} -DSD_KERNEL_CACHING=ON"
    fi
    if [ -n "$HELPER_PRIORITY" ]; then
        # Convert comma-separated to semicolon-separated for CMake
        HELPER_PRIORITY_CMAKE=$(echo "$HELPER_PRIORITY" | tr ',' ';')
        KERNEL_CMAKE="${KERNEL_CMAKE} -DSD_HELPER_PRIORITY=\"${HELPER_PRIORITY_CMAKE}\""
    fi
else
    KERNEL_CMAKE="-DSD_DYNAMIC_KERNEL_SELECTION=OFF"
fi

# Build MLIR CMake arguments
MLIR_ARG=""
if [ "$MLIR" == "ON" ]; then
    MLIR_ARG="-DHELPERS_mlir=ON -DMLIR_VERSION=${MLIR_VERSION}"
    if [ "$MLIR_GPU" == "ON" ]; then
        MLIR_ARG="${MLIR_ARG} -DMLIR_ENABLE_GPU=ON"
    fi
fi
TRITON_CMAKE="-DSD_TRITON=${TRITON}"
UNITY_BUILD_CMAKE="-DSD_UNITY_BUILD=${UNITY_BUILD}"

# Dependency cache CMake arguments
DEP_CACHE_CMAKE="-DSD_DEP_CACHE=${DEP_CACHE}"
if [ -n "$DEP_CACHE_DIR" ]; then
    DEP_CACHE_CMAKE="${DEP_CACHE_CMAKE} -DSD_DEP_CACHE_DIR=${DEP_CACHE_DIR}"
fi
DEP_CACHE_CMAKE="${DEP_CACHE_CMAKE} -DSD_DEP_CACHE_CLEAR=${DEP_CACHE_CLEAR}"
if [ -n "$DEP_CACHE_CLEAR_DEP" ]; then
    DEP_CACHE_CMAKE="${DEP_CACHE_CMAKE} -DSD_DEP_CACHE_CLEAR_DEP=${DEP_CACHE_CLEAR_DEP}"
fi

echo PACKAGING           = "${PACKAGING}"
echo BUILD               = "${BUILD}"
echo CHIP                = "${CHIP}"
echo ARCH                = "${ARCH}"
echo CHIP_EXTENSION      = "${CHIP_EXTENSION}"
echo CHIP_VERSION        = "${CHIP_VERSION}"
echo GPU_COMPUTE_CAPABILITY = "${COMPUTE}"
echo EXPERIMENTAL        = "${EXPERIMENTAL}"
echo LIBRARY TYPE        = "${LIBTYPE}"
echo OPERATIONS          = "${OPERATIONS_ARG}"
echo DATATYPES           = "${DATATYPES_ARG}"
echo MINIFIER            = "${MINIFIER_ARG}"
echo TESTS               = "${TESTS_ARG}"
echo NAME                = "${NAME_ARG}"
echo OPENBLAS_PATH       = "$OPENBLAS_PATH"
echo CHECK_VECTORIZATION = "$CHECK_VECTORIZATION"
echo HELPER              = "$HELPER"
echo HELPERS             = "$HELPERS"
echo HELPERS_CMAKE       = "$HELPERS_CMAKE"
echo DYNAMIC_KERNEL_SEL  = "$DYNAMIC_KERNEL_SELECTION"
echo KERNEL_STRATEGY     = "$KERNEL_STRATEGY"
echo KERNEL_AUTOTUNING   = "$KERNEL_AUTOTUNING"
echo KERNEL_CACHING      = "$KERNEL_CACHING"
echo HELPER_PRIORITY     = "$HELPER_PRIORITY"
echo MLIR                = "$MLIR"
echo MLIR_VERSION        = "$MLIR_VERSION"
echo MLIR_GPU            = "$MLIR_GPU"
echo TRITON              = "$TRITON"
echo UNITY_BUILD         = "$UNITY_BUILD"
echo OP_OUTPUT_FILE      = "$OP_OUTPUT_FILE"
echo USE_LTO             = "$USE_LTO"
echo CUDA_LTO            = "$CUDA_LTO"
echo CUDA_THREADS        = "$CUDA_THREADS"
echo CUDA_SPLIT_COMPILE  = "$CUDA_SPLIT_COMPILE"
echo SANITIZE            = "$SANITIZE"
echo FUNC_TRACE          = "$FUNC_TRACE"
echo LOG_OUTPUT          = "$LOG_OUTPUT"
echo KEEP_NVCC           = "$KEEP_NVCC"
echo PRINT_INDICES       = "$PRINT_INDICES"
echo PRINT_MATH          = "$PRINT_MATH"
echo PREPROCESS          = "$PREPROCESS"
echo BUILD_PPSTEP        = "$BUILD_PPSTEP"
echo MAKEJ               = "$MAKEJ"
mkbuilddir
pwd

# ----------------------- CMake Configuration -----------------------

# Configure compiler if specified
COMPILER_ARG=""
if [ -n "$COMPILER" ]; then
    case "$COMPILER" in
        clang|clang++)
            COMPILER_ARG="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
            print_colored "cyan" "Using Clang compiler"
            ;;
        gcc|g++)
            COMPILER_ARG="-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++"
            print_colored "cyan" "Using GCC compiler"
            ;;
        *)
            COMPILER_ARG="-DCMAKE_C_COMPILER=$COMPILER -DCMAKE_CXX_COMPILER=${COMPILER}++"
            print_colored "cyan" "Using custom compiler: $COMPILER"
            ;;
    esac
fi

# Configure CMake
echo "$CMAKE_COMMAND - -DSD_KEEP_NVCC_OUTPUT=$KEEP_NVCC -DSD_GCC_FUNCTRACE=$FUNC_TRACE $BLAS_ARG $ARCH_ARG $NAME_ARG $OP_OUTPUT_FILE_ARG -DSD_SANITIZERS=${SANITIZERS} -DSD_SANITIZE=${SANITIZE} -DSD_CHECK_VECTORIZATION=${CHECK_VECTORIZATION} $USE_LTO $HELPERS $MLIR_ARG $TRITON_CMAKE $SHARED_LIBS_ARG $MINIFIER_ARG $OPERATIONS_ARG $DATATYPES_ARG $BUILD_TYPE $PACKAGING_ARG $EXPERIMENTAL_ARG $TESTS_ARG $CUDA_COMPUTE -DOPENBLAS_PATH=$OPENBLAS_PATH -DDEV=FALSE -DCMAKE_NEED_RESPONSE=YES -DMKL_MULTI_THREADED=TRUE $COMPILER_ARG $SOURCE_PATH"

# Handle the PREPROCESS flag first - before any build
if [ "$PREPROCESS" == "ON" ]; then
    echo "Running preprocessing step..."
    if [ "$LOG_OUTPUT" == "none" ]; then
        eval "$CMAKE_COMMAND" \
            -DSD_PREPROCESS=ON \
            -DBUILD_PPSTEP="$BUILD_PPSTEP" \
            "$BLAS_ARG" \
            "$ARCH_ARG" \
            "$NAME_ARG" \
            "$OP_OUTPUT_FILE_ARG" \
            -DSD_SANITIZE="${SANITIZE}" \
            -DSD_BUILD_WITH_JAVA="${BUILD_WITH_JAVA}" \
            "$USE_LTO" \
            -DSD_CUDA_DEVICE_LTO="${CUDA_LTO}" \
            -DSD_CUDA_THREADS="${CUDA_THREADS}" \
            -DSD_CUDA_SPLIT_COMPILE="${CUDA_SPLIT_COMPILE}" \
            $HELPERS_CMAKE \
            $KERNEL_CMAKE \
            $DEP_CACHE_CMAKE \
            $MLIR_ARG \
            $TRITON_CMAKE \
        $UNITY_BUILD_CMAKE \
            $UNITY_BUILD_CMAKE \
            "$SHARED_LIBS_ARG" \
            "$OPERATIONS_ARG" \
            "$DATATYPES_ARG" \
            "$BUILD_TYPE" \
            "$PACKAGING_ARG" \
            "$CUDA_COMPUTE" \
            -DOPENBLAS_PATH="$OPENBLAS_PATH" \
            $MKL_CMAKE \
            $BLAS_CMAKE \
            -DDEV=FALSE \
            -DCMAKE_NEED_RESPONSE=YES \
            -DMKL_MULTI_THREADED=TRUE \
            $COMPILER_ARG \
            "$SOURCE_PATH"
    else
        eval "$CMAKE_COMMAND" \
            -DSD_PREPROCESS=ON \
            -DBUILD_PPSTEP="$BUILD_PPSTEP" \
            "$BLAS_ARG" \
            "$ARCH_ARG" \
            "$NAME_ARG" \
            "$OP_OUTPUT_FILE_ARG" \
            -DSD_SANITIZE="${SANITIZE}" \
            -DSD_BUILD_WITH_JAVA="${BUILD_WITH_JAVA}" \
            "$USE_LTO" \
            -DSD_CUDA_DEVICE_LTO="${CUDA_LTO}" \
            -DSD_CUDA_THREADS="${CUDA_THREADS}" \
            -DSD_CUDA_SPLIT_COMPILE="${CUDA_SPLIT_COMPILE}" \
            $HELPERS_CMAKE \
            $KERNEL_CMAKE \
            $DEP_CACHE_CMAKE \
            $MLIR_ARG \
            $TRITON_CMAKE \
        $UNITY_BUILD_CMAKE \
            $UNITY_BUILD_CMAKE \
            "$SHARED_LIBS_ARG" \
            "$OPERATIONS_ARG" \
            "$DATATYPES_ARG" \
            "$BUILD_TYPE" \
            "$PACKAGING_ARG" \
            "$CUDA_COMPUTE" \
            -DOPENBLAS_PATH="$OPENBLAS_PATH" \
            $MKL_CMAKE \
            $BLAS_CMAKE \
            -DDEV=FALSE \
            -DCMAKE_NEED_RESPONSE=YES \
            -DMKL_MULTI_THREADED=TRUE \
            $COMPILER_ARG \
            "$SOURCE_PATH" >> "$LOG_OUTPUT" 2>&1
    fi
    echo "Preprocessing complete - exiting"
    exit 0
fi

# For dry-run mode: Clear cache before cmake runs, then set CMAKE_ONLY to exit after cmake
if [ "$DRY_RUN" == "ON" ]; then
    print_colored "blue" "=========================================="
    print_colored "blue" "DRY RUN MODE - CMake Configuration Only"
    print_colored "blue" "=========================================="
    echo

    # Clear CMake cache to ensure fresh configuration
    BLASBUILD_DIR="$DIR/blasbuild/${CHIP}"
    if [ -d "$BLASBUILD_DIR" ]; then
        print_colored "yellow" "Clearing CMake cache for fresh configuration..."
        rm -rf "$BLASBUILD_DIR/CMakeCache.txt" "$BLASBUILD_DIR/CMakeFiles"
        print_colored "green" "✅ Cache cleared"
        echo
    fi

    print_colored "cyan" "CMake will now configure the build..."
    echo

    # Set CMAKE_ONLY to exit after CMake completes (and run validation then)
    CMAKE_ONLY="ON"
fi
# Normal build path
if [ "$LOG_OUTPUT" == "none" ]; then
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
        -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
        -DSD_PTXAS="$PTXAS_INFO" \
        -DSD_EXTRACT_INSTANTIATIONS="$EXTRACT_INSTANTIATIONS" \
        -DSD_PARALLEL_COMPILE_JOBS="${MAKEJ}" \
        "$BLAS_ARG" \
        "$ARCH_ARG" \
        "$NAME_ARG" \
        "$OP_OUTPUT_FILE_ARG" \
        -DSD_SANITIZE="${SANITIZE}" \
        -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
        -DSD_BUILD_WITH_JAVA="${BUILD_WITH_JAVA}" \
        "$USE_LTO" \
        -DSD_CUDA_DEVICE_LTO="${CUDA_LTO}" \
        -DSD_CUDA_THREADS="${CUDA_THREADS}" \
        -DSD_CUDA_SPLIT_COMPILE="${CUDA_SPLIT_COMPILE}" \
        $HELPERS_CMAKE \
        $KERNEL_CMAKE \
        $DEP_CACHE_CMAKE \
        $MLIR_ARG \
        $TRITON_CMAKE \
        $UNITY_BUILD_CMAKE \
        "$SHARED_LIBS_ARG" \
        "$MINIFIER_ARG" \
        "$OPERATIONS_ARG" \
        "$DATATYPES_ARG" \
        "$BUILD_TYPE" \
        "$PACKAGING_ARG" \
        "$TESTS_ARG" \
        "$CUDA_COMPUTE" \
        -DOPENBLAS_PATH="$OPENBLAS_PATH" \
        $MKL_CMAKE \
        $BLAS_CMAKE \
        -DDEV=FALSE \
        -DCMAKE_NEED_RESPONSE=YES \
        -DMKL_MULTI_THREADED=TRUE \
        $COMPILER_ARG \
        "$SOURCE_PATH"
else
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
        -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
        -DSD_PTXAS="$PTXAS_INFO" \
        -DSD_EXTRACT_INSTANTIATIONS="$EXTRACT_INSTANTIATIONS" \
        -DSD_PARALLEL_COMPILE_JOBS="${MAKEJ}" \
        "$BLAS_ARG" \
        "$ARCH_ARG" \
        "$NAME_ARG" \
        "$OP_OUTPUT_FILE_ARG" \
        -DSD_SANITIZE="${SANITIZE}" \
        -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
        -DSD_BUILD_WITH_JAVA="${BUILD_WITH_JAVA}" \
        "$USE_LTO" \
        -DSD_CUDA_DEVICE_LTO="${CUDA_LTO}" \
        -DSD_CUDA_THREADS="${CUDA_THREADS}" \
        -DSD_CUDA_SPLIT_COMPILE="${CUDA_SPLIT_COMPILE}" \
        $HELPERS_CMAKE \
        $KERNEL_CMAKE \
        $DEP_CACHE_CMAKE \
        $MLIR_ARG \
        $TRITON_CMAKE \
        $UNITY_BUILD_CMAKE \
        "$SHARED_LIBS_ARG" \
        "$MINIFIER_ARG" \
        "$OPERATIONS_ARG" \
        "$DATATYPES_ARG" \
        "$BUILD_TYPE" \
        "$PACKAGING_ARG" \
        "$TESTS_ARG" \
        "$CUDA_COMPUTE" \
        -DOPENBLAS_PATH="$OPENBLAS_PATH" \
        $MKL_CMAKE \
        $BLAS_CMAKE \
        -DDEV=FALSE \
        -DCMAKE_NEED_RESPONSE=YES \
        -DMKL_MULTI_THREADED=TRUE \
        $COMPILER_ARG \
        "$SOURCE_PATH" >> "$LOG_OUTPUT" 2>&1
fi


if [[ "$EXTRACT_INSTANTIATIONS" == "ON" ]]; then
    INST_DIR="${DIR}/blasbuild/${CHIP}/instantiation_analysis"
    
    # Always create the directory structure if it doesn't exist
    mkdir -p "$INST_DIR/reports"
    
    # Create a simple report even if no data was collected
    if [[ ! -f "$INST_DIR/all_missing.txt" ]]; then
        echo "No missing templates detected" > "$INST_DIR/all_missing.txt"
    fi
    if [[ ! -f "$INST_DIR/all_used.txt" ]]; then
        echo "No template usage data collected" > "$INST_DIR/all_used.txt"
    fi
    if [[ ! -f "$INST_DIR/all_provided.txt" ]]; then
        echo "No template provision data collected" > "$INST_DIR/all_provided.txt"
    fi
    
    # Generate a simple summary
    echo "=== Template Instantiation Analysis Complete ===" > "$INST_DIR/reports/summary.txt"
    echo "Date: $(date)" >> "$INST_DIR/reports/summary.txt"
    echo "Missing: $(wc -l < "$INST_DIR/all_missing.txt" 2>/dev/null || echo 0)" >> "$INST_DIR/reports/summary.txt"
    echo "Used: $(wc -l < "$INST_DIR/all_used.txt" 2>/dev/null || echo 0)" >> "$INST_DIR/reports/summary.txt"
    echo "Provided: $(wc -l < "$INST_DIR/all_provided.txt" 2>/dev/null || echo 0)" >> "$INST_DIR/reports/summary.txt"
    
    cat "$INST_DIR/reports/summary.txt"
    
    # Exit after extraction
    exit 0
fi


# IMPORTANT: CMake invocations happen below (lines ~1465-1550)
# The CMAKE_ONLY check must come AFTER those invocations

# Exit if cmake-only mode is enabled (this runs AFTER cmake completes)
if [ "$CMAKE_ONLY" == "ON" ]; then
    print_colored "green" "✅ CMake configuration completed - exiting without build"
    print_colored "cyan" "=== CMAKE CONFIGURATION SUMMARY ==="
    print_colored "cyan" "Build directory: $(pwd)"
    print_colored "cyan" "To inspect CMake cache: cmake -L"
    print_colored "cyan" "To inspect specific values: cmake -LA | grep <pattern>"
    print_colored "cyan" "To run build manually: $MAKE_COMMAND"
    echo

    # If this was a dry-run, run post-cmake validation
    if [ "$DRY_RUN" == "ON" ] && [ -d "$DIR/preflight-checks" ] && [ -x "$DIR/preflight-checks/validate_build_config.sh" ]; then
        echo
        print_colored "blue" "=========================================="
        print_colored "blue" "Running Post-CMake Validation"
        print_colored "blue" "=========================================="
        echo

        builtin cd "$DIR/preflight-checks"
        if ./validate_build_config.sh --stage cmake --chip "$CHIP"; then
            print_colored "green" "✅ Configuration validation passed"
            echo
            print_colored "cyan" "Build configuration is ready!"
            print_colored "cyan" "To start the actual build, run:"
            print_colored "cyan" "  ./buildnativeoperations.sh <same-arguments-without-dry-run>"
        else
            print_colored "red" "❌ Configuration validation failed"
            print_colored "yellow" "Fix the issues above before building"
            exit 1
        fi
    else
        print_colored "yellow" "NEXT STEPS FOR SMOKE TEST:"
        print_colored "yellow" "1. Check CMake cache: cd blasbuild/${CHIP} && cmake -LA"
        print_colored "yellow" "2. Verify type configuration: grep 'SD_TYPES_LIST\\|SD_SELECTIVE_TYPES' CMakeCache.txt"
        print_colored "yellow" "3. Check selective rendering: cat include/system/selective_rendering.h | head -50"
        print_colored "yellow" "4. If values look correct, run build: cd ../.. && ./buildnativeoperations.sh"
    fi
    echo
    exit 0
fi


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Function to check template instantiation status
check_instantiation_status() {
    local inst_dir="instantiation_analysis"
    
    if [[ ! -d "$inst_dir" ]]; then
        print_colored "yellow" "No instantiation analysis found. Run with --extract-instantiations"
        return 1
    fi
    
    if [[ -f "$inst_dir/all_missing.txt" ]]; then
        local missing_count=$(wc -l < "$inst_dir/all_missing.txt")
        if [[ $missing_count -gt 0 ]]; then
            print_colored "red" "⚠️  $missing_count missing template instantiations from last analysis"
            print_colored "yellow" "Run with --extract-instantiations for updated analysis"
            return 1
        fi
    fi
    
    return 0
}

# Function to apply instantiation fixes
apply_instantiation_fixes() {
    local fixes_dir="instantiation_analysis/fixes"
    
    if [[ ! -d "$fixes_dir" ]]; then
        print_colored "yellow" "No instantiation fixes available"
        return 1
    fi
    
    if [[ -f "$fixes_dir/missing_instantiations.cpp" ]]; then
        print_colored "cyan" "Found instantiation fix file"
        # Add to source list or copy to appropriate location
        # This depends on your build system setup
        return 0
    fi
    
    return 1
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

show_instantiation_help() {
    cat << EOF

TEMPLATE INSTANTIATION ANALYSIS OPTIONS:
  --extract-instantiations    Extract template usage and instantiation data
  --generate-fix-files        Generate C++ files to fix missing instantiations  
  --instantiation-report      Generate detailed HTML analysis report
  --instantiation-verbose     Enable verbose output during analysis
  --analyze-templates         Quick mode: analyze and generate report, then exit

EXAMPLES:
  # Basic analysis
  ./build.sh --extract-instantiations
  
  # Analysis with specific datatypes
  ./build.sh --extract-instantiations --datatypes "float32;double;int32;int64"
  
  # Generate fixes for missing instantiations
  ./build.sh --generate-fix-files
  
  # Full analysis with report
  ./build.sh --analyze-templates --datatypes "float32;double"
  
  # Check status of last analysis
  ./build.sh --check-instantiations

WORKFLOW:
  1. Run analysis: ./build.sh --extract-instantiations
  2. Check results in instantiation_analysis/
  3. If missing instantiations found, generate fixes:
     ./build.sh --generate-fix-files
  4. Add generated fix file to your build
  5. Rebuild normally

EOF
}
# Build the project


if [ "$BUILD_PPSTEP" == "ON" ]; then
    print_colored "cyan" "Building ppstep preprocessor tool..."
    
    if [ "$LOG_OUTPUT" == "none" ]; then
        eval "$CMAKE_COMMAND" \
            -DBUILD_PPSTEP=ON \
            "$BLAS_ARG" \
            "$ARCH_ARG" \
            "$NAME_ARG" \
            $TRITON_CMAKE \
        $UNITY_BUILD_CMAKE \
            $UNITY_BUILD_CMAKE \
            -DOPENBLAS_PATH="$OPENBLAS_PATH" \
            $MKL_CMAKE \
            $BLAS_CMAKE \
            $COMPILER_ARG \
            "$SOURCE_PATH"
    else
        eval "$CMAKE_COMMAND" \
            -DBUILD_PPSTEP=ON \
            "$BLAS_ARG" \
            "$ARCH_ARG" \
            "$NAME_ARG" \
            $TRITON_CMAKE \
        $UNITY_BUILD_CMAKE \
            $UNITY_BUILD_CMAKE \
            -DOPENBLAS_PATH="$OPENBLAS_PATH" \
            $MKL_CMAKE \
            $BLAS_CMAKE \
            $COMPILER_ARG \
            "$SOURCE_PATH" >> "$LOG_OUTPUT" 2>&1
    fi

    # Build ppstep
    if [ "$LOG_OUTPUT" == "none" ]; then
        eval "$MAKE_COMMAND" ppstep
    else
        eval "$MAKE_COMMAND" ppstep >> "$LOG_OUTPUT" 2>&1
    fi
    
    print_colored "green" "✅ ppstep build complete"
    print_colored "cyan" "Wrapper created at: blasbuild/$CHIP/ppstep-nd4j"
    echo "Usage: ./blasbuild/$CHIP/ppstep-nd4j <source_file.cpp>"
    
    # Exit early like preprocess does
    exit 0
fi


# Link-only mode: skip compilation, only relink
if [ "$LINK_ONLY" == "ON" ]; then
    print_colored "cyan" "═══════════════════════════════════════════════════════════"
    print_colored "cyan" "LINK-ONLY MODE: Skipping compilation, relinking only"
    print_colored "cyan" "═══════════════════════════════════════════════════════════"

    # Check if build directory and objects exist
    if [ ! -d "blasbuild/$CHIP" ]; then
        print_colored "red" "❌ ERROR: Build directory blasbuild/$CHIP does not exist"
        print_colored "yellow" "   You must run a full build first before using --link-only"
        exit 1
    fi

    builtin cd blasbuild/$CHIP

    # Check if object files exist
    OBJ_COUNT=$(find . -name "*.o" 2>/dev/null | wc -l)
    if [ "$OBJ_COUNT" -eq 0 ]; then
        print_colored "red" "❌ ERROR: No object files found in blasbuild/$CHIP"
        print_colored "yellow" "   You must run a full build first before using --link-only"
        builtin cd ../..
        exit 1
    fi

    print_colored "green" "✓ Found $OBJ_COUNT compiled object files"
    print_colored "cyan" "Building library target only (no compilation)..."

    # Determine the library target name based on chip type
    if [ "$CHIP" == "cpu" ]; then
        LINK_TARGET="nd4jcpu"
    elif [ "$CHIP" == "cuda" ]; then
        LINK_TARGET="nd4jcuda"
    elif [ "$CHIP" == "tpu" ]; then
        LINK_TARGET="nd4jtpu"
    else
        LINK_TARGET="nd4j${CHIP}"
    fi

    print_colored "cyan" "Linking target: $LINK_TARGET"

    # Run make with the specific target (CMake will skip up-to-date compilation)
    if [ "$LOG_OUTPUT" == "none" ]; then
        eval "$MAKE_COMMAND" "$LINK_TARGET" && builtin cd ../../..
    else
        eval "$MAKE_COMMAND" "$LINK_TARGET" >> "$LOG_OUTPUT" 2>&1 && builtin cd ../../..
    fi

    LINK_EXIT_CODE=$?

    if [ $LINK_EXIT_CODE -eq 0 ]; then
        print_colored "green" "═══════════════════════════════════════════════════════════"
        print_colored "green" "✅ LINK-ONLY BUILD COMPLETE"
        print_colored "green" "═══════════════════════════════════════════════════════════"
        print_colored "cyan" "Library: blasbuild/$CHIP/lib${LINK_TARGET}.so"
        print_colored "yellow" "Note: Only linking was performed. No source files were recompiled."
    else
        print_colored "red" "═══════════════════════════════════════════════════════════"
        print_colored "red" "❌ LINK FAILED (exit code: $LINK_EXIT_CODE)"
        print_colored "red" "═══════════════════════════════════════════════════════════"
        exit $LINK_EXIT_CODE
    fi

    # Continue with post-build steps (flatc copy if needed)
    # Don't exit here - let the script continue to the flatc section
else
    # Normal build mode: full compilation and linking
    BUILD_EXIT_CODE=0

    # =========================================================================
    # POST-CMAKE FIXUP: Strip MSVC PDB flags from Ninja build files on Windows
    # =========================================================================
    # CMake's Ninja generator injects -Xcompiler=-FdPath\,-FS into CUDA compile
    # rules. nvcc misparses the backslash-comma, treating -FS as a second input
    # file → "nvcc fatal: A single input file is required". We strip these flags
    # from build.ninja and all .rsp files after cmake configure completes.
    if [[ "$CHIP" == "cuda" ]] && [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -o 2>/dev/null)" == "Msys" ]]; then
        if [ -f "build.ninja" ]; then
            print_colored "cyan" "Patching Ninja build files: removing MSVC PDB flags from CUDA rules..."
            # Count matches before patching for diagnostic output
            MATCH_COUNT=$(grep -c 'Xcompiler.*-Fd\|Xcompiler.*\/Fd\|Xcompiler.*\/FS' build.ninja 2>/dev/null || echo 0)
            # Remove -Xcompiler=-Fd...,-FS patterns (backslash-comma form)
            sed -i 's/ -Xcompiler=-Fd[^ ]*,-FS//g' build.ninja 2>/dev/null || true
            # Remove -Xcompiler=/Fd... patterns (forward-slash form)
            sed -i 's/ -Xcompiler=\/Fd[^ ]*//g' build.ninja 2>/dev/null || true
            # Remove standalone -Xcompiler=/FS
            sed -i 's/ -Xcompiler=\/FS//g' build.ninja 2>/dev/null || true
            # Also check rules.ninja if it exists (some cmake versions split rules)
            if [ -f "rules.ninja" ]; then
                sed -i 's/ -Xcompiler=-Fd[^ ]*,-FS//g' rules.ninja 2>/dev/null || true
                sed -i 's/ -Xcompiler=\/Fd[^ ]*//g' rules.ninja 2>/dev/null || true
                sed -i 's/ -Xcompiler=\/FS//g' rules.ninja 2>/dev/null || true
            fi
            REMAINING=$(grep -c 'Xcompiler.*-Fd\|Xcompiler.*\/Fd\|Xcompiler.*\/FS' build.ninja 2>/dev/null || echo 0)
            print_colored "green" "Patched Ninja build files: removed $MATCH_COUNT MSVC PDB flag occurrences ($REMAINING remaining)"
        fi
    fi

    # Auto-disable OOM monitor on Windows/MSYS2: powershell.exe startup is 500ms-2s,
    # and the 1-second polling loop stacks PowerShell processes faster than they exit,
    # deadlocking the build. Linux uses /proc/meminfo (~0ms), macOS uses vm_stat (~1ms).
    # Windows GitHub Actions runners handle OOM at the OS level.
    if [[ "$OOM_KILLER_ENABLED" == "ON" ]]; then
        _oom_platform=$(detect_oom_platform)
        if [[ "$_oom_platform" == "windows" ]]; then
            print_colored "yellow" "OOM monitor auto-disabled on Windows (powershell.exe polling causes deadlocks)"
            OOM_KILLER_ENABLED="OFF"
        fi
    fi

    if [[ "$OOM_KILLER_ENABLED" == "ON" ]]; then
        print_colored "cyan" "═══════════════════════════════════════════════════════════"
        print_colored "cyan" "🛡️  OOM KILLER ENABLED - ACTIVE MEMORY PROTECTION"
        print_colored "cyan" "═══════════════════════════════════════════════════════════"
        print_colored "blue" "System memory threshold: ${OOM_MEMORY_THRESHOLD}%"
        print_colored "blue" "Monitor interval: ${OOM_MONITOR_INTERVAL}s"
        if [[ "$OOM_PROCESS_MAX_MB" -gt 0 ]]; then
            print_colored "blue" "Process tree limit: ${OOM_PROCESS_MAX_MB}MB"
        else
            print_colored "yellow" "Process tree limit: DISABLED (use --oom-process-max-mb to enable)"
        fi
        print_colored "blue" "Velocity threshold: ${OOM_VELOCITY_THRESHOLD}% per 5 seconds"

        local_current_mem=$(get_memory_usage_percent)
        print_colored "blue" "Current memory usage: ${local_current_mem}%"
        echo ""

        # Validate threshold
        if [[ ! "$OOM_MEMORY_THRESHOLD" =~ ^[0-9]+$ ]] || \
           [[ "$OOM_MEMORY_THRESHOLD" -lt 1 ]] || \
           [[ "$OOM_MEMORY_THRESHOLD" -gt 99 ]]; then
            print_colored "red" "❌ ERROR: Invalid OOM memory threshold: $OOM_MEMORY_THRESHOLD"
            print_colored "yellow" "   Must be a number between 1 and 99"
            exit 1
        fi

        # Run build in background with OOM monitoring
        if [ "$LOG_OUTPUT" == "none" ]; then
            eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" &
        else
            eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" >> "$LOG_OUTPUT" 2>&1 &
        fi
        BUILD_PID=$!

        # Start OOM monitor with all parameters
        start_oom_monitor "$OOM_MEMORY_THRESHOLD" "$OOM_MONITOR_INTERVAL" "$BUILD_PID" "$OOM_PROCESS_MAX_MB" "$OOM_VELOCITY_THRESHOLD" "$OOM_CRITICAL_THRESHOLD"

        # Wait for build to complete
        wait "$BUILD_PID"
        BUILD_EXIT_CODE=$?

        # Stop OOM monitor
        stop_oom_monitor

        if [ $BUILD_EXIT_CODE -eq 137 ]; then
            print_colored "red" "═══════════════════════════════════════════════════════════"
            print_colored "red" "❌ BUILD KILLED BY OOM KILLER"
            print_colored "red" "═══════════════════════════════════════════════════════════"
            builtin cd ../../..
            exit 137
        elif [ $BUILD_EXIT_CODE -ne 0 ]; then
            print_colored "red" "❌ BUILD FAILED (exit code: $BUILD_EXIT_CODE)"
            builtin cd ../../..
            exit $BUILD_EXIT_CODE
        fi

        builtin cd ../../..
    else
        # Normal build without OOM monitoring
        if [ "$LOG_OUTPUT" == "none" ]; then
            eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" && builtin cd ../../..
        else
            eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" >> "$LOG_OUTPUT" 2>&1 && builtin cd ../../..
        fi
        BUILD_EXIT_CODE=$?
    fi
fi

# Determine script location
SCRIPT_DIR="$( builtin cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Find the libnd4j directory
if [[ "$SCRIPT_DIR" == */libnd4j* ]]; then
    # Already in or under libnd4j directory
    LIBND4J_DIR=$(echo "$SCRIPT_DIR" | sed 's/\(.*libnd4j\).*/\1/')
else
    # Check if we're in the parent deeplearning4j directory
    if [ -d "$SCRIPT_DIR/libnd4j" ]; then
        LIBND4J_DIR="$SCRIPT_DIR/libnd4j"
    else
        # Try going up one directory to see if libnd4j is there
        if [ -d "$SCRIPT_DIR/../libnd4j" ]; then
            LIBND4J_DIR="$SCRIPT_DIR/../libnd4j"
        else
            echo "Error: Could not locate libnd4j directory"
            exit 1
        fi
    fi
fi

# Save the original directory before potentially changing directories
ORIGINAL_DIR=$(pwd)

# cd to the script directory for the operations that need to be performed there
builtin cd "$SCRIPT_DIR"

if [ "$GENERATE_FLATC" == "ON" ]; then
    echo "Copying flatc generated for java"
    # ensure proper flatc sources are in place
    bash "$LIBND4J_DIR/copy-flatc-java.sh"
fi

# Return to the original directory where the script was launched from
builtin cd "$ORIGINAL_DIR"

echo "Build process completed successfully."
