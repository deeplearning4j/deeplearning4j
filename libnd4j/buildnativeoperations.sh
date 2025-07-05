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
set -eu

# cd to the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


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
export MAKE_COMMAND="make"
export MAKE_ARGUMENTS=
echo eval $CMAKE_COMMAND

[[ -z ${MAKEJ:-} ]] && MAKEJ=4

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
MINIFIER="${MINIFIER:-false}"
TESTS="${TESTS:-false}"
PRINT_INDICES="${PRINT_INDICES:-OFF}"
VERBOSE="${VERBOSE:-true}"
VERBOSE_ARG="${VERBOSE_ARG:-VERBOSE=1}"
HELPER="${HELPER:-}"
CHECK_VECTORIZATION="${CHECK_VECTORIZATION:-OFF}"
NAME="${NAME:-}"
OP_OUTPUT_FILE="${OP_OUTPUT_FILE:-include/generated/include_ops.h}"
USE_LTO="${USE_LTO:-}"
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

# Type validation specific variables
DEBUG_TYPE_PROFILE="${DEBUG_TYPE_PROFILE:-}"
DEBUG_CUSTOM_TYPES="${DEBUG_CUSTOM_TYPES:-}"
DEBUG_AUTO_REDUCE="${DEBUG_AUTO_REDUCE:-true}"
STRICT_TYPE_VALIDATION="${STRICT_TYPE_VALIDATION:-false}"
SD_TYPES_VALIDATED="${SD_TYPES_VALIDATED:-false}"
SD_VALIDATED_TYPES="${SD_VALIDATED_TYPES:-}"

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]
do
    key="$1"
    value="${2:-}"

    case $key in
        # CRITICAL: Type-related arguments
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

print_colored "blue" "\n🔍 PROCESSING TYPE CONFIGURATION"

# CRITICAL FIX: Remember the user's explicit datatypes setting
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
            export MAKE_COMMAND="ninja"
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
            export MAKE_COMMAND="make"
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
fi

if [ -z "$NAME" ]; then
    if [ "$CHIP" == "cpu" ]; then
        NAME="nd4jcpu"
    elif [ "$CHIP" == "cuda" ]; then
        NAME="nd4jcuda"
    elif [ "$CHIP" == "aurora" ]; then
        NAME="nd4jaurora"
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
        echo "Removing blasbuild"
        rm -Rf blasbuild
    fi
    mkdir -p "blasbuild/$CHIP"
    cd "blasbuild/$CHIP"
}

HELPERS=""
if [ "$HELPER" == "" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                 WARNING!                                                  !!"
    echo "!!                                      No helper packages configured!                                       !!"
    echo "!!                          You can specify helper by using -h key. I.e. <-h onednn>                         !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!                                                                                                           !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
else
    # If helpers were defined, we'll propagate them to CMake
    IFS=','
    read -ra HLP <<< "$HELPER"
    for i in "${HLP[@]}"; do
        HELPERS="${HELPERS} -DHELPERS_$i=true"
    done
    IFS=' '
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
echo CHECK_VECTORIZATION = "$CHECK_VECTORIZATION"
echo HELPERS             = "$HELPERS"
echo OP_OUTPUT_FILE      = "$OP_OUTPUT_FILE"
echo USE_LTO             = "$USE_LTO"
echo SANITIZE            = "$SANITIZE"
echo FUNC_TRACE          = "$FUNC_TRACE"
echo LOG_OUTPUT          = "$LOG_OUTPUT"
echo KEEP_NVCC           = "$KEEP_NVCC"
echo PRINT_INDICES       = "$PRINT_INDICES"
echo PRINT_MATH          = "$PRINT_MATH"
echo PREPROCESS          = "$PREPROCESS"

mkbuilddir
pwd

# ----------------------- CMake Configuration -----------------------

# Configure CMake
echo "$CMAKE_COMMAND - -DSD_KEEP_NVCC_OUTPUT=$KEEP_NVCC -DSD_GCC_FUNCTRACE=$FUNC_TRACE $BLAS_ARG $ARCH_ARG $NAME_ARG $OP_OUTPUT_FILE_ARG -DSD_SANITIZERS=${SANITIZERS} -DSD_SANITIZE=${SANITIZE} -DSD_CHECK_VECTORIZATION=${CHECK_VECTORIZATION} $USE_LTO $HELPERS $SHARED_LIBS_ARG $MINIFIER_ARG $OPERATIONS_ARG $DATATYPES_ARG $BUILD_TYPE $PACKAGING_ARG $EXPERIMENTAL_ARG $TESTS_ARG $CUDA_COMPUTE -DOPENBLAS_PATH=$OPENBLAS_PATH -DDEV=FALSE -DCMAKE_NEED_RESPONSE=YES -DMKL_MULTI_THREADED=TRUE ../.."

if [ "$LOG_OUTPUT" == "none" ]; then
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
        -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
        -DSD_PTXAS="$PTXAS_INFO" \
        "$BLAS_ARG" \
        "$ARCH_ARG" \
        "$NAME_ARG" \
        "$OP_OUTPUT_FILE_ARG" \
        -DSD_SANITIZE="${SANITIZE}" \
        -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
        "$USE_LTO" \
        "$HELPERS" \
        "$SHARED_LIBS_ARG" \
        "$MINIFIER_ARG" \
        "$OPERATIONS_ARG" \
        "$DATATYPES_ARG" \
        "$BUILD_TYPE" \
        "$PACKAGING_ARG" \
        "$TESTS_ARG" \
        "$CUDA_COMPUTE" \
        -DOPENBLAS_PATH="$OPENBLAS_PATH" \
        -DDEV=FALSE \
        -DCMAKE_NEED_RESPONSE=YES \
        -DMKL_MULTI_THREADED=TRUE \
        ../..
else
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
        -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
        -DSD_PTXAS="$PTXAS_INFO" \
        "$BLAS_ARG" \
        "$ARCH_ARG" \
        "$NAME_ARG" \
        "$OP_OUTPUT_FILE_ARG" \
        -DSD_SANITIZE="${SANITIZE}" \
        -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
        "$USE_LTO" \
        "$HELPERS" \
        "$SHARED_LIBS_ARG" \
        "$MINIFIER_ARG" \
        "$OPERATIONS_ARG" \
        "$DATATYPES_ARG" \
        "$BUILD_TYPE" \
        "$PACKAGING_ARG" \
        "$TESTS_ARG" \
        "$CUDA_COMPUTE" \
        -DOPENBLAS_PATH="$OPENBLAS_PATH" \
        -DDEV=FALSE \
        -DCMAKE_NEED_RESPONSE=YES \
        -DMKL_MULTI_THREADED=TRUE \
        ../.. >> "$LOG_OUTPUT" 2>&1
fi

# ----------------------- Preprocessing Step -----------------------

# Handle the PREPROCESS flag
if [ "$PREPROCESS" == "ON" ]; then
   if [ "$LOG_OUTPUT" == "none" ]; then
       eval "$CMAKE_COMMAND" \
           -DPRINT_MATH="$PRINT_MATH" \
           -DPRINT_INDICES="$PRINT_INDICES" \
           -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
           -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
            -DSD_PREPROCESS="$PREPROCESS" \
           -DSD_PTXAS="$PTXAS_INFO" \
           "$BLAS_ARG" \
           "$ARCH_ARG" \
           "$NAME_ARG" \
           "$OP_OUTPUT_FILE_ARG" \
           -DSD_SANITIZE="${SANITIZE}" \
           "$USE_LTO" \
           "$HELPERS" \
           "$SHARED_LIBS_ARG" \
           "$MINIFIER_ARG" \
           "$OPERATIONS_ARG" \
           "$DATATYPES_ARG" \
           "$BUILD_TYPE" \
           "$PACKAGING_ARG" \
           "$TESTS_ARG" \
           "$CUDA_COMPUTE" \
           -DOPENBLAS_PATH="$OPENBLAS_PATH" \
           -DDEV=FALSE \
           -DCMAKE_NEED_RESPONSE=YES \
           -DMKL_MULTI_THREADED=TRUE \
           ../..
   else
       eval "$CMAKE_COMMAND" \
           -DPRINT_MATH="$PRINT_MATH" \
           -DPRINT_INDICES="$PRINT_INDICES" \
           -DSD_KEEP_NVCC_OUTPUT="$KEEP_NVCC" \
           -DSD_GCC_FUNCTRACE="$FUNC_TRACE" \
           -DSD_PREPROCESS="$PREPROCESS" \
           "$BLAS_ARG" \
           "$ARCH_ARG" \
           "$NAME_ARG" \
           "$OP_OUTPUT_FILE_ARG" \
           -DSD_PTXAS="$PTXAS_INFO" \
           -DSD_SANITIZE="${SANITIZE}" \
           -DSD_CHECK_VECTORIZATION="${CHECK_VECTORIZATION}" \
           "$USE_LTO" \
           "$HELPERS" \
           "$SHARED_LIBS_ARG" \
           "$MINIFIER_ARG" \
           "$OPERATIONS_ARG" \
           "$DATATYPES_ARG" \
           "$BUILD_TYPE" \
           "$PACKAGING_ARG" \
           "$TESTS_ARG" \
           "$CUDA_COMPUTE" \
           -DOPENBLAS_PATH="$OPENBLAS_PATH" \
           -DDEV=FALSE \
           -DCMAKE_NEED_RESPONSE=YES \
           -DMKL_MULTI_THREADED=TRUE \
           ../.. >> "$LOG_OUTPUT" 2>&1
   fi
    echo "Running preprocessing step..."
    exit 0
fi

# --------------------- End of Preprocessing Step ----------------------

# Set Make arguments based on user options
if [ "$PARALLEL" == "true" ]; then
    MAKE_ARGUMENTS="$MAKE_ARGUMENTS -j $MAKEJ"
fi
if [ "$VERBOSE" == "true" ]; then
    MAKE_ARGUMENTS="$MAKE_ARGUMENTS $VERBOSE_ARG"
fi

# Build the project


if [ "$LOG_OUTPUT" == "none" ]; then
    eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" && cd ../../..
else
    eval "$MAKE_COMMAND" "$MAKE_ARGUMENTS" >> "$LOG_OUTPUT" 2>&1 && cd ../../..
fi

# Determine script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
cd "$SCRIPT_DIR"

if [ "$GENERATE_FLATC" == "ON" ]; then
    echo "Copying flatc generated for java"
    # ensure proper flatc sources are in place
    bash "$LIBND4J_DIR/copy-flatc-java.sh"
fi

# Return to the original directory where the script was launched from
cd "$ORIGINAL_DIR"

echo "Build process completed successfully."