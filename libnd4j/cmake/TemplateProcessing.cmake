################################################################################
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

# TemplateProcessing.cmake - Complete Rewrite with Comprehensive Type Handling
# This version generates ALL type variants to prevent undefined references
#
# ═══════════════════════════════════════════════════════════════
# BINARY SIZE WARNING: Template instantiations create MASSIVE binaries
# ═══════════════════════════════════════════════════════════════
#
# MEASURED SIZES:
# - Normal build (no functrace): ~200MB
# - Functrace build (SD_GCC_FUNCTRACE=ON): ~3.3GB (!)
#
# ROOT CAUSE: Template instantiations for ALL type combinations
# - Each operation × each type combination = thousands of instantiations
# - With debug symbols + functrace: exceeds 2GB relocation limit
#
# HOW TO REDUCE BINARY SIZE:
# 1. **Disable functrace**: -Dlibnd4j.calltrace=OFF (90% reduction)
# 2. **Reduce types**: Use -Dlibnd4j.datatypes="float32;double;int32;int64"
# 3. **Split files**: Chunk sizes below control file splitting
# 4. **Selective rendering**: Enable SD_SELECTIVE_TYPES in types.h
#
# For debugging without functrace: Use GDB with core dumps (see docs)
# ═══════════════════════════════════════════════════════════════

set(CUSTOMOPS_GENERIC_SOURCES "" CACHE INTERNAL "Template-generated source files")

# Chunk sizing for functrace builds: adapt chunk sizes so total compile memory is roughly 60% of
# system RAM, even when high parallelism (buildthreads ~14) is used.
set(USE_MULTI_PASS_GENERATION "ON" CACHE STRING "Use multi-pass generation (ON/OFF/AUTO)")

if(SD_GCC_FUNCTRACE)
    set(_functrace_jobs 0)
    if(DEFINED CMAKE_BUILD_PARALLEL_LEVEL AND NOT CMAKE_BUILD_PARALLEL_LEVEL STREQUAL "")
        set(_functrace_jobs ${CMAKE_BUILD_PARALLEL_LEVEL})
    elseif(DEFINED SD_PARALLEL_COMPILE_JOBS AND NOT SD_PARALLEL_COMPILE_JOBS STREQUAL "0")
        set(_functrace_jobs ${SD_PARALLEL_COMPILE_JOBS})
    endif()
    if(_functrace_jobs LESS 1)
        include(ProcessorCount)
        ProcessorCount(_functrace_detected_jobs)
        if(_functrace_detected_jobs GREATER 0)
            set(_functrace_jobs ${_functrace_detected_jobs})
        else()
            set(_functrace_jobs 4)
        endif()
    endif()

    cmake_host_system_information(RESULT _functrace_total_kb QUERY TOTAL_PHYSICAL_MEMORY)
    if(_functrace_total_kb)
        math(EXPR _functrace_total_mb "${_functrace_total_kb} / 1024")
    else()
        set(_functrace_total_mb 0)
    endif()
    if(_functrace_total_mb LESS 1024)
        set(_functrace_total_mb 1024)
    endif()

    # PERFORMANCE OPTIMIZATION: Use 75% of RAM for faster builds (increased from 60%)
    # Assumes system has adequate swap if needed
    math(EXPR _functrace_budget_mb "${_functrace_total_mb} * 3 / 4")
    math(EXPR _per_job_mb "${_functrace_budget_mb} / ${_functrace_jobs}")
    if(_per_job_mb LESS 512)
        set(_per_job_mb 512)
    endif()

    # Note: We now count actual template lines, not combinations
    # REDUCED from 30 to 3 for functrace+CUDA builds to avoid .eh_frame PC32 relocation overflow
    # AND reduce memory usage during compilation (each CUDA compilation uses 500MB-1GB RAM)
    # Each template instantiation generates ~200+ symbols due to template expansion, inline functions, etc.
    # With 30 instantiations per file, we get 6000+ symbols which exceeds PC32 relocation limits
    # Reduced to 3 to allow more parallel jobs without OOM
    set(_chunk_target 3)
    set(_direct_chunk 3)

    set(CHUNK_TARGET_INSTANTIATIONS "${_chunk_target}" CACHE STRING "Adaptive chunk size for functrace builds" FORCE)
    set(MULTI_PASS_CHUNK_SIZE "${_direct_chunk}" CACHE STRING "Adaptive direct chunk size for functrace builds" FORCE)
    if(NOT DEFINED CHUNK_MAX_INSTANTIATIONS OR CHUNK_MAX_INSTANTIATIONS LESS CHUNK_TARGET_INSTANTIATIONS)
        set(CHUNK_MAX_INSTANTIATIONS "${CHUNK_TARGET_INSTANTIATIONS}" CACHE STRING "Maximum template instantiations per chunk file (aligned with target)" FORCE)
    endif()

    message(STATUS "Functrace chunk sizing: target=${CHUNK_TARGET_INSTANTIATIONS} template lines, jobs=${_functrace_jobs}, per-job-limit≈${_per_job_mb}MB (75% of ${_functrace_total_mb}MB)")
    message(STATUS "⚠️  WARNING: Functrace builds still create ~3GB binaries (may exceed linker limits)")
else()
    if(NOT DEFINED CHUNK_TARGET_INSTANTIATIONS)
        set(CHUNK_TARGET_INSTANTIATIONS "5" CACHE STRING "Target template instantiations per chunk file (1-20)")
    endif()
    if(NOT DEFINED CHUNK_MAX_INSTANTIATIONS)
        set(CHUNK_MAX_INSTANTIATIONS "10" CACHE STRING "Maximum template instantiations per chunk file")
    endif()
    # Force chunk size to 10 for CUDA builds - 30 templates = ~8min, 10 = ~2.5min per file
    set(MULTI_PASS_CHUNK_SIZE "10" CACHE STRING "Chunk size for direct instantiation files" FORCE)
endif()

# Enable selective rendering diagnostics in debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SRCORE_ENABLE_DIAGNOSTICS ON)
endif()

# ============================================================================
# COMPREHENSIVE TYPE EQUIVALENCE SYSTEM
# ============================================================================

# Get ALL type variants for a given type - returns complete equivalence class
function(get_all_type_variants type all_variants_var)
    set(all_variants "")
    string(STRIP "${type}" normalized_type)
    
    # Special handling for float16/half types
    if("${normalized_type}" MATCHES "float16|half|__half")
        # Only use float16 in all cases - never use __half or half as those are CUDA-specific
        set(all_variants "float16")
        set(${all_variants_var} "${all_variants}" PARENT_SCOPE)
        return()
    endif()
    
    # 8-bit signed integers
    set(INT8_CLASS "int8_t;SignedChar;signed char;char")
    foreach(variant ${INT8_CLASS})
        if("${normalized_type}" STREQUAL "${variant}")
            set(all_variants ${INT8_CLASS})
            break()
        endif()
    endforeach()
    
    # 8-bit unsigned integers
    set(UINT8_CLASS "uint8_t;UnsignedChar;unsigned char")
    if(NOT all_variants)
        foreach(variant ${UINT8_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${UINT8_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # 16-bit signed integers  
    set(INT16_CLASS "int16_t;short;short int;signed short")
    if(NOT all_variants)
        foreach(variant ${INT16_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${INT16_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # 16-bit unsigned integers
    set(UINT16_CLASS "uint16_t;unsigned short;unsigned short int")
    if(NOT all_variants)
        foreach(variant ${UINT16_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${UINT16_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # 32-bit signed integers
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(INT32_CLASS "int32_t;int;signed int;Int32Type")
    else()
        set(INT32_CLASS "int32_t;int;signed int;long;Int32Type")
    endif()
    if(NOT all_variants)
        foreach(variant ${INT32_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${INT32_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # 32-bit unsigned integers
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(UINT32_CLASS "uint32_t;unsigned int;unsigned")
    else()
        set(UINT32_CLASS "uint32_t;unsigned int;unsigned;unsigned long")
    endif()
    if(NOT all_variants)
        foreach(variant ${UINT32_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${UINT32_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # 64-bit signed integers
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(INT64_CLASS "int64_t;long long;long;sd::LongType;LongType")
    else()
        set(INT64_CLASS "int64_t;long long;sd::LongType;LongType")
    endif()
    if(NOT all_variants)
        foreach(variant ${INT64_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${INT64_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # 64-bit unsigned integers
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(UINT64_CLASS "uint64_t;unsigned long long;unsigned long;sd::UnsignedLong;UnsignedLong;size_t")
    else()
        set(UINT64_CLASS "uLongType;unsigned long long;sd::UnsignedLong;UnsignedLong")
    endif()
    if(NOT all_variants)
        foreach(variant ${UINT64_CLASS})
            if("${normalized_type}" STREQUAL "${variant}")
                set(all_variants ${UINT64_CLASS})
                break()
            endif()
        endforeach()
    endif()
    
    # Non-aliased types
    if(NOT all_variants)
        if("${normalized_type}" STREQUAL "float")
            set(all_variants "float")
        elseif("${normalized_type}" STREQUAL "double")
            set(all_variants "double")
        elseif("${normalized_type}" STREQUAL "bfloat16")
            set(all_variants "bfloat16")
        elseif("${normalized_type}" STREQUAL "bool")
            set(all_variants "bool")
        else()
            # Unknown type - use as-is
            set(all_variants "${normalized_type}")
        endif()
    endif()
    
    # Remove duplicates
    if(all_variants)
        list(REMOVE_DUPLICATES all_variants)
    endif()
    
    set(${all_variants_var} "${all_variants}" PARENT_SCOPE)
endfunction()


# Get all types that should have cross-type conversions
function(get_conversion_capable_types types_var)
    set(types "bool;int8_t;uint8_t;int16_t;uint16_t;int32_t;uint32_t;int64_t;uint64_t;float16;bfloat16;float;double")    # Add common aliases that might appear
    list(APPEND types "SignedChar;UnsignedChar;short;unsigned short;int;unsigned int;long;unsigned long;long long;unsigned long long")
    set(${types_var} "${types}" PARENT_SCOPE)
endfunction()

# Deduplication helper
function(add_unique_instantiation instantiation dedupe_set_var content_var)
    set(dedupe_set ${${dedupe_set_var}})
    set(content "${${content_var}}")
    
    list(FIND dedupe_set "${instantiation}" found_idx)
    if(found_idx EQUAL -1)
        string(APPEND content "${instantiation}\n")
        list(APPEND dedupe_set "${instantiation}")
    endif()
    
    set(${dedupe_set_var} "${dedupe_set}" PARENT_SCOPE)
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Memory configuration
# ============================================================================

function(configure_memory_chunking)
    # CLEAN BUILD FIX: Chunking configuration is now set in Options.cmake based on build flags
    # This eliminates the need for CACHE FORCE which only worked for incremental builds
    # For clean builds, the correct values are set FROM THE START in Options.cmake

    # This function now only handles:
    # 1. Validation of the configured values
    # 2. AUTO mode for USE_MULTI_PASS_GENERATION

    # Validate chunking configuration
    message(STATUS "📊 Using chunking configuration: CHUNK_TARGET=${CHUNK_TARGET_INSTANTIATIONS}, MULTI_PASS=${MULTI_PASS_CHUNK_SIZE}")

    # Handle USE_MULTI_PASS_GENERATION AUTO mode
    if(USE_MULTI_PASS_GENERATION STREQUAL "AUTO")
        cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
        if(AVAILABLE_MEMORY LESS 3000 OR DEFINED ENV{CI})
            set(USE_MULTI_PASS_GENERATION ON CACHE STRING "Use multi-pass generation" FORCE)
            message(STATUS "Auto-detected low memory/CI: enabling multi-pass")
        else()
            set(USE_MULTI_PASS_GENERATION OFF CACHE STRING "Use multi-pass generation" FORCE)
        endif()
    endif()

    set(CHUNK_TARGET_INSTANTIATIONS_SET TRUE CACHE INTERNAL "Memory auto-detection completed")
endfunction()

# ============================================================================
# Main entry point
# ============================================================================

function(execute_template_processing_with_selective_rendering)
    message(STATUS "🔧 Starting Template Processing with Comprehensive Type Handling...")

    # Step 1: Include selective rendering core
    include(${CMAKE_CURRENT_LIST_DIR}/SelectiveRenderingCore.cmake OPTIONAL RESULT_VARIABLE SR_FOUND)
    
    # Step 2: Setup selective rendering if available
    if(SR_FOUND)
        setup_selective_rendering_unified_safe()
        
        # Verify selective rendering worked
        if(NOT DEFINED UNIFIED_COMBINATIONS_2 OR NOT DEFINED UNIFIED_COMBINATIONS_3)
            message(WARNING "⚠️ Selective rendering failed to initialize, falling back to full matrix")
            setup_fallback_combinations()
        endif()
    else()
        message(STATUS "⚠️ Selective rendering not available, using full type matrix")
        setup_fallback_combinations()
    endif()

    # Step 3: Configure memory chunking
    configure_memory_chunking()

    # Step 4: Display filtering results if available
    if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
        list(LENGTH UNIFIED_COMBINATIONS_2 filtered_2_count)
        list(LENGTH UNIFIED_COMBINATIONS_3 filtered_3_count)
        message(STATUS "🎯 Selective Rendering Results:")
        message(STATUS "   - 2-type combinations: ${filtered_2_count}")
        message(STATUS "   - 3-type combinations: ${filtered_3_count}")
    endif()

    message(STATUS "🔧 Template processing: Multi-pass=${USE_MULTI_PASS_GENERATION}, Chunks=${CHUNK_TARGET_INSTANTIATIONS}")

    # Step 5: Execute template generation
    execute_template_generation()
    
    # Step 6: Propagate results to parent scope
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

# ============================================================================
# Template handlers with comprehensive type coverage
# ============================================================================

# Handler for specials_single templates
function(handle_specials_single t1 content_var)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)
    
    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    string(APPEND content "namespace sd {\n")
    
    # Instantiate only the specific member functions needed (class instantiation would duplicate these)
    add_unique_instantiation("template void sd::SpecialMethods<${t1}>::concatCpuGeneric(const std::vector<NDArray*>&, NDArray&, const sd::LongType);" dedupe_set content)
    add_unique_instantiation("template void sd::SpecialMethods<${t1}>::sortGeneric(NDArray*, bool);" dedupe_set content)
    add_unique_instantiation("template void sd::SpecialMethods<${t1}>::sortTadGeneric(NDArray*, sd::LongType*, int, bool);" dedupe_set content)

    string(APPEND content "}\n")
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for specials_double templates - includes cross-type conversions
function(handle_specials_double t1 t2 content_var)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Map C++ types to their canonical form to detect same-type conversions
    normalize_to_canonical_type("${t1}" canonical_t1)
    normalize_to_canonical_type("${t2}" canonical_t2)
    
    # Skip if they're the same type after normalization
    if(canonical_t1 STREQUAL canonical_t2)
        # Same type - no TypeCast needed, only sorting operations
        string(APPEND content "namespace sd {\n")
        
        # DoubleMethods for sorting (same-type sorting is valid)
        add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortByKey(sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
        add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortByValue(sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
        add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortTadByKey(sd::NDArray*, sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
        add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortTadByValue(sd::NDArray*, sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
        
        string(APPEND content "}\n")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Rest of validation for actual type conversions...
    # [existing validation code]
    
    string(APPEND content "namespace sd {\n")

    # Only generate SpecialTypeConverter for actual conversions - using sd::LongType as canonical form
    # Note: This is SpecialTypeConverter (from specials.h), NOT TypeCast (from type_conversions.h)
    add_unique_instantiation("template void SpecialTypeConverter::convertGeneric<${t1}, ${t2}>(sd::Pointer*, void*, sd::LongType, void*);" dedupe_set content)

    # DoubleMethods instantiations for sorting operations
    add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortByKey(sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
    add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortByValue(sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
    add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortTadByKey(sd::NDArray*, sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
    add_unique_instantiation("template void DoubleMethods<${t1}, ${t2}>::sortTadByValue(sd::NDArray*, sd::NDArray*, sd::NDArray*, bool);" dedupe_set content)
    
    string(APPEND content "}\n")
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


function(normalize_to_canonical_type cpp_type canonical_var)
    # Use a local variable to avoid modifying the input parameter directly
    set(local_cpp_type "${cpp_type}")

    # Strip any namespace prefixes (e.g., sd::)
    string(REGEX REPLACE "^[a-zA-Z0-9_]+::" "" local_cpp_type "${local_cpp_type}")

    # 1. DEFINE TYPE ALIASES
    # ========================
    # Group all possible spellings for each canonical type in lists.
    set(types_int64   "int64_t" "long long" "LongType")
    set(types_uint64  "uint64_t" "unsigned long long" "UnsignedLong")
    set(types_int32   "int32_t" "int" "Int32Type" "signed int" "signed")
    set(types_uint32  "uint32_t" "unsigned int" "unsigned")
    set(types_int16   "int16_t" "short" "short int" "signed short" "signed short int")
    set(types_uint16  "uint16_t" "unsigned short" "unsigned short int")
    set(types_int8    "int8_t" "signed char" "char" "SignedChar")
    set(types_uint8   "uint8_t" "unsigned char" "UnsignedChar")
    set(types_float16 "float16" "half" "__half")
    set(types_bfloat16 "bfloat16")
    set(types_float   "float")
    set(types_double  "double")
    set(types_bool    "bool")
    set(types_string  "std::string" "utf8")
    set(types_u16string "std::u16string" "utf16")
    set(types_u32string "std::u32string" "utf32")

    # 2. HANDLE PLATFORM-DEPENDENT TYPES
    # ==================================
    # The 'long' type is 32-bit on 32-bit systems and some 64-bit systems (like Windows),
    # but 64-bit on most other 64-bit systems (like Linux).
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # For 64-bit architectures, assume 'long' is 64-bit.
        list(APPEND types_int64 "long")
        list(APPEND types_uint64 "unsigned long")
    else()
        # For 32-bit architectures, 'long' is 32-bit.
        list(APPEND types_int32 "long")
        list(APPEND types_uint32 "unsigned long")
    endif()

    # 3. BUILD REGEX PATTERNS FROM ALIASES
    # ======================================
    # This creates a regex like "^(alias1|alias2|alias3)$" for each type list.
    foreach(type_name "int64" "uint64" "int32" "uint32" "int16" "uint16" "int8" "uint8" "float16" "bfloat16" "float" "double" "bool" "string" "u16string" "u32string")
        string(REPLACE ";" "|" patterns_${type_name} "${types_${type_name}}")
        set(patterns_${type_name} "^(${patterns_${type_name}})$")
    endforeach()

    # 4. MATCH AND SET CANONICAL TYPE
    # =================================
    set(canonical "")
    if(local_cpp_type MATCHES "${patterns_int64}")
        set(canonical "sd::LongType")
    elseif(local_cpp_type MATCHES "${patterns_uint64}")
        set(canonical "sd::UnsignedLong")
    elseif(local_cpp_type MATCHES "${patterns_int32}")
        set(canonical "int32_t")
    elseif(local_cpp_type MATCHES "${patterns_uint32}")
        set(canonical "uint32_t")
    elseif(local_cpp_type MATCHES "${patterns_int16}")
        set(canonical "int16_t")
    elseif(local_cpp_type MATCHES "${patterns_uint16}")
        set(canonical "uint16_t")
    elseif(local_cpp_type MATCHES "${patterns_int8}")
        set(canonical "int8_t")
    elseif(local_cpp_type MATCHES "${patterns_uint8}")
        set(canonical "uint8_t")
    elseif(local_cpp_type MATCHES "${patterns_float16}")
        set(canonical "float16")
    elseif(local_cpp_type MATCHES "${patterns_bfloat16}")
        set(canonical "bfloat16")
    elseif(local_cpp_type MATCHES "${patterns_float}")
        set(canonical "float")
    elseif(local_cpp_type MATCHES "${patterns_double}")
        set(canonical "double")
    elseif(local_cpp_type MATCHES "${patterns_bool}")
        set(canonical "bool")
    elseif(local_cpp_type MATCHES "${patterns_string}")
        set(canonical "std::string")
    elseif(local_cpp_type MATCHES "${patterns_u16string}")
        set(canonical "std::u16string")
    elseif(local_cpp_type MATCHES "${patterns_u32string}")
        set(canonical "std::u32string")
    else()
        # Default to the original (namespace-stripped) type if no match is found
        set(canonical "${local_cpp_type}")
    endif()

    set(${canonical_var} "${canonical}" PARENT_SCOPE)
endfunction()

# NOTE: First handle_random definition - subsequent definitions override this
function(handle_random t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)
    
    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void const* vy, sd::LongType const* yShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::execTransform(int opNum, sd::Pointer state, void* z, sd::LongType const* zShapeBuffer, void* extraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::execTransform(int opNum, sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::execTransform(int opNum, sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y, sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

function(handle_reduce_same t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Generate instantiations for the actual type used (including 'long')
    # Don't skip aliases because NativeOpExecutioner.cpp uses 'long' explicitly
    if(is_cuda)
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* dXShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, const sd::LongType* dims);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* zShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);" dedupe_set content)
        add_unique_instantiation("template ${t1} functions::reduce::ReduceSameFunction<${t1}>::execScalarCudaLegacy(int opNum, void const* vx, sd::LongType const* xShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::exec(int opNum, sd::memory::Workspace* workspace, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo, sd::LongType const* dimension);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::execScalar(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template ${t1} functions::reduce::ReduceSameFunction<${t1}>::execScalar(int opNum, const void* x, const sd::LongType* xShapeInfo, void* extraParams);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()



# Handler for broadcast_bool templates

function(handle_broadcast_bool t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Only process if t2 is bool
    if(NOT "${t2}" STREQUAL "bool")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Normalize t1
    normalize_to_canonical_type("${t1}" norm_t1)
    
    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    if(type1_enum STREQUAL "")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Validate pair
    _internal_srcore_is_valid_pair("${type1_enum}" "BOOL" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::broadcast::BroadcastBool<${t1}, bool>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, void const *x, sd::LongType const *xShapeInfo, void const *y, sd::LongType const *yShapeInfo, void *result, sd::LongType const *resultShapeInfo, void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType const *tadOnlyShapeInfo, sd::LongType const *tadOffsets, sd::LongType const *tadOnlyShapeInfoZ, sd::LongType const *tadOffsetsZ);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastBool<${t1}, bool>::execBroadcast(dim3 launchDims, cudaStream_t *stream, const int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo, void *extraParams);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastBool<${t1}, bool>::execInverseBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::broadcast::BroadcastBool<${t1}, bool>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastBool<${t1}, bool>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo, void *extraParams);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastBool<${t1}, bool>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


# Handler for broadcast (3-type) templates
function(handle_broadcast_int t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    if(is_cuda)
        add_unique_instantiation("template class BroadcastInt<${t1}>;" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::broadcast::BroadcastInt<${t1}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastInt<${t1}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastInt<${t1}>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for pairwise templates



# Handler for scalar templates
function(handle_scalar t1 t2 t3 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    normalize_to_canonical_type("${t3}" norm_t3)

    # Skip any combination where ANY type isn't in its canonical form
    # This prevents duplicates like (unsigned short int, uint16_t, uint32_t) when (uint16_t, uint16_t, uint32_t) exists
    if(NOT (t1 STREQUAL norm_t1 AND t2 STREQUAL norm_t2 AND t3 STREQUAL norm_t3))
        # At least one type is a non-canonical alias - skip to avoid duplicates
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    map_cpp_to_enum("${norm_t3}" type3_enum)

    # Validation
    _internal_srcore_is_valid_pair("${type1_enum}" "${type3_enum}" valid_13)
    _internal_srcore_is_valid_pair("${type2_enum}" "${type3_enum}" valid_23)
    _internal_srcore_is_valid_triple("${type1_enum}" "${type2_enum}" "${type3_enum}" valid_triple)

    if(NOT valid_13 OR NOT valid_23 OR NOT valid_triple)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations using NORMALIZED types (which are same as originals now due to check above)
    if(is_cuda)
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, const sd::LongType *hxShapeInfo, void *vz, const sd::LongType *zShapeInfo, const sd::LongType *hzShapeInfo, const void *vscalar, void *vextraParams);" dedupe_set content)
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, void *vz, const sd::LongType *zShapeInfo, const void *vscalars, void *vextraParams, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *result, const sd::LongType *resultShapeInfo, const void *scalar, void *extraParams, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z, const sd::LongType *zShapeInfo, const void *scalars, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for random templates

function(handle_random t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)
    
    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void const* vy, sd::LongType const* yShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::execTransform(int opNum, sd::Pointer state, void* z, sd::LongType const* zShapeBuffer, void* extraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::execTransform(int opNum, sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments);" dedupe_set content)
        add_unique_instantiation("template void functions::random::RandomFunction<${t1}>::execTransform(int opNum, sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y, sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for reduce_same templates

function(handle_reduce_same t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Generate ONLY for types.h typedefs (sd::LongType, sd::UnsignedLong)
    if(is_cuda)
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* dXShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, const sd::LongType* dims);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* zShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);" dedupe_set content)
        add_unique_instantiation("template ${t1} functions::reduce::ReduceSameFunction<${t1}>::execScalarCudaLegacy(int opNum, void const* vx, sd::LongType const* xShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::exec(int opNum, sd::memory::Workspace* workspace, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo, sd::LongType const* dimension);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceSameFunction<${t1}>::execScalar(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template ${t1} functions::reduce::ReduceSameFunction<${t1}>::execScalar(int opNum, const void* x, const sd::LongType* xShapeInfo, void* extraParams);" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


# Handler for reduce3 templates
function(handle_reduce3 t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    if(is_cuda)
        # Existing CUDA instantiations - these are correct
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationPointer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo, sd::LongType const* yTadOffsets);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execAll(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationPointer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo, sd::LongType const* yTadOffsets);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execScalar(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* allocationPointer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo);" dedupe_set content)
    else()
        # CPU instantiations - all signatures from the implementation file
        
        # exec with 11 parameters (dimension, start, stop)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType start, sd::LongType stop);" dedupe_set content)
        
        # exec with 12 parameters (includes tadShapeInfo and tadOffsets)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, sd::LongType start, sd::LongType stop);" dedupe_set content)
        
        # execScalar
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execScalar(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo);" dedupe_set content)
        
        # execAll with full TAD information for both x and y
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execAll(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* xTadShapeInfo, sd::LongType const* xOffsets, sd::LongType const* yTadShapeInfo, sd::LongType const* yOffsets, sd::LongType start, sd::LongType stop);" dedupe_set content)
        
        # Note: The following are template methods with OpType and are typically instantiated elsewhere:
        # - execScalar<OpType>
        # - exec<OpType> (both overloads)
        # - execAll<OpType>
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for indexreduce templates
# ============================================================================
# TYPE NORMALIZATION FUNCTION
# ============================================================================


# ============================================================================
# PAIRWISE HANDLER WITH NORMALIZATION
# ============================================================================

function(handle_pairwise t1 t2 t3 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    message(STATUS "DEBUG: handle_pairwise called with t1='${t1}', t2='${t2}', t3='${t3}'")

    # Normalize types first
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    normalize_to_canonical_type("${t3}" norm_t3)
    message(STATUS "DEBUG: handle_pairwise normalized: '${norm_t1}', '${norm_t2}', '${norm_t3}'")

    # Skip any combination where ANY type isn't in its canonical form
    # This prevents duplicates like (unsigned short int, uint16_t, uint32_t) when (uint16_t, uint16_t, uint32_t) exists
    if(NOT (t1 STREQUAL norm_t1 AND t2 STREQUAL norm_t2 AND t3 STREQUAL norm_t3))
        # At least one type is a non-canonical alias - skip to avoid duplicates
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map normalized types to enum names for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    map_cpp_to_enum("${norm_t3}" type3_enum)

    # Validation checks...
    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" valid_12)
    _internal_srcore_is_valid_pair("${type2_enum}" "${type3_enum}" valid_23)
    _internal_srcore_is_valid_triple("${type1_enum}" "${type2_enum}" "${type3_enum}" valid_123)

    if(NOT valid_12 OR NOT valid_23 OR NOT valid_123)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations using NORMALIZED types (which are same as originals now due to check above)
    if(is_cuda)
        add_unique_instantiation("template void functions::pairwise_transforms::PairWiseTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo, void *vextraParams);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::pairwise_transforms::PairWiseTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo, void *extraParams, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::pairwise_transforms::PairWiseTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::exec(int opNum, const void *x, sd::LongType xStride, const void *y, sd::LongType yStride, void *z, sd::LongType resultStride, void *extraParams, sd::LongType len, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)

endfunction()

# ============================================================================
# SCALAR HANDLER WITH NORMALIZATION
# ============================================================================


function(handle_scalar t1 t2 t3 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    normalize_to_canonical_type("${t3}" norm_t3)

    # Skip any combination where ANY type isn't in its canonical form
    # This prevents duplicates like (unsigned short int, uint16_t, uint32_t) when (uint16_t, uint16_t, uint32_t) exists
    if(NOT (t1 STREQUAL norm_t1 AND t2 STREQUAL norm_t2 AND t3 STREQUAL norm_t3))
        # At least one type is a non-canonical alias - skip to avoid duplicates
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    map_cpp_to_enum("${norm_t3}" type3_enum)

    # Validation
    _internal_srcore_is_valid_pair("${type1_enum}" "${type3_enum}" valid_13)
    _internal_srcore_is_valid_pair("${type2_enum}" "${type3_enum}" valid_23)
    _internal_srcore_is_valid_triple("${type1_enum}" "${type2_enum}" "${type3_enum}" valid_triple)

    if(NOT valid_13 OR NOT valid_23 OR NOT valid_triple)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations using NORMALIZED types (which are same as originals now due to check above)
    if(is_cuda)
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, const sd::LongType *hxShapeInfo, void *vz, const sd::LongType *zShapeInfo, const sd::LongType *hzShapeInfo, const void *vscalar, void *vextraParams);" dedupe_set content)
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, void *vz, const sd::LongType *zShapeInfo, const void *vscalars, void *vextraParams, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *result, const sd::LongType *resultShapeInfo, const void *scalar, void *extraParams, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::scalar::ScalarTransform<${norm_t1}, ${norm_t2}, ${norm_t3}>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z, const sd::LongType *zShapeInfo, const void *scalars, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# BROADCAST HANDLER WITH NORMALIZATION
# ============================================================================

function(handle_broadcast t1 t2 t3 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    normalize_to_canonical_type("${t3}" norm_t3)

    # Skip any combination where ANY type isn't in its canonical form
    # This prevents duplicates like (unsigned short int, uint16_t, uint32_t) when (uint16_t, uint16_t, uint32_t) exists
    if(NOT (t1 STREQUAL norm_t1 AND t2 STREQUAL norm_t2 AND t3 STREQUAL norm_t3))
        # At least one type is a non-canonical alias - skip to avoid duplicates
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    map_cpp_to_enum("${norm_t3}" type3_enum)

    # Validation
    _internal_srcore_is_valid_triple("${type1_enum}" "${type2_enum}" "${type3_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations using NORMALIZED types (which are same as originals now due to check above)
    if(is_cuda)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execInverseBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LoopKind::Kind loopKind, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# REDUCE3 HANDLER WITH NORMALIZATION
# ============================================================================

function(handle_reduce3 t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    
    # Skip redundant type aliases
    if(norm_t1 STREQUAL norm_t2)
        if(NOT t1 STREQUAL t2)
            set(${content_var} "${content}" PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    
    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationPointer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo, sd::LongType const* yTadOffsets);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execAll(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationPointer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo, sd::LongType const* yTadOffsets);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execScalar(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* allocationPointer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execScalar(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce3::Reduce3<${t1}, ${t2}>::execAll(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* xTadShapeInfo, sd::LongType const* xOffsets, sd::LongType const* yTadShapeInfo, sd::LongType const* yOffsets, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


# ============================================================================
# INDEXREDUCE HANDLER WITH NORMALIZATION
# ============================================================================


function(handle_indexreduce t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    
    # Skip redundant combinations for index type (t2)
    # Index type should always be int64_t after normalization
    if(norm_t2 STREQUAL "int64_t" AND NOT t2 STREQUAL "int64_t")
        # Skip LongType, sd::LongType, long, long long variants
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    
    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::indexreduce::IndexReduce<${t1}, ${t2}>::executeIndexReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, void const* dx, sd::LongType const* xShapeInfo, sd::LongType xRank, void* extraParams, void* result, sd::LongType const* zShapeInfo, sd::LongType zRank, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationBuffer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
        add_unique_instantiation("template void functions::indexreduce::IndexReduce<${t1}, ${t2}>::executeIndexReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, void const* dx, sd::LongType const* xShapeInfo, sd::LongType xRank, void* extraParams, void* result, sd::LongType const* zShapeInfo, sd::LongType zRank, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationBuffer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
    else()
        add_unique_instantiation("template class sd::IndexReductionLoops<${t1}, ${t2}>;" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()
# Handler for reduce_float templates
function(handle_reduce_float t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    
    # Only process if t2 is a float type
    if(NOT norm_t2 MATCHES "float|double|float16|bfloat16")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Skip redundant type alias combinations
    if(norm_t1 STREQUAL norm_t2 AND NOT t1 STREQUAL t2)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    
    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::reduce::ReduceFloatFunction<${t1}, ${t2}>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* dXShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, const sd::LongType* dims);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceFloatFunction<${t1}, ${t2}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::reduce::ReduceFloatFunction<${t1}, ${t2}>::exec(int opNum, sd::memory::Workspace* workspace, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo, sd::LongType const* dimension);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceFloatFunction<${t1}, ${t2}>::execScalar(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for reduce_bool templates

function(handle_reduce_bool t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Only process if t2 is bool
    if(NOT t2 STREQUAL "bool")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Normalize t1 but don't skip aliases - NativeOpExecutioner uses 'long' explicitly
    normalize_to_canonical_type("${t1}" norm_t1)
    
    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    
    _internal_srcore_is_valid_pair("${type1_enum}" "BOOL" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::reduce::ReduceBoolFunction<${t1}, bool>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, sd::LongType* dXShapeInfo, sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, sd::LongType* dZShapeInfo, sd::LongType* hZShapeInfo, sd::LongType* dims);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceBoolFunction<${t1}, bool>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* zShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::reduce::ReduceBoolFunction<${t1}, bool>::exec(int opNum, sd::memory::Workspace* workspace, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo, sd::LongType const* dimension);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceBoolFunction<${t1}, bool>::execScalar(int opNum, void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# Handler for reduce_long templates

function(handle_reduce_long t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types to canonical forms for validation
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)

    # Only process if t2 is a 64-bit integer type (maps to INT64 or UINT64)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    if(NOT (type2_enum STREQUAL "INT64" OR type2_enum STREQUAL "UINT64"))
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)

    message(STATUS "DEBUG handle_reduce_long: t1='${t1}' norm_t1='${norm_t1}' type1_enum='${type1_enum}'")
    message(STATUS "DEBUG handle_reduce_long: t2='${t2}' norm_t2='${norm_t2}' type2_enum='${type2_enum}'")

    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" is_valid)
    message(STATUS "DEBUG handle_reduce_long: is_valid='${is_valid}' for pair (${type1_enum}, ${type2_enum})")

    if(NOT is_valid)
        message(STATUS "DEBUG handle_reduce_long: REJECTED pair (${type1_enum}, ${type2_enum})")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "DEBUG handle_reduce_long: ACCEPTED pair (${type1_enum}, ${type2_enum}), generating instantiations")

    # Generate instantiations using types from types.h (sd::LongType = long long)
    # Generate ONLY for types.h typedefs (sd::LongType, sd::UnsignedLong)
    if(is_cuda)
        add_unique_instantiation("template void functions::reduce::ReduceLongFunction<${t1}, ${t2}>::execReduce(dim3 launchDims, cudaStream_t* stream, int opNum, const void* x, sd::LongType* dXShapeInfo, sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, sd::LongType* dZShapeInfo, sd::LongType* hZShapeInfo, sd::LongType* dims);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceLongFunction<${t1}, ${t2}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, sd::LongType* xShapeInfo, sd::LongType* hXShapeInfo, void* extraParams, void* z, sd::LongType* zShapeInfo, sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, sd::LongType* tadOnlyShapeInfo);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::reduce::ReduceLongFunction<${t1}, ${t2}>::exec(int, sd::memory::Workspace*, void const*, sd::LongType const*, void*, void*, sd::LongType const*, sd::LongType*);" dedupe_set content)
        add_unique_instantiation("template void functions::reduce::ReduceLongFunction<${t1}, ${t2}>::execScalar(int, void const*, sd::LongType const*, void*, void*, sd::LongType const*);" dedupe_set content)
        add_unique_instantiation("template class sd::ReductionLongLoops<${t1}, ${t2}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for pairwise_int templates (single type - integer only operations)
# ============================================================================

function(handle_pairwise_int t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation - only integer types
    map_cpp_to_enum("${norm_t1}" type1_enum)
    if(NOT type1_enum MATCHES "INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::pairwise_transforms::PairWiseIntTransform<${t1}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* vz, sd::LongType const* zShapeInfo, void* vextraParams);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::pairwise_transforms::PairWiseIntTransform<${t1}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for pairwise_bool templates (two types - input and bool output)
# ============================================================================

function(handle_pairwise_bool t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Only process if t2 is bool
    if(NOT "${t2}" STREQUAL "bool")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Normalize t1
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    if(type1_enum STREQUAL "")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Validate pair
    _internal_srcore_is_valid_pair("${type1_enum}" "BOOL" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::pairwise_transforms::PairWiseBoolTransform<${t1}, bool>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* vz, sd::LongType const* zShapeInfo, void* vextraParams);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::pairwise_transforms::PairWiseBoolTransform<${t1}, bool>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for scalar_int templates (single type - integer only operations)
# ============================================================================

function(handle_scalar_int t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation - only integer types
    map_cpp_to_enum("${norm_t1}" type1_enum)
    if(NOT type1_enum MATCHES "INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::scalar::ScalarIntTransform<${t1}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void* vz, sd::LongType const* zShapeInfo, void const* vscalar, void* vextraParams);" dedupe_set content)
        add_unique_instantiation("template void functions::scalar::ScalarIntTransform<${t1}>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void* vz, sd::LongType const* zShapeInfo, void const* vscalars, void* vextraParams, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadShapeInfoZ, sd::LongType const* tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::scalar::ScalarIntTransform<${t1}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for scalar_bool templates (two types - input and bool output)
# ============================================================================

function(handle_scalar_bool t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Only process if t2 is bool
    if(NOT "${t2}" STREQUAL "bool")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Normalize t1
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    if(type1_enum STREQUAL "")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Validate pair
    _internal_srcore_is_valid_pair("${type1_enum}" "BOOL" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::scalar::ScalarBoolTransform<${t1}, bool>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void* vz, sd::LongType const* zShapeInfo, void const* vscalar, void const* vextraParams);" dedupe_set content)
        add_unique_instantiation("template void functions::scalar::ScalarBoolTransform<${t1}, bool>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void* vz, sd::LongType const* zShapeInfo, void const* vscalars, void* vextraParams, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadShapeInfoZ, sd::LongType const* tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::scalar::ScalarBoolTransform<${t1}, bool>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for summarystatsreduce templates
# ============================================================================

function(handle_summarystatsreduce t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)

    # Only process if t2 is a float type
    if(NOT norm_t2 MATCHES "float|double|float16|bfloat16")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)

    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::summarystats::SummaryStatsReduce<${t1}, ${t2}>::execSummaryStatsReduce(dim3& launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShapeInfo, sd::LongType const* hXShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo, sd::LongType const* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, bool biasCorrected, void* reductionBuffer);" dedupe_set content)
        add_unique_instantiation("template void functions::summarystats::SummaryStatsReduce<${t1}, ${t2}>::execSummaryStatsReduceScalar(dim3& launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShapeInfo, sd::LongType const* hXShapeInfo, void* extraParams, void* z, sd::LongType const* zShapeInfo, sd::LongType const* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, bool biasCorrected, void* reductionBuffer);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::summarystats::SummaryStatsReduce<${t1}, ${t2}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for transform_any templates (single type)
# ============================================================================

function(handle_transform_any t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::transform::TransformAny<${t1}>::executeTransformShaped(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShape, sd::LongType xRank, void* params, void* z, sd::LongType const* zShape, sd::LongType zRank, sd::LongType* allocationPointer, void* reductionPointer, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::transform::TransformAny<${t1}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for transform_bool templates (two types - input and bool output)
# ============================================================================

function(handle_transform_bool t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Only process if t2 is bool
    if(NOT "${t2}" STREQUAL "bool")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Normalize t1
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    if(type1_enum STREQUAL "")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Validate pair
    _internal_srcore_is_valid_pair("${type1_enum}" "BOOL" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::transform::TransformBool<${t1}, bool>::executeTransformShaped(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShape, sd::LongType xRank, void* params, void* z, sd::LongType const* zShape, sd::LongType zRank, sd::LongType* allocationPointer, void* reductionPointer, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::transform::TransformBool<${t1}, bool>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for transform_float templates (two types)
# ============================================================================

function(handle_transform_float t1 t2 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)

    # Only process if t2 is a float type
    if(NOT norm_t2 MATCHES "float|double|float16|bfloat16")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)

    _internal_srcore_is_valid_pair("${type1_enum}" "${type2_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::transform::TransformFloat<${t1}, ${t2}>::executeTransformShaped(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShape, sd::LongType xRank, void* params, void* z, sd::LongType const* zShape, sd::LongType zRank, sd::LongType* allocationPointer, void* reductionPointer, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::transform::TransformFloat<${t1}, ${t2}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for transform_same templates (single type)
# ============================================================================

function(handle_transform_same t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Skip string types
    if(norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::transform::TransformSame<${t1}>::executeTransformShaped(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShape, sd::LongType xRank, void* params, void* z, sd::LongType const* zShape, sd::LongType zRank, sd::LongType* allocationPointer, void* reductionPointer, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::transform::TransformSame<${t1}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Handler for transform_strict templates (single float type)
# ============================================================================

function(handle_transform_strict t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)

    # Skip redundant type aliases
    if(NOT t1 STREQUAL norm_t1)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Only process float types
    if(NOT norm_t1 MATCHES "float|double|float16|bfloat16")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template void functions::transform::TransformStrict<${t1}>::executeTransformShaped(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShape, sd::LongType xRank, void* params, void* z, sd::LongType const* zShape, sd::LongType zRank, sd::LongType* allocationPointer, void* reductionPointer, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets);" dedupe_set content)
    else()
        add_unique_instantiation("template class functions::transform::TransformStrict<${t1}>;" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()

# ============================================================================
# Legacy genCompilation function
# ============================================================================

function(genCompilation FL_ITEM generated_sources_var)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)
    set(EXTENSION "cpp")
    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()
    file(READ ${FL_ITEM} CONTENT_FL)
    set(SD_FLOAT_TYPES_GEN 0)
    set(SD_INTEGER_TYPES_GEN 0)
    set(SD_COMMON_TYPES_GEN 0)
    set(SD_PAIRWISE_TYPES_GEN 0)
    set(RANGE_STOP -1)
    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    # Use selective rendering type count if available
    if(DEFINED UNIFIED_TYPE_COUNT AND UNIFIED_TYPE_COUNT GREATER 0)
        math(EXPR SD_INTEGER_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        math(EXPR SD_COMMON_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        math(EXPR SD_FLOAT_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        math(EXPR SD_PAIRWISE_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
    else()
        set(SD_INTEGER_TYPES_END 7)
        set(SD_COMMON_TYPES_END 12)
        set(SD_FLOAT_TYPES_END 3)
        set(SD_PAIRWISE_TYPES_END 12)
    endif()

    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set(SD_INTEGER_TYPES_GEN 1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set(SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set(SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set(SD_PAIRWISE_TYPES_GEN 1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()
    endforeach()

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/compilation_units")
    set(local_generated_sources ${${generated_sources_var}})
    if(RANGE_STOP GREATER -1)
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            set(CURRENT_SD_FLOAT_TYPES_GEN ${SD_FLOAT_TYPES_GEN})
            set(CURRENT_SD_INTEGER_TYPES_GEN ${SD_INTEGER_TYPES_GEN})
            set(CURRENT_SD_COMMON_TYPES_GEN ${SD_COMMON_TYPES_GEN})
            set(CURRENT_SD_PAIRWISE_TYPES_GEN ${SD_PAIRWISE_TYPES_GEN})
            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set(CURRENT_SD_FLOAT_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set(CURRENT_SD_INTEGER_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set(CURRENT_SD_COMMON_TYPES_GEN 0)
            endif()
            set(SD_FLOAT_TYPES_GEN ${CURRENT_SD_FLOAT_TYPES_GEN})
            set(SD_INTEGER_TYPES_GEN ${CURRENT_SD_INTEGER_TYPES_GEN})
            set(SD_COMMON_TYPES_GEN ${CURRENT_SD_COMMON_TYPES_GEN})
            set(SD_PAIRWISE_TYPES_GEN ${CURRENT_SD_PAIRWISE_TYPES_GEN})
            set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            configure_file("${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
            list(APPEND local_generated_sources ${GENERATED_SOURCE})
        endforeach()
    endif()
    set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
endfunction()

# ============================================================================
# Direct instantiation file creation
# ============================================================================


function(setup_type_mapping)
    # Setup type index to C++ type mapping
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(idx 0)
        foreach(type ${UNIFIED_ACTIVE_TYPES})
            if(type STREQUAL "BOOL")
                set(SRCORE_TYPE_CPP_${idx} "bool" PARENT_SCOPE)
              elseif(type STREQUAL "INT8")
                set(SRCORE_TYPE_CPP_${idx} "int8_t;SignedChar;signed char;char" PARENT_SCOPE)
            elseif(type STREQUAL "UINT8")
                set(SRCORE_TYPE_CPP_${idx} "uint8_t;UnsignedChar;unsigned char" PARENT_SCOPE)
            elseif(type STREQUAL "INT16")
                set(SRCORE_TYPE_CPP_${idx} "int16_t;short;short int;signed short" PARENT_SCOPE)
            elseif(type STREQUAL "UINT16")
                set(SRCORE_TYPE_CPP_${idx} "uint16_t;unsigned short;unsigned short int" PARENT_SCOPE)
            elseif(type STREQUAL "INT32")
                set(SRCORE_TYPE_CPP_${idx} "int32_t;int;signed int;Int32Type" PARENT_SCOPE)
            elseif(type STREQUAL "UINT32")
                set(SRCORE_TYPE_CPP_${idx} "uint32_t;unsigned int;unsigned" PARENT_SCOPE)
            elseif(type STREQUAL "INT64")
                # Use sd::LongType (long long) from types.h
                set(SRCORE_TYPE_CPP_${idx} "sd::LongType" PARENT_SCOPE)
            elseif(type STREQUAL "UINT64")
                # Use sd::UnsignedLong (uint64_t) from types.h
                set(SRCORE_TYPE_CPP_${idx} "sd::UnsignedLong" PARENT_SCOPE)
            elseif(type STREQUAL "FLOAT32")
                set(SRCORE_TYPE_CPP_${idx} "float" PARENT_SCOPE)
            elseif(type STREQUAL "DOUBLE")
                set(SRCORE_TYPE_CPP_${idx} "double" PARENT_SCOPE)
            elseif(type STREQUAL "HALF")
                set(SRCORE_TYPE_CPP_${idx} "float16" PARENT_SCOPE)
            elseif(type STREQUAL "BFLOAT16")
                set(SRCORE_TYPE_CPP_${idx} "bfloat16" PARENT_SCOPE)
            elseif(type STREQUAL "UTF8")
                set(SRCORE_TYPE_CPP_${idx} "std::string" PARENT_SCOPE)
                set(SRCORE_IS_STRING_${idx} TRUE PARENT_SCOPE)
            elseif(type STREQUAL "UTF16")
                set(SRCORE_TYPE_CPP_${idx} "std::u16string" PARENT_SCOPE)
                set(SRCORE_IS_STRING_${idx} TRUE PARENT_SCOPE)
            elseif(type STREQUAL "UTF32")
                set(SRCORE_TYPE_CPP_${idx} "std::u32string" PARENT_SCOPE)
                set(SRCORE_IS_STRING_${idx} TRUE PARENT_SCOPE)
            else()
                message(WARNING "Unknown type in UNIFIED_ACTIVE_TYPES: ${type}")
                set(SRCORE_TYPE_CPP_${idx} "" PARENT_SCOPE)
            endif()
            math(EXPR idx "${idx} + 1")
        endforeach()

    endif()
endfunction()



function(create_direct_instantiation_file template_file combinations output_dir generated_sources_var)
    get_filename_component(template_name ${template_file} NAME_WE)
    file(READ "${template_file}" template_content)

    # MEMORY OPTIMIZATION: Check if template has @OP_GROUP@ placeholder
    # If yes, generate 3 versions (one per op group) to reduce per-file memory usage
    string(FIND "${template_content}" "@OP_GROUP@" has_op_group)
    if(NOT has_op_group EQUAL -1)
        message(STATUS "🔧 Op group template detected: ${template_name} - generating 3 op group versions")
        set(local_generated_sources ${${generated_sources_var}})

        foreach(op_group 1 2 3)
            # Create a modified template content with the op group set
            string(REPLACE "@OP_GROUP@" "${op_group}" group_template_content "${template_content}")

            # Write temporary template file for this op group
            set(temp_template_file "${output_dir}/${template_name}_opgroup${op_group}.cu.in.tmp")
            file(WRITE "${temp_template_file}" "${group_template_content}")

            # Recursively call to process this op group version
            # Use a different name to avoid conflicts
            create_direct_instantiation_file_impl(
                "${temp_template_file}"
                "${combinations}"
                "${output_dir}"
                local_generated_sources
                "${template_name}_g${op_group}")

            # Clean up temp file
            file(REMOVE "${temp_template_file}")
        endforeach()

        set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
        return()
    endif()

    # No @OP_GROUP@ - proceed normally
    create_direct_instantiation_file_impl("${template_file}" "${combinations}" "${output_dir}" ${generated_sources_var} "${template_name}")
    set(${generated_sources_var} ${${generated_sources_var}} PARENT_SCOPE)
endfunction()

# Implementation function for create_direct_instantiation_file
function(create_direct_instantiation_file_impl template_file combinations output_dir generated_sources_var output_name)
    set(template_name "${output_name}")
    file(READ "${template_file}" template_content)
    string(REGEX MATCHALL "#include[^\n]*" includes "${template_content}")

    # Determine file extension (handle both .cu.in and .cu.in.tmp for temp files)
    set(IS_CUDA_FILE FALSE)
    if(template_file MATCHES "\\.cu\\.in" OR template_file MATCHES "\\.cu\\.in\\.tmp$")
        set(IS_CUDA_FILE TRUE)
        set(file_extension "cu")
    else()
        set(file_extension "cpp")
    endif()
    
    # Build file header
    set(file_header "")

    foreach(inc ${includes})
        string(APPEND file_header "${inc}\n")
    endforeach()

    # Add extra includes for type conversions
    if(template_name MATCHES ".*specials_double.*")
        string(APPEND file_header "#include <loops/type_conversions.h>\n")
    endif()

    string(APPEND file_header "\n// Direct instantiations - comprehensive type variant generation\n\n")

    # Setup type mapping
    setup_type_mapping()
    
    # Process combinations
    set(chunk_content "${file_header}")
    set(instantiation_count 0)
    set(chunk_index 0)
    set(local_generated_sources ${${generated_sources_var}})
    set(total_instantiations 0)
    set(processed_normalized_combinations "")  # Track normalized combinations to avoid duplicates

    foreach(combination ${combinations})
        string(REPLACE "," ";" parts "${combination}")
        list(LENGTH parts parts_count)
        
        # Extract types from indices
        set(t1 "")
        set(t2 "")
        set(t3 "")
        set(has_string FALSE)
        
        if(parts_count GREATER_EQUAL 1)
            list(GET parts 0 idx1)
            if(DEFINED SRCORE_TYPE_CPP_${idx1})
                set(raw_t1 "${SRCORE_TYPE_CPP_${idx1}}")
                # Normalize all types to use typedefs from types.h (sd::LongType, sd::UnsignedLong)
                normalize_to_canonical_type("${raw_t1}" t1)
                if(DEFINED SRCORE_IS_STRING_${idx1} AND SRCORE_IS_STRING_${idx1})
                    set(has_string TRUE)
                endif()
            endif()
        endif()
        if(parts_count GREATER_EQUAL 2)
            list(GET parts 1 idx2)
            if(DEFINED SRCORE_TYPE_CPP_${idx2})
                set(raw_t2 "${SRCORE_TYPE_CPP_${idx2}}")
                # Normalize all types to use typedefs from types.h (sd::LongType, sd::UnsignedLong)
                normalize_to_canonical_type("${raw_t2}" t2)
                if(DEFINED SRCORE_IS_STRING_${idx2} AND SRCORE_IS_STRING_${idx2})
                    set(has_string TRUE)
                endif()
            endif()
        endif()
        if(parts_count GREATER_EQUAL 3)
            list(GET parts 2 idx3)
            if(DEFINED SRCORE_TYPE_CPP_${idx3})
                set(raw_t3 "${SRCORE_TYPE_CPP_${idx3}}")
                # Normalize all types to use typedefs from types.h (sd::LongType, sd::UnsignedLong)
                normalize_to_canonical_type("${raw_t3}" t3)
                if(DEFINED SRCORE_IS_STRING_${idx3} AND SRCORE_IS_STRING_${idx3})
                    set(has_string TRUE)
                endif()
            endif()
        endif()
        
        # Skip string combinations
        if(has_string)
            continue()
        endif()

        # Skip if we couldn't map the types
        if(NOT t1)
            continue()
        endif()

        # Skip if we've already processed this normalized combination
        # This prevents duplicates when 'long' and 'int64_t' both normalize to 'int64_t'
        set(normalized_combo "${t1}")
        if(t2)
            set(normalized_combo "${normalized_combo},${t2}")
        endif()
        if(t3)
            set(normalized_combo "${normalized_combo},${t3}")
        endif()

        list(FIND processed_normalized_combinations "${normalized_combo}" already_processed)
        if(already_processed GREATER -1)
            continue()
        endif()
        list(APPEND processed_normalized_combinations "${normalized_combo}")

        # Count template lines before dispatch
        string(REGEX MATCHALL "template [^\n]+" templates_before "${chunk_content}")
        list(LENGTH templates_before count_before)

        # Dispatch to appropriate handler
        dispatch_to_handler("${template_name}" "${t1}" "${t2}" "${t3}" "${parts_count}" chunk_content ${IS_CUDA_FILE})

        # Count template lines after dispatch to get actual instantiations added
        string(REGEX MATCHALL "template [^\n]+" templates_after "${chunk_content}")
        list(LENGTH templates_after count_after)
        math(EXPR templates_added "${count_after} - ${count_before}")

        math(EXPR instantiation_count "${instantiation_count} + ${templates_added}")
        math(EXPR total_instantiations "${total_instantiations} + ${templates_added}")

        # Write chunk if limit reached (now counting actual template instantiations)
        if(instantiation_count GREATER_EQUAL ${MULTI_PASS_CHUNK_SIZE})
            set(chunk_file "${output_dir}/${template_name}_direct_${chunk_index}.${file_extension}")
            file(WRITE "${chunk_file}" "${chunk_content}")
            list(APPEND local_generated_sources "${chunk_file}")

            set(chunk_content "${file_header}")
            set(instantiation_count 0)
            math(EXPR chunk_index "${chunk_index} + 1")
        endif()
    endforeach()
    
    # Write remaining content
    if(instantiation_count GREATER 0)
        # If call tracing is enabled, don't create huge "_final" files
        # Instead, write them as regular chunks to prevent relocation overflow
        if(SD_GCC_FUNCTRACE)
            set(chunk_file "${output_dir}/${template_name}_direct_${chunk_index}.${file_extension}")
        else()
            set(chunk_file "${output_dir}/${template_name}_direct_final.${file_extension}")
        endif()
        file(WRITE "${chunk_file}" "${chunk_content}")
        list(APPEND local_generated_sources "${chunk_file}")
    endif()

    set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
    
    if(IS_CUDA_FILE)
        message(STATUS "✅ Generated CUDA instantiation files for ${template_name}: ${total_instantiations} instantiations")
    else()
        message(STATUS "✅ Generated CPU instantiation files for ${template_name}: ${total_instantiations} instantiations")
    endif()
endfunction()


# Handler for broadcast_int templates - excludes bool and string types

function(handle_broadcast_int t1 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")
    
    # Normalize type
    normalize_to_canonical_type("${t1}" norm_t1)
    
    # Skip bool and string types
    if(norm_t1 STREQUAL "bool" OR norm_t1 MATCHES "std::string|std::u16string|std::u32string")
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Skip redundant type aliases (only keep canonical form)
    if(NOT t1 STREQUAL norm_t1)
        # This is an alias, skip if we're not the canonical form
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()
    
    # Generate instantiations
    if(is_cuda)
        add_unique_instantiation("template class BroadcastInt<${t1}>;" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::broadcast::BroadcastInt<${t1}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastInt<${t1}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::BroadcastInt<${t1}>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


# Handler for broadcast (3-type) templates

function(handle_broadcast t1 t2 t3 content_var is_cuda)
    set(content "${${content_var}}")
    set(dedupe_set "")

    # Normalize types
    normalize_to_canonical_type("${t1}" norm_t1)
    normalize_to_canonical_type("${t2}" norm_t2)
    normalize_to_canonical_type("${t3}" norm_t3)

    # Skip any combination where ANY type isn't in its canonical form
    # This prevents duplicates like (unsigned short int, uint16_t, uint32_t) when (uint16_t, uint16_t, uint32_t) exists
    if(NOT (t1 STREQUAL norm_t1 AND t2 STREQUAL norm_t2 AND t3 STREQUAL norm_t3))
        # At least one type is a non-canonical alias - skip to avoid duplicates
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Map to enum for validation
    map_cpp_to_enum("${norm_t1}" type1_enum)
    map_cpp_to_enum("${norm_t2}" type2_enum)
    map_cpp_to_enum("${norm_t3}" type3_enum)

    # Validation
    _internal_srcore_is_valid_triple("${type1_enum}" "${type2_enum}" "${type3_enum}" is_valid)
    if(NOT is_valid)
        set(${content_var} "${content}" PARENT_SCOPE)
        return()
    endif()

    # Generate instantiations using NORMALIZED types (which are same as originals now due to check above)
    if(is_cuda)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execInverseBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);" dedupe_set content)
    else()
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LoopKind::Kind loopKind, sd::LongType start, sd::LongType stop);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);" dedupe_set content)
        add_unique_instantiation("template void functions::broadcast::Broadcast<${norm_t1}, ${norm_t2}, ${norm_t3}>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetZ, sd::LongType start, sd::LongType stop);" dedupe_set content)
    endif()

    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


 # Helper function to map C++ type to enum (local to this function)
    function(map_cpp_to_enum cpp_type enum_var)
        set(enum_name "")
        if(cpp_type MATCHES "int8_t|SignedChar|signed char|char")
            set(enum_name "INT8")
        elseif(cpp_type MATCHES "uint8_t|UnsignedChar|unsigned char")
            set(enum_name "UINT8")
        elseif(cpp_type MATCHES "int16_t|short")
            set(enum_name "INT16")
        elseif(cpp_type MATCHES "uint16_t|unsigned short")
            set(enum_name "UINT16")
        elseif(cpp_type MATCHES "int32_t|int|Int32Type|signed int")
            set(enum_name "INT32")
        elseif(cpp_type MATCHES "uint32_t|unsigned int|unsigned")
            set(enum_name "UINT32")
        elseif(cpp_type MATCHES "int64_t|long long|long|LongType|sd::LongType")
            set(enum_name "INT64")
        elseif(cpp_type MATCHES "uint64_t|unsigned long long|unsigned long|UnsignedLong|sd::UnsignedLong")
            set(enum_name "UINT64")
        elseif(cpp_type STREQUAL "float")
            set(enum_name "FLOAT32")
        elseif(cpp_type STREQUAL "double")
            set(enum_name "DOUBLE")
        elseif(cpp_type STREQUAL "float16")
            set(enum_name "HALF")
        elseif(cpp_type STREQUAL "bfloat16")
            set(enum_name "BFLOAT16")
        elseif(cpp_type STREQUAL "bool")
            set(enum_name "BOOL")
        endif()
        set(${enum_var} "${enum_name}" PARENT_SCOPE)
    endfunction()
    

function(dispatch_to_handler template_name t1 t2 t3 parts_count content_var is_cuda)
    set(content "${${content_var}}")
    
    # DEBUG: Log every call
    message(STATUS "DEBUG dispatch_to_handler:")
    message(STATUS "  template_name: '${template_name}'")
    message(STATUS "  parts_count: ${parts_count}")
    message(STATUS "  t1: '${t1}'")
    message(STATUS "  t2: '${t2}'")
    message(STATUS "  t3: '${t3}'")
    
    # Create a template-specific deduplication set name
    if(is_cuda)
        set(dedupe_property_name "DEDUPE_SET_${template_name}_CUDA")
    else()
        set(dedupe_property_name "DEDUPE_SET_${template_name}_CPU")
    endif()
    
    get_property(template_dedupe_set GLOBAL PROPERTY ${dedupe_property_name})
    if(NOT template_dedupe_set)
        set(template_dedupe_set "")
    endif()
    
   
    # Parse type lists
    if(t1)
        string(REPLACE ";" "\\;" t1_escaped "${t1}")
        string(REPLACE "\\;" ";" t1_list "${t1_escaped}")
    else()
        set(t1_list "")
    endif()
    
    if(NOT t2 STREQUAL "")
        string(REPLACE ";" "\\;" t2_escaped "${t2}")
        string(REPLACE "\\;" ";" t2_list "${t2_escaped}")
    else()
        set(t2_list "")
    endif()
    
    if(NOT t3 STREQUAL "")
        string(REPLACE ";" "\\;" t3_escaped "${t3}")
        string(REPLACE "\\;" ";" t3_list "${t3_escaped}")
    else()
        set(t3_list "")
    endif()
    
    # Single type templates
    if(template_name MATCHES ".*specials_single.*")
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)
            
            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                handle_specials_single("${norm_type}" content)
                list(APPEND template_dedupe_set "${combo_key}")
            endif()
        endforeach()
        
    # Two type templates
    elseif(template_name MATCHES ".*specials_double.*" AND parts_count GREATER_EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                normalize_to_canonical_type("${v1}" norm_v1)
                normalize_to_canonical_type("${v2}" norm_v2)
                
                set(combo_key "${norm_v1},${norm_v2}")
                list(FIND template_dedupe_set "${combo_key}" found_idx)
                if(found_idx EQUAL -1)
                    map_cpp_to_enum("${norm_v1}" v1_enum)
                    map_cpp_to_enum("${norm_v2}" v2_enum)
                    if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "")
                        _internal_srcore_is_valid_pair("${v1_enum}" "${v2_enum}" is_valid)
                        if(is_valid)
                            handle_specials_double("${norm_v1}" "${norm_v2}" content)
                            list(APPEND template_dedupe_set "${combo_key}")
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()
        
    elseif(template_name MATCHES ".*broadcast_int.*")
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)
            
            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                if(NOT norm_type STREQUAL "bool" AND NOT norm_type MATCHES "std::string|std::u16string|std::u32string")
                    handle_broadcast_int("${norm_type}" content ${is_cuda})
                    list(APPEND template_dedupe_set "${combo_key}")
                endif()
            endif()
        endforeach()
        
    elseif(template_name MATCHES ".*broadcast_bool.*" AND parts_count EQUAL 2)
        foreach(v2 IN LISTS t2_list)
            if(v2 STREQUAL "bool")
                foreach(v1 IN LISTS t1_list)
                    normalize_to_canonical_type("${v1}" norm_v1)
                    
                    set(combo_key "${norm_v1},bool")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        if(NOT norm_v1 MATCHES "std::string|std::u16string|std::u32string")
                            handle_broadcast_bool("${norm_v1}" "bool" content ${is_cuda})
                            list(APPEND template_dedupe_set "${combo_key}")
                        endif()
                    endif()
                endforeach()
            endif()
        endforeach()
        
    # Three type templates
    elseif(template_name MATCHES ".*broadcast.*" AND NOT template_name MATCHES ".*broadcast_bool.*|.*broadcast_int.*" AND parts_count EQUAL 3)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                foreach(v3 IN LISTS t3_list)
                    normalize_to_canonical_type("${v1}" norm_v1)
                    normalize_to_canonical_type("${v2}" norm_v2)
                    normalize_to_canonical_type("${v3}" norm_v3)
                    
                    set(combo_key "${norm_v1},${norm_v2},${norm_v3}")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        map_cpp_to_enum("${norm_v2}" v2_enum)
                        map_cpp_to_enum("${norm_v3}" v3_enum)
                        if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "" AND NOT v3_enum STREQUAL "")
                            _internal_srcore_is_valid_triple("${v1_enum}" "${v2_enum}" "${v3_enum}" is_valid)
                            if(is_valid)
                                handle_broadcast("${norm_v1}" "${norm_v2}" "${norm_v3}" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endforeach()
            endforeach()
        endforeach()
        
    elseif(template_name MATCHES "pairwise" AND parts_count EQUAL 3)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                foreach(v3 IN LISTS t3_list)
                    normalize_to_canonical_type("${v1}" norm_v1)
                    normalize_to_canonical_type("${v2}" norm_v2)
                    normalize_to_canonical_type("${v3}" norm_v3)
                    
                    set(combo_key "${norm_v1},${norm_v2},${norm_v3}")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        map_cpp_to_enum("${norm_v2}" v2_enum)
                        map_cpp_to_enum("${norm_v3}" v3_enum)
                        if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "" AND NOT v3_enum STREQUAL "")
                            _internal_srcore_is_valid_triple("${v1_enum}" "${v2_enum}" "${v3_enum}" is_valid)
                            if(is_valid)
                                handle_pairwise("${norm_v1}" "${norm_v2}" "${norm_v3}" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endforeach()
            endforeach()
        endforeach()
        
    elseif(template_name MATCHES ".*scalar.*" AND parts_count EQUAL 3)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                foreach(v3 IN LISTS t3_list)
                    normalize_to_canonical_type("${v1}" norm_v1)
                    normalize_to_canonical_type("${v2}" norm_v2)
                    normalize_to_canonical_type("${v3}" norm_v3)
                    
                    set(combo_key "${norm_v1},${norm_v2},${norm_v3}")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        map_cpp_to_enum("${norm_v2}" v2_enum)
                        map_cpp_to_enum("${norm_v3}" v3_enum)
                        if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "" AND NOT v3_enum STREQUAL "")
                            _internal_srcore_is_valid_pair("${v1_enum}" "${v3_enum}" valid_13)
                            _internal_srcore_is_valid_pair("${v2_enum}" "${v3_enum}" valid_23)
                            _internal_srcore_is_valid_triple("${v1_enum}" "${v2_enum}" "${v3_enum}" valid_triple)
                            
                            if(valid_13 AND valid_23 AND valid_triple)
                                handle_scalar("${norm_v1}" "${norm_v2}" "${norm_v3}" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endforeach()
            endforeach()
        endforeach()
        
    # Single type handlers
    elseif(template_name MATCHES ".*random.*")
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)
            
            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                handle_random("${norm_type}" content ${is_cuda})
                list(APPEND template_dedupe_set "${combo_key}")
            endif()
        endforeach()
        
    elseif(template_name MATCHES ".*reduce_same.*")
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)
            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                handle_reduce_same("${type}" content ${is_cuda})
                list(APPEND template_dedupe_set "${combo_key}")
            endif()
        endforeach()
        
    # Two type handlers with validation
    elseif(template_name MATCHES ".*reduce3.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                normalize_to_canonical_type("${v1}" norm_v1)
                normalize_to_canonical_type("${v2}" norm_v2)
                
                set(combo_key "${norm_v1},${norm_v2}")
                list(FIND template_dedupe_set "${combo_key}" found_idx)
                if(found_idx EQUAL -1)
                    map_cpp_to_enum("${norm_v1}" v1_enum)
                    map_cpp_to_enum("${norm_v2}" v2_enum)
                    if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "")
                        _internal_srcore_is_valid_pair("${v1_enum}" "${v2_enum}" is_valid)
                        if(is_valid)
                            handle_reduce3("${norm_v1}" "${norm_v2}" content ${is_cuda})
                            list(APPEND template_dedupe_set "${combo_key}")
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()
        
    elseif(template_name MATCHES ".*indexreduc.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                normalize_to_canonical_type("${v1}" norm_v1)
                normalize_to_canonical_type("${v2}" norm_v2)
                
                set(combo_key "${norm_v1},${norm_v2}")
                list(FIND template_dedupe_set "${combo_key}" found_idx)
                if(found_idx EQUAL -1)
                    map_cpp_to_enum("${norm_v1}" v1_enum)
                    map_cpp_to_enum("${norm_v2}" v2_enum)
                    if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "")
                        _internal_srcore_is_valid_pair("${v1_enum}" "${v2_enum}" is_valid)
                        if(is_valid)
                            handle_indexreduce("${norm_v1}" "${norm_v2}" content ${is_cuda})
                            list(APPEND template_dedupe_set "${combo_key}")
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()
        
    elseif(template_name MATCHES ".*reduce_float.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                normalize_to_canonical_type("${v1}" norm_v1)
                normalize_to_canonical_type("${v2}" norm_v2)
                
                if(norm_v2 MATCHES "float|double|float16|bfloat16")
                    set(combo_key "${norm_v1},${norm_v2}")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        map_cpp_to_enum("${norm_v2}" v2_enum)
                        if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "")
                            _internal_srcore_is_valid_pair("${v1_enum}" "${v2_enum}" is_valid)
                            if(is_valid)
                                handle_reduce_float("${norm_v1}" "${norm_v2}" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()
        
    elseif(template_name MATCHES ".*reduce_bool.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                if(v2 STREQUAL "bool")
                    normalize_to_canonical_type("${v1}" norm_v1)
                    set(combo_key "${norm_v1},bool")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        if(NOT v1_enum STREQUAL "")
                            _internal_srcore_is_valid_pair("${v1_enum}" "BOOL" is_valid)
                            if(is_valid)
                                handle_reduce_bool("${v1}" "bool" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()
        
elseif(template_name MATCHES ".*reduce_long.*" AND parts_count EQUAL 2)
    foreach(v1 IN LISTS t1_list)
        foreach(v2 IN LISTS t2_list)
            normalize_to_canonical_type("${v2}" norm_v2)
            if(norm_v2 STREQUAL "sd::LongType" OR norm_v2 STREQUAL "sd::UnsignedLong")
                normalize_to_canonical_type("${v1}" norm_v1)
                set(combo_key "${norm_v1},${norm_v2}")
                list(FIND template_dedupe_set "${combo_key}" found_idx)
                if(found_idx EQUAL -1)
                    map_cpp_to_enum("${norm_v1}" v1_enum)
                    if(NOT v1_enum STREQUAL "")
                        # Check if valid pair for either INT64 or UINT64
                        _internal_srcore_is_valid_pair("${v1_enum}" "INT64" is_valid_int64)
                        _internal_srcore_is_valid_pair("${v1_enum}" "UINT64" is_valid_uint64)
                        if(is_valid_int64 OR is_valid_uint64)
                            handle_reduce_long("${v1}" "${v2}" content ${is_cuda})
                            list(APPEND template_dedupe_set "${combo_key}")
                        endif()
                    endif()
                endif()
            endif()
        endforeach()
    endforeach()

    # Pairwise int (single type - integer only)
    elseif(template_name MATCHES ".*pairwise_int.*" AND parts_count EQUAL 1)
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)

            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                handle_pairwise_int("${norm_type}" content ${is_cuda})
                list(APPEND template_dedupe_set "${combo_key}")
            endif()
        endforeach()

    # Pairwise bool (two types)
    elseif(template_name MATCHES ".*pairwise_bool.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                if(v2 STREQUAL "bool")
                    normalize_to_canonical_type("${v1}" norm_v1)
                    set(combo_key "${norm_v1},bool")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        handle_pairwise_bool("${norm_v1}" "bool" content ${is_cuda})
                        list(APPEND template_dedupe_set "${combo_key}")
                    endif()
                endif()
            endforeach()
        endforeach()

    # Scalar int (single type - integer only)
    elseif(template_name MATCHES ".*scalar_int.*" AND parts_count EQUAL 1)
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)

            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                handle_scalar_int("${norm_type}" content ${is_cuda})
                list(APPEND template_dedupe_set "${combo_key}")
            endif()
        endforeach()

    # Scalar bool (two types)
    elseif(template_name MATCHES ".*scalar_bool.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                if(v2 STREQUAL "bool")
                    normalize_to_canonical_type("${v1}" norm_v1)
                    set(combo_key "${norm_v1},bool")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        handle_scalar_bool("${norm_v1}" "bool" content ${is_cuda})
                        list(APPEND template_dedupe_set "${combo_key}")
                    endif()
                endif()
            endforeach()
        endforeach()

    # Summary stats reduce (two types)
    elseif(template_name MATCHES ".*summarystats.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                normalize_to_canonical_type("${v1}" norm_v1)
                normalize_to_canonical_type("${v2}" norm_v2)

                if(norm_v2 MATCHES "float|double|float16|bfloat16")
                    set(combo_key "${norm_v1},${norm_v2}")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        map_cpp_to_enum("${norm_v2}" v2_enum)
                        if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "")
                            _internal_srcore_is_valid_pair("${v1_enum}" "${v2_enum}" is_valid)
                            if(is_valid)
                                handle_summarystatsreduce("${norm_v1}" "${norm_v2}" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()

    # Transform any (single type)
    elseif(template_name MATCHES ".*transform_any.*" AND parts_count EQUAL 1)
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)

            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                if(NOT norm_type MATCHES "std::string|std::u16string|std::u32string")
                    handle_transform_any("${norm_type}" content ${is_cuda})
                    list(APPEND template_dedupe_set "${combo_key}")
                endif()
            endif()
        endforeach()

    # Transform bool (two types)
    elseif(template_name MATCHES ".*transform_bool.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                if(v2 STREQUAL "bool")
                    normalize_to_canonical_type("${v1}" norm_v1)
                    set(combo_key "${norm_v1},bool")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        handle_transform_bool("${norm_v1}" "bool" content ${is_cuda})
                        list(APPEND template_dedupe_set "${combo_key}")
                    endif()
                endif()
            endforeach()
        endforeach()

    # Transform float (two types)
    elseif(template_name MATCHES ".*transform_float.*" AND parts_count EQUAL 2)
        foreach(v1 IN LISTS t1_list)
            foreach(v2 IN LISTS t2_list)
                normalize_to_canonical_type("${v1}" norm_v1)
                normalize_to_canonical_type("${v2}" norm_v2)

                if(norm_v2 MATCHES "float|double|float16|bfloat16")
                    set(combo_key "${norm_v1},${norm_v2}")
                    list(FIND template_dedupe_set "${combo_key}" found_idx)
                    if(found_idx EQUAL -1)
                        map_cpp_to_enum("${norm_v1}" v1_enum)
                        map_cpp_to_enum("${norm_v2}" v2_enum)
                        if(NOT v1_enum STREQUAL "" AND NOT v2_enum STREQUAL "")
                            _internal_srcore_is_valid_pair("${v1_enum}" "${v2_enum}" is_valid)
                            if(is_valid)
                                handle_transform_float("${norm_v1}" "${norm_v2}" content ${is_cuda})
                                list(APPEND template_dedupe_set "${combo_key}")
                            endif()
                        endif()
                    endif()
                endif()
            endforeach()
        endforeach()

    # Transform same (single type)
    elseif(template_name MATCHES ".*transform_same.*" AND parts_count EQUAL 1)
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)

            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                if(NOT norm_type MATCHES "std::string|std::u16string|std::u32string")
                    handle_transform_same("${norm_type}" content ${is_cuda})
                    list(APPEND template_dedupe_set "${combo_key}")
                endif()
            endif()
        endforeach()

    # Transform strict (single float type)
    elseif(template_name MATCHES ".*transform_strict.*" AND parts_count EQUAL 1)
        foreach(type IN LISTS t1_list)
            normalize_to_canonical_type("${type}" norm_type)

            set(combo_key "${norm_type}")
            list(FIND template_dedupe_set "${combo_key}" found_idx)
            if(found_idx EQUAL -1)
                if(norm_type MATCHES "float|double|float16|bfloat16")
                    handle_transform_strict("${norm_type}" content ${is_cuda})
                    list(APPEND template_dedupe_set "${combo_key}")
                endif()
            endif()
        endforeach()

    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "No handler for template: ${template_name} with ${parts_count} types")
        endif()
    endif()

    # Save the updated template-specific dedupe set
    set_property(GLOBAL PROPERTY ${dedupe_property_name} "${template_dedupe_set}")
    
    set(${content_var} "${content}" PARENT_SCOPE)
endfunction()


function(setup_fallback_combinations)
    # Fallback when selective rendering is not available
    set(UNIFIED_COMBINATIONS_1 "0;1;2;3;4;5;6;7;8;9;10;11;12" PARENT_SCOPE)
    
    set(combos_2 "")
    foreach(i RANGE 0 12)
        foreach(j RANGE 0 12)
            list(APPEND combos_2 "${i},${j}")
        endforeach()
    endforeach()
    set(UNIFIED_COMBINATIONS_2 ${combos_2} PARENT_SCOPE)
    
    set(combos_3 "")
    foreach(i RANGE 0 12)
        foreach(j RANGE 0 12)
            foreach(k RANGE 0 12)
                list(APPEND combos_3 "${i},${j},${k}")
            endforeach()
        endforeach()
    endforeach()
    set(UNIFIED_COMBINATIONS_3 ${combos_3} PARENT_SCOPE)
    
    # Setup default active types
    set(UNIFIED_ACTIVE_TYPES "BOOL;INT8;UINT8;INT16;UINT16;INT32;UINT32;INT64;UINT64;FLOAT32;DOUBLE;HALF;BFLOAT16" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT 13 PARENT_SCOPE)
endfunction()

# ============================================================================
# CUDA template processing
# ============================================================================

function(process_cuda_comb_templates output_dir generated_sources_var)
    message(STATUS "🔧 Processing CUDA combination templates...")

    # Find all CUDA combination templates
    file(GLOB_RECURSE CUDA_COMB_TEMPLATES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/comb_compilation_units/*.cu.in")

    if(NOT CUDA_COMB_TEMPLATES)
        message(STATUS "⚠️  No CUDA combination templates found")
        return()
    endif()

    if(NOT DEFINED UNIFIED_COMBINATIONS_2 OR NOT DEFINED UNIFIED_COMBINATIONS_3)
        message(FATAL_ERROR "❌ CUDA processing requires selective rendering combinations!")
    endif()

    set(local_generated_sources ${${generated_sources_var}})
    list(LENGTH CUDA_COMB_TEMPLATES template_count)
    message(STATUS "📋 Found ${template_count} CUDA combination templates")

    # Use selective rendering combinations
    if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_1 "")
        set(COMBINATIONS_2 ${UNIFIED_COMBINATIONS_2})
        set(COMBINATIONS_3 ${UNIFIED_COMBINATIONS_3})
        # Same-type only combinations for BUILD_SINGLE_SELECTOR_THRICE templates (e.g., Broadcast)
        if(DEFINED UNIFIED_COMBINATIONS_3_SAME)
            set(COMBINATIONS_3_SAME ${UNIFIED_COMBINATIONS_3_SAME})
        else()
            # Fallback: generate same-type combinations from COMBINATIONS_1
            set(COMBINATIONS_3_SAME "")
            foreach(idx ${COMBINATIONS_1})
                list(APPEND COMBINATIONS_3_SAME "${idx},${idx},${idx}")
            endforeach()
        endif()

        # Generate 1-type combinations from 2-type combinations
        foreach(combination ${COMBINATIONS_2})
            string(REPLACE "," ";" combo_parts "${combination}")
            list(GET combo_parts 0 comb1)
            list(FIND COMBINATIONS_1 "${comb1}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND COMBINATIONS_1 "${comb1}")
            endif()
            list(GET combo_parts 1 comb2)
            list(FIND COMBINATIONS_1 "${comb2}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND COMBINATIONS_1 "${comb2}")
            endif()
        endforeach()

        foreach(combination ${COMBINATIONS_3})
            string(REPLACE "," ";" combo_parts "${combination}")
            foreach(part ${combo_parts})
                list(FIND COMBINATIONS_1 "${part}" found_idx)
                if(found_idx EQUAL -1)
                    list(APPEND COMBINATIONS_1 "${part}")
                endif()
            endforeach()
        endforeach()

        list(LENGTH COMBINATIONS_1 combo_1_count)
        list(LENGTH COMBINATIONS_2 combo_2_count)
        list(LENGTH COMBINATIONS_3 combo_3_count)
        list(LENGTH COMBINATIONS_3_SAME combo_3_same_count)
        message(STATUS "🎯 Using selective CUDA combinations: 1-type=${combo_1_count}, 2-type=${combo_2_count}, 3-type=${combo_3_count}, 3-type-same=${combo_3_same_count}")
    endif()

    foreach(TEMPLATE_FILE ${CUDA_COMB_TEMPLATES})
        get_filename_component(template_name ${TEMPLATE_FILE} NAME_WE)

        # Determine which combination set to use based on template name
        if(template_name MATCHES ".*_template_1\$")
            set(combinations_to_use ${COMBINATIONS_1})
            message(STATUS "🔄 Processing 1-type CUDA template: ${template_name}")
        elseif(template_name MATCHES ".*_template_2\$")
            set(combinations_to_use ${COMBINATIONS_2})
            message(STATUS "🔄 Processing 2-type CUDA template: ${template_name}")
        elseif(template_name MATCHES "^(broadcast|pairwise|scalar)_(instantiation_template_3|exec_.*)\$")
            # OPTIMIZATION: These templates use BUILD_SINGLE_SELECTOR_THRICE which only dispatches (T, T, T)
            # Cross-type combinations are NEVER used at runtime, so only generate same-type
            # Templates matched:
            # - broadcast_instantiation_template_3, broadcast_exec_broadcast_with_dimension, broadcast_exec_inverse_broadcast
            # - pairwise_instantiation_template_3, pairwise_exec_cuda_shaped
            # - scalar_instantiation_template_3, scalar_exec_cuda_along_dimension, scalar_exec_cuda_shaped
            set(combinations_to_use ${COMBINATIONS_3_SAME})
            list(LENGTH COMBINATIONS_3_SAME same_count)
            message(STATUS "🔄 Processing 3-type SAME-TYPE CUDA template: ${template_name} (${same_count} combinations instead of ${combo_3_count})")
        elseif(template_name MATCHES ".*_template_3\$")
            set(combinations_to_use ${COMBINATIONS_3})
            message(STATUS "🔄 Processing 3-type CUDA template: ${template_name}")
        else()
            # Default to 3-type for templates without explicit numbering
            set(combinations_to_use ${COMBINATIONS_3})
            message(STATUS "🔄 Processing CUDA template (defaulting to 3-type): ${template_name}")
        endif()

        create_direct_instantiation_file("${TEMPLATE_FILE}" "${combinations_to_use}" "${output_dir}" local_generated_sources)
    endforeach()

    set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
    list(LENGTH local_generated_sources cuda_generated_count)
    message(STATUS "✅ CUDA combination template processing complete: ${cuda_generated_count} files generated")
endfunction()

# ============================================================================
# Main template generation execution
# ============================================================================

function(execute_template_generation)
    message(STATUS "🔧 STARTING UNIFIED CPU/CUDA TEMPLATE GENERATION")
    set(ALL_GENERATED_SOURCES "")

    # Use the combinations provided by selective rendering
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 ${UNIFIED_COMBINATIONS_3})
    else()
        message(WARNING "⚠️ No 3-type combinations provided by selective rendering")
        set(COMBINATIONS_3 "")
    endif()
    
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 ${UNIFIED_COMBINATIONS_2})
    else()
        message(WARNING "⚠️ No 2-type combinations provided by selective rendering")
        set(COMBINATIONS_2 "")
    endif()

    # Derive 1-type combinations from 2-type combinations
    set(COMBINATIONS_1 "")
    foreach(combination ${COMBINATIONS_2})
        string(REPLACE "," ";" combo_parts "${combination}")
        list(GET combo_parts 0 comb1)
        list(FIND COMBINATIONS_1 "${comb1}" found_idx)
        if(found_idx EQUAL -1)
            list(APPEND COMBINATIONS_1 "${comb1}")
        endif()
        list(GET combo_parts 1 comb2)
        list(FIND COMBINATIONS_1 "${comb2}" found_idx)
        if(found_idx EQUAL -1)
            list(APPEND COMBINATIONS_1 "${comb2}")
        endif()
    endforeach()

    # Report what we're working with
    list(LENGTH COMBINATIONS_1 combo_1_count)
    list(LENGTH COMBINATIONS_2 combo_2_count)
    list(LENGTH COMBINATIONS_3 combo_3_count)
    message(STATUS "📊 Template combinations from selective rendering:")
    message(STATUS "   - 1-type: ${combo_1_count} combinations")
    message(STATUS "   - 2-type: ${combo_2_count} combinations")
    message(STATUS "   - 3-type: ${combo_3_count} combinations")

    if(SD_CUDA)
        message(STATUS "🚀 Processing CUDA templates...")

        # Process legacy CUDA compilation units
        file(GLOB_RECURSE CUDA_COMPILATION_UNITS
                ./include/loops/cuda/compilation_units/*.cu.in
                ./include/ops/impl/compilation_units/*.cpp.in)
        foreach(FL_ITEM ${CUDA_COMPILATION_UNITS})
            genCompilation(${FL_ITEM} ALL_GENERATED_SOURCES)
        endforeach()

        set(CUDA_INST_DIR "${CMAKE_BINARY_DIR}/cuda_instantiations")
        file(MAKE_DIRECTORY "${CUDA_INST_DIR}")

        # Process CUDA combination templates
        process_cuda_comb_templates("${CUDA_INST_DIR}" ALL_GENERATED_SOURCES)
        
    else()
        message(STATUS "🖥️  Processing CPU templates...")

        # Process legacy CPU compilation units
        file(GLOB_RECURSE REGULAR_COMPILATION_UNITS
                ./include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
                ./include/helpers/cpu/loops/*.cpp.in)
        file(GLOB_RECURSE EXCLUDED_OLD_TEMPLATES
                ./include/loops/cpu/compilation_units/scalar_p.cpp.in
                ./include/loops/cpu/compilation_units/broadcast_p.cpp.in)
        file(GLOB_RECURSE REMAINING_COMPILATION_UNITS
                ./include/loops/cpu/compilation_units/*.cpp.in)
        if(EXCLUDED_OLD_TEMPLATES)
            list(REMOVE_ITEM REMAINING_COMPILATION_UNITS ${EXCLUDED_OLD_TEMPLATES})
        endif()
        list(APPEND REGULAR_COMPILATION_UNITS ${REMAINING_COMPILATION_UNITS})
        foreach(FL_ITEM ${REGULAR_COMPILATION_UNITS})
            genCompilation(${FL_ITEM} ALL_GENERATED_SOURCES)
        endforeach()

        set(CPU_INST_DIR "${CMAKE_BINARY_DIR}/cpu_instantiations")
        file(MAKE_DIRECTORY "${CPU_INST_DIR}")

        # Generate same-type 3-type combinations for BUILD_SINGLE_SELECTOR_THRICE templates
        if(DEFINED UNIFIED_COMBINATIONS_3_SAME)
            set(COMBINATIONS_3_SAME ${UNIFIED_COMBINATIONS_3_SAME})
        else()
            # Fallback: generate same-type combinations from COMBINATIONS_1
            set(COMBINATIONS_3_SAME "")
            foreach(idx ${COMBINATIONS_1})
                list(APPEND COMBINATIONS_3_SAME "${idx},${idx},${idx}")
            endforeach()
        endif()
        list(LENGTH COMBINATIONS_3_SAME combo_3_same_count)
        message(STATUS "   - 3-type same-type: ${combo_3_same_count} combinations")

        # Process all CPU template files

        # 3-type same-type templates (pairwise, scalar, broadcast use BUILD_SINGLE_SELECTOR_THRICE)
        # These only dispatch (T, T, T) at runtime, so cross-type combinations are wasted
        set(TEMPLATES_3_SAME
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/scalar_instantiation_template_3.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/broadcast_instantiation_template_3.cpp.in")

        foreach(TEMPLATE_FILE ${TEMPLATES_3_SAME})
            if(EXISTS "${TEMPLATE_FILE}")
                message(STATUS "   Processing 3-type SAME-TYPE template: ${TEMPLATE_FILE} (${combo_3_same_count} combinations instead of ${combo_3_count})")
                create_direct_instantiation_file("${TEMPLATE_FILE}" "${COMBINATIONS_3_SAME}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
            else()
                message(WARNING "   Template file not found: ${TEMPLATE_FILE}")
            endif()
        endforeach()

        # 2-type templates (use COMBINATIONS_2)
        set(TEMPLATES_2
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/broadcast_bool_instantiation_template_2.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/indexreduction_instantiation_template_2.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/reduce3_instantiation_template_2.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/reduce_float_instantiation_template_2.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/reduce_bool_instantiation_template_2.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/reduce_long_instantiation_template_2.cpp.in")
                
        foreach(TEMPLATE_FILE ${TEMPLATES_2})
            if(EXISTS "${TEMPLATE_FILE}")
                message(STATUS "   Processing 2-type template: ${TEMPLATE_FILE}")
                create_direct_instantiation_file("${TEMPLATE_FILE}" "${COMBINATIONS_2}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
            else()
                message(WARNING "   Template file not found: ${TEMPLATE_FILE}")
            endif()
        endforeach()

        # 1-type templates (use COMBINATIONS_1)
        set(TEMPLATES_1
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/broadcast_int_instantiation_template_1.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/random_instantiation_template_1.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/reduce_same_instantiation_template_1.cpp.in")

        foreach(TEMPLATE_FILE ${TEMPLATES_1})
            if(EXISTS "${TEMPLATE_FILE}")
                message(STATUS "   Processing 1-type template: ${TEMPLATE_FILE}")
                create_direct_instantiation_file("${TEMPLATE_FILE}" "${COMBINATIONS_1}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
            else()
                message(WARNING "   Template file not found: ${TEMPLATE_FILE}")
            endif()
        endforeach()

        # Process specials templates
        set(SPECIALS_DOUBLE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_double.cpp.in")
        if(EXISTS "${SPECIALS_DOUBLE_TEMPLATE}")
            message(STATUS "   Processing specials_double template")
            create_direct_instantiation_file("${SPECIALS_DOUBLE_TEMPLATE}" "${COMBINATIONS_2}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
        endif()

        set(SPECIALS_SINGLE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_single.cpp.in")
        if(EXISTS "${SPECIALS_SINGLE_TEMPLATE}")
            message(STATUS "   Processing specials_single template")
            create_direct_instantiation_file("${SPECIALS_SINGLE_TEMPLATE}" "${COMBINATIONS_1}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
        endif()
    endif()
    
    # Final reporting
    set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} CACHE INTERNAL "Template-generated source files" FORCE)
    list(LENGTH ALL_GENERATED_SOURCES total_count)

    if(SD_CUDA)
        message(STATUS "✅ CUDA template processing complete: ${total_count} files generated")
    else()
        message(STATUS "✅ CPU template processing complete: ${total_count} files generated")
    endif()

    # Export to parent scope
    set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} PARENT_SCOPE)
endfunction()

# ============================================================================
# Main entry point for external calls
# ============================================================================

function(setup_template_processing)
    # Check if selective rendering has already been executed
    get_property(cached_sources CACHE CUSTOMOPS_GENERIC_SOURCES PROPERTY VALUE)
    if(cached_sources)
        # Use cached results
        set(CUSTOMOPS_GENERIC_SOURCES ${cached_sources} PARENT_SCOPE)
        list(LENGTH cached_sources cached_count)
        if(SD_CUDA)
            message(STATUS "🔄 Using cached CUDA template processing results (${cached_count} files)")
        else()
            message(STATUS "🔄 Using cached CPU template processing results (${cached_count} files)")
        endif()
    else()
        # Execute selective rendering + template processing
        execute_template_processing_with_selective_rendering()
        set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
    endif()
endfunction()

message(STATUS "✅ Enhanced TemplateProcessing.cmake loaded with comprehensive type handling - functions defined, execution deferred")
