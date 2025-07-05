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

# TemplateProcessing.cmake - UPDATED TO SUPPORT CUDA TEMPLATE PARITY
# IMPORTANT: This file now defines functions. Execution happens in setup_template_processing()

set(CUSTOMOPS_GENERIC_SOURCES "" CACHE INTERNAL "Template-generated source files")

set(CHUNK_TARGET_INSTANTIATIONS "5" CACHE STRING "Target template instantiations per chunk file (1-20)")
set(CHUNK_MAX_INSTANTIATIONS "10" CACHE STRING "Maximum template instantiations per chunk file")
set(USE_MULTI_PASS_GENERATION "ON" CACHE STRING "Use multi-pass generation (ON/OFF/AUTO)")
set(MULTI_PASS_CHUNK_SIZE "20" CACHE STRING "Chunk size for direct instantiation files")

# Enable selective rendering diagnostics in debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SRCORE_ENABLE_DIAGNOSTICS ON)
endif()

# Auto-detect memory and adjust chunking - BUT ONLY WHEN EXPLICITLY CALLED
function(configure_memory_chunking)
    if(NOT CHUNK_TARGET_INSTANTIATIONS_SET)
        cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
        if(AVAILABLE_MEMORY LESS 4000)
            set(CHUNK_TARGET_INSTANTIATIONS 3 PARENT_SCOPE)
            set(MULTI_PASS_CHUNK_SIZE 25 PARENT_SCOPE)
            message(STATUS "Low memory detected: Conservative chunking (chunks=3, direct=25)")
        elseif(AVAILABLE_MEMORY LESS 8000)
            set(CHUNK_TARGET_INSTANTIATIONS 6 PARENT_SCOPE)
            set(MULTI_PASS_CHUNK_SIZE 35 PARENT_SCOPE)
            message(STATUS "Medium memory detected: Moderate chunking (chunks=6, direct=35)")
        elseif(AVAILABLE_MEMORY LESS 16000)
            set(CHUNK_TARGET_INSTANTIATIONS 10 PARENT_SCOPE)
            set(MULTI_PASS_CHUNK_SIZE 50 PARENT_SCOPE)
            message(STATUS "High memory detected: Balanced chunking (chunks=10, direct=50)")
        else()
            set(CHUNK_TARGET_INSTANTIATIONS 12 PARENT_SCOPE)
            set(MULTI_PASS_CHUNK_SIZE 60 PARENT_SCOPE)
            message(STATUS "Very high memory detected: Optimized chunking (chunks=12, direct=60)")
        endif()
        set(CHUNK_TARGET_INSTANTIATIONS_SET TRUE CACHE INTERNAL "Memory auto-detection completed")

        if(USE_MULTI_PASS_GENERATION STREQUAL "AUTO")
            if(AVAILABLE_MEMORY LESS 3000 OR DEFINED ENV{CI})
                set(USE_MULTI_PASS_GENERATION ON PARENT_SCOPE)
                message(STATUS "Auto-detected low memory/CI: enabling multi-pass")
            else()
                set(USE_MULTI_PASS_GENERATION OFF PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

# Initialize selective rendering and execute template processing - CALLED FROM MainBuildFlow
function(execute_template_processing_with_selective_rendering)
    message(STATUS "🔧 Starting Selective Template Processing...")

    # Step 1: Include selective rendering core
    include(${CMAKE_CURRENT_LIST_DIR}/SelectiveRenderingCore.cmake)

    # Step 2: Setup selective rendering FIRST
    setup_selective_rendering_unified_safe()

    # Step 3: Verify selective rendering worked
    if(NOT DEFINED UNIFIED_COMBINATIONS_2 OR NOT DEFINED UNIFIED_COMBINATIONS_3)
        message(WARNING "⚠️ Selective rendering failed to initialize, falling back to emergency mode")
        srcore_emergency_fallback()
    endif()

    # Step 4: Configure memory chunking
    configure_memory_chunking()

    # Step 5: Display filtering results
    list(LENGTH UNIFIED_COMBINATIONS_2 filtered_2_count)
    list(LENGTH UNIFIED_COMBINATIONS_3 filtered_3_count)
    message(STATUS "🎯 Selective Rendering Results:")
    message(STATUS "   - 2-type combinations: ${filtered_2_count} (filtered from 25)")
    message(STATUS "   - 3-type combinations: ${filtered_3_count} (filtered from 125)")
    math(EXPR savings_percent_3 "100 - (100 * ${filtered_3_count} / 125)")
    message(STATUS "   - Template instantiation reduction: ${savings_percent_3}%")

    message(STATUS "🔧 Template processing: Multi-pass=${USE_MULTI_PASS_GENERATION}, Chunks=${CHUNK_TARGET_INSTANTIATIONS}")

    # Step 6: Now execute the actual template processing
    execute_template_generation()

    # Step 7: Propagate results to parent scope
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

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
        # Fallback to hardcoded values
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

# UPDATED: Enhanced direct instantiation function with CUDA support
function(create_direct_instantiation_file template_file combinations output_dir generated_sources_var)
    get_filename_component(template_name ${template_file} NAME_WE)
    file(READ "${template_file}" template_content)
    string(REGEX MATCHALL "#include[^\n]*" includes "${template_content}")

    # Determine if this is a CUDA file
    set(IS_CUDA_FILE FALSE)
    if(template_file MATCHES "\\.cu\\.in$")
        set(IS_CUDA_FILE TRUE)
        set(file_extension "cu")
    else()
        set(file_extension "cpp")
    endif()

    set(file_header "")
    foreach(inc ${includes})
        string(APPEND file_header "${inc}\n")
    endforeach()
    string(APPEND file_header "\n// Direct instantiations - generated with selective rendering\n")
    string(APPEND file_header "// Platform: ${IS_CUDA_FILE}")
    string(APPEND file_header "// Combinations: ${combinations}\n\n")

    # Split types into parts based on selective rendering results
    set(TYPE_PART_0 "bool" "float16" "bfloat16" "float" "double")
    set(TYPE_PART_1 "int8_t" "int16_t" "int32_t" "sd::LongType")
    set(TYPE_PART_2 "uint8_t" "uint16_t" "uint32_t" "uint64_t")

    set(chunk_content "${file_header}")
    set(instantiation_count 0)
    set(chunk_index 0)
    set(local_generated_sources ${${generated_sources_var}})
    set(total_instantiations 0)

    foreach(combination ${combinations})
        string(REPLACE "," ";" parts "${combination}")
        list(LENGTH parts parts_count)

        if(parts_count EQUAL 1)
            # Handle 1-type combinations
            list(GET parts 0 p1)
            set(types1 ${TYPE_PART_${p1}})

            # Calculate adaptive chunk size for 1-type combinations
            set(float_count 0)
            if(p1 EQUAL 0)
                set(float_count 1)
            endif()

            if(float_count EQUAL 1)
                math(EXPR current_chunk_limit "${MULTI_PASS_CHUNK_SIZE} * 80 / 100")
            else()
                set(current_chunk_limit ${MULTI_PASS_CHUNK_SIZE})
            endif()

            if(current_chunk_limit LESS 5)
                set(current_chunk_limit 5)
            endif()

            foreach(t1 ${types1})
                # Generate CUDA-specific or CPU-specific instantiations
                if(IS_CUDA_FILE)
                    if(template_name MATCHES "random")
                        string(APPEND chunk_content "template void functions::random::RandomFunction<${t1}>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);\n")
                        string(APPEND chunk_content "template void functions::random::RandomFunction<${t1}>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);\n")
                        string(APPEND chunk_content "template void functions::random::RandomFunction<${t1}>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void const* vy, sd::LongType const* yShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments);\n")
                    elseif(template_name MATCHES "reduce_same")
                        string(APPEND chunk_content "template void functions::reduce::ReduceSameFunction<${t1}>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* dXShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, const sd::LongType* dims);\n")
                        string(APPEND chunk_content "template void functions::reduce::ReduceSameFunction<${t1}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* zShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);\n")
                    else()
                        string(APPEND chunk_content "template class ${template_name}<${t1}>;\n")
                    endif()
                else()
                    # CPU instantiations
                    if(template_name MATCHES "specials_single")
                        string(APPEND chunk_content "namespace sd {\n")
                        string(APPEND chunk_content "template class sd::SpecialMethods<${t1}>;\n")
                        string(APPEND chunk_content "}\n")
                    else()
                        string(APPEND chunk_content "template class ${template_name}<${t1}>;\n")
                    endif()
                endif()

                math(EXPR instantiation_count "${instantiation_count} + 1")
                math(EXPR total_instantiations "${total_instantiations} + 1")
                if(instantiation_count GREATER_EQUAL current_chunk_limit)
                    set(chunk_file "${output_dir}/${template_name}_direct_${chunk_index}.${file_extension}")
                    file(WRITE "${chunk_file}" "${chunk_content}")
                    list(APPEND local_generated_sources "${chunk_file}")
                    set(chunk_content "${file_header}")
                    set(instantiation_count 0)
                    math(EXPR chunk_index "${chunk_index} + 1")
                endif()
            endforeach()

        elseif(parts_count EQUAL 2)
            # Handle 2-type combinations
            list(GET parts 0 p1)
            list(GET parts 1 p2)
            set(types1 ${TYPE_PART_${p1}})
            set(types2 ${TYPE_PART_${p2}})

            # Calculate adaptive chunk size for 2-type combinations
            set(float_count 0)
            foreach(part_idx ${p1} ${p2})
                if(part_idx EQUAL 0)
                    math(EXPR float_count "${float_count} + 1")
                endif()
            endforeach()

            if(float_count EQUAL 2)
                math(EXPR current_chunk_limit "${MULTI_PASS_CHUNK_SIZE} * 60 / 100")
            elseif(float_count EQUAL 1)
                math(EXPR current_chunk_limit "${MULTI_PASS_CHUNK_SIZE} * 80 / 100")
            else()
                set(current_chunk_limit ${MULTI_PASS_CHUNK_SIZE})
            endif()

            if(current_chunk_limit LESS 5)
                set(current_chunk_limit 5)
            endif()

            foreach(t1 ${types1})
                foreach(t2 ${types2})
                    # Generate CUDA-specific or CPU-specific instantiations
                    if(IS_CUDA_FILE)
                        if(template_name MATCHES "reduce3")
                            string(APPEND chunk_content "template void functions::reduce3::Reduce3<${t1}, ${t2}>::exec(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationPointer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo, sd::LongType const* yTadOffsets);\n")
                            string(APPEND chunk_content "template void functions::reduce3::Reduce3<${t1}, ${t2}>::execAll(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* dimension, long long int dimensionLength, int postProcessOrNot, sd::LongType* allocationPointer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo, sd::LongType const* yTadOffsets);\n")
                            string(APPEND chunk_content "template void functions::reduce3::Reduce3<${t1}, ${t2}>::execScalar(dim3 launchDims, cudaStream_t* stream, int opNum, void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* extraParams, void* vz, sd::LongType const* zShapeInfo, sd::LongType* allocationPointer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo);\n")
                        elseif(template_name MATCHES "reduce_float")
                            string(APPEND chunk_content "template void functions::reduce::ReduceFloatFunction<${t1}, ${t2}>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* dXShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, const sd::LongType* dims);\n")
                            string(APPEND chunk_content "template void functions::reduce::ReduceFloatFunction<${t1}, ${t2}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* dZShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, long long int dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);\n")
                        elseif(template_name MATCHES "indexreduce")
                            string(APPEND chunk_content "template void functions::indexreduce::IndexReduce<${t1}, ${t2}>::executeIndexReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, void const* dx, sd::LongType const* xShapeInfo, sd::LongType xRank, void* extraParams, void* result, sd::LongType const* zShapeInfo, sd::LongType zRank, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationBuffer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets);\n")
                            string(APPEND chunk_content "template void functions::indexreduce::IndexReduce<${t1}, ${t2}>::executeIndexReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, void const* dx, sd::LongType const* xShapeInfo, sd::LongType xRank, void* extraParams, void* result, sd::LongType const* zShapeInfo, sd::LongType zRank, sd::LongType* dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType* allocationBuffer, void* reductionBuffer, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets);\n")
                        elseif(template_name MATCHES "reduce_bool")
                            string(APPEND chunk_content "template void functions::reduce::ReduceBoolFunction<${t1}, ${t2}>::execReduce(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, sd::LongType* dXShapeInfo, sd::LongType* hXShapeInfo, void* extraParams, void* vreductionBuffer, void* z, sd::LongType* dZShapeInfo, sd::LongType* hZShapeInfo, sd::LongType* dims);\n")
                            string(APPEND chunk_content "template void functions::reduce::ReduceBoolFunction<${t1}, ${t2}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, const sd::LongType* xShapeInfo, const sd::LongType* hXShapeInfo, void* extraParams, void* z, const sd::LongType* zShapeInfo, const sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, const sd::LongType* tadOnlyShapeInfo);\n")
                        elseif(template_name MATCHES "reduce_long")
                            string(APPEND chunk_content "template void functions::reduce::ReduceLongFunction<${t1}, ${t2}>::execReduce(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx, sd::LongType *dXShapeInfo, sd::LongType *hXShapeInfo, void *extraParams, void *vreductionBuffer, void *vz, sd::LongType *dZShapeInfo, sd::LongType *hZShapeInfo, sd::LongType *dims);\n")
                            string(APPEND chunk_content "template void functions::reduce::ReduceLongFunction<${t1}, ${t2}>::execReduceScalar(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x, sd::LongType* xShapeInfo, sd::LongType* hXShapeInfo, void* extraParams, void* z, sd::LongType* zShapeInfo, sd::LongType* hZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, void* reductionBuffer, sd::LongType* tadOnlyShapeInfo);\n")
                        else()
                            string(APPEND chunk_content "template class ${template_name}<${t1}, ${t2}>;\n")
                        endif()
                    else()
                        # CPU instantiations
                        if(template_name MATCHES "indexreduction")
                            string(APPEND chunk_content "template class sd::IndexReductionLoops<${t1}, ${t2}>;\n")
                        elseif(template_name MATCHES "specials_double")
                            string(APPEND chunk_content "namespace sd {\n")
                            string(APPEND chunk_content "template class sd::DoubleMethods<${t1}, ${t2}>;\n")
                            string(APPEND chunk_content "template void sd::SpecialTypeConverter::convertGeneric<${t1}, ${t2}>(sd::Pointer * extras, void *dx, sd::LongType N, void *dz);\n")
                            string(APPEND chunk_content "}\n")
                        else()
                            string(APPEND chunk_content "template class ${template_name}<${t1}, ${t2}>;\n")
                        endif()
                    endif()

                    math(EXPR instantiation_count "${instantiation_count} + 1")
                    math(EXPR total_instantiations "${total_instantiations} + 1")
                    if(instantiation_count GREATER_EQUAL current_chunk_limit)
                        set(chunk_file "${output_dir}/${template_name}_direct_${chunk_index}.${file_extension}")
                        file(WRITE "${chunk_file}" "${chunk_content}")
                        list(APPEND local_generated_sources "${chunk_file}")
                        set(chunk_content "${file_header}")
                        set(instantiation_count 0)
                        math(EXPR chunk_index "${chunk_index} + 1")
                    endif()
                endforeach()
            endforeach()

        else()
            # Handle 3-type combinations
            list(GET parts 0 p1)
            list(GET parts 1 p2)
            list(GET parts 2 p3)
            set(types1 ${TYPE_PART_${p1}})
            set(types2 ${TYPE_PART_${p2}})
            set(types3 ${TYPE_PART_${p3}})

            # Calculate adaptive chunk size based on type complexity
            set(float_count 0)
            foreach(part_idx ${p1} ${p2} ${p3})
                if(part_idx EQUAL 0)
                    math(EXPR float_count "${float_count} + 1")
                endif()
            endforeach()

            # Scale chunk size based on floating-point density
            if(float_count EQUAL 3)
                math(EXPR current_chunk_limit "${MULTI_PASS_CHUNK_SIZE} * 50 / 100")
            elseif(float_count EQUAL 2)
                math(EXPR current_chunk_limit "${MULTI_PASS_CHUNK_SIZE} * 70 / 100")
            elseif(float_count EQUAL 1)
                math(EXPR current_chunk_limit "${MULTI_PASS_CHUNK_SIZE} * 85 / 100")
            else()
                set(current_chunk_limit ${MULTI_PASS_CHUNK_SIZE})
            endif()

            if(current_chunk_limit LESS 5)
                set(current_chunk_limit 5)
            endif()

            foreach(t1 ${types1})
                foreach(t2 ${types2})
                    foreach(t3 ${types3})
                        # Generate CUDA-specific or CPU-specific instantiations
                        if(IS_CUDA_FILE)
                            if(template_name MATCHES "pairwise")
                                string(APPEND chunk_content "template void functions::pairwise_transforms::PairWiseTransform<${t1}, ${t2}, ${t3}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo, void *vextraParams);\n")
                            elseif(template_name MATCHES "scalar")
                                string(APPEND chunk_content "template void functions::scalar::ScalarTransform<${t1}, ${t2}, ${t3}>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, const sd::LongType *hxShapeInfo, void *vz, const sd::LongType *zShapeInfo, const sd::LongType *hzShapeInfo, const void *vscalar, void *vextraParams);\n")
                                string(APPEND chunk_content "template void functions::scalar::ScalarTransform<${t1}, ${t2}, ${t3}>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t* stream, int opNum, const void *vx, const sd::LongType *xShapeInfo, void *vz, const sd::LongType *zShapeInfo, const void *vscalars, void *vextraParams, sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);\n")
                            elseif(template_name MATCHES "broadcast")
                                string(APPEND chunk_content "template void functions::broadcast::Broadcast<${t1}, ${t2}, ${t3}>::execBroadcast(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShapeInfo, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ);\n")
                                string(APPEND chunk_content "template void functions::broadcast::Broadcast<${t1}, ${t2}, ${t3}>::execBroadcast(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShapeInfo, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo);\n")
                                string(APPEND chunk_content "template void functions::broadcast::Broadcast<${t1}, ${t2}, ${t3}>::execInverseBroadcast(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShapeInfo, void const* y, sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ);\n")
                            else()
                                string(APPEND chunk_content "template class ${template_name}<${t1}, ${t2}, ${t3}>;\n")
                            endif()
                        else()
                            # CPU instantiations
                            if(template_name MATCHES "pairwise")
                                string(APPEND chunk_content "template void functions::pairwise_transforms::PairWiseTransform<${t1}, ${t2}, ${t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo, void *extraParams, sd::LongType start, sd::LongType stop);\n")
                            elseif(template_name MATCHES "scalar")
                                string(APPEND chunk_content "template void functions::scalar::ScalarTransform<${t1}, ${t2}, ${t3}>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *result, const sd::LongType *resultShapeInfo, const void *scalar, void *extraParams, sd::LongType start, sd::LongType stop);\n")
                            elseif(template_name MATCHES "broadcast")
                                string(APPEND chunk_content "template void functions::broadcast::Broadcast<${t1}, ${t2}, ${t3}>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);\n")
                            else()
                                string(APPEND chunk_content "template class ${template_name}<${t1}, ${t2}, ${t3}>;\n")
                            endif()
                        endif()

                        math(EXPR instantiation_count "${instantiation_count} + 1")
                        math(EXPR total_instantiations "${total_instantiations} + 1")
                        if(instantiation_count GREATER_EQUAL current_chunk_limit)
                            set(chunk_file "${output_dir}/${template_name}_direct_${chunk_index}.${file_extension}")
                            file(WRITE "${chunk_file}" "${chunk_content}")
                            list(APPEND local_generated_sources "${chunk_file}")
                            set(chunk_content "${file_header}")
                            set(instantiation_count 0)
                            math(EXPR chunk_index "${chunk_index} + 1")
                        endif()
                    endforeach()
                endforeach()
            endforeach()
        endif()
    endforeach()

    if(instantiation_count GREATER 0)
        set(chunk_file "${output_dir}/${template_name}_direct_final.${file_extension}")
        file(WRITE "${chunk_file}" "${chunk_content}")
        list(APPEND local_generated_sources "${chunk_file}")
    endif()
    set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)

    # Enhanced logging for CUDA vs CPU
    if(IS_CUDA_FILE)
        message(STATUS "✅ Generated CUDA instantiation files for ${template_name}: ${total_instantiations} instantiations in ${chunk_index} files")
    else()
        message(STATUS "✅ Generated CPU instantiation files for ${template_name}: ${total_instantiations} instantiations in ${chunk_index} files")
    endif()
endfunction()

function(removeFileIfExcluded)
    cmake_parse_arguments(PARSED_ARGS "" "FILE_ITEM" "LIST_ITEM" ${ARGN})
    file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
    string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)
    if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
        set(local_list ${${PARSED_ARGS_LIST_ITEM}})
        set(file_removed FALSE)
        foreach(OP ${SD_OPS_LIST})
            string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
            if(${NOT_EXCLUDED_OP_IDX} LESS 0)
                list(REMOVE_ITEM local_list "${PARSED_ARGS_FILE_ITEM}")
                set(file_removed TRUE)
                break()
            endif()
        endforeach()
        if(file_removed)
            set(${PARSED_ARGS_LIST_ITEM} ${local_list} PARENT_SCOPE)
        endif()
    endif()
endfunction()

# NEW: CUDA template processing function
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

    # USE SELECTIVE RENDERING COMBINATIONS
    # USE SELECTIVE RENDERING COMBINATIONS - EXACT CPU MIMIC
    if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
        message(STATUS "✅ Using selective rendering combinations for CUDA (CPU parity)")
        set(COMBINATIONS_1 "")
        set(COMBINATIONS_2 ${UNIFIED_COMBINATIONS_2})
        set(COMBINATIONS_3 ${UNIFIED_COMBINATIONS_3})

        # Verify combinations are actually filtered
        list(LENGTH UNIFIED_COMBINATIONS_2 actual_2_count)
        list(LENGTH UNIFIED_COMBINATIONS_3 actual_3_count)
        if(actual_2_count EQUAL 25 OR actual_3_count EQUAL 125)
            message(WARNING "⚠️ Combinations appear unfiltered - selective rendering may have failed")
        endif()

        # Generate 1-type combinations from 2-type combinations
        # Generate 1-type combinations from 2-type combinations (CPU parity)
        foreach(combination ${COMBINATIONS_2})
            string(REPLACE "," ";" combo_parts "${combination}")
            list(GET combo_parts 0 comb1)
            list(FIND COMBINATIONS_1 "${comb1}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND COMBINATIONS_1 "${comb1}")
            endif()
        endforeach()

        # Verify 1-type derivation worked
        list(LENGTH COMBINATIONS_1 derived_1_count)
        if(derived_1_count EQUAL 0)
            message(FATAL_ERROR "❌ Failed to derive 1-type combinations from 2-type")
        endif()


        list(LENGTH COMBINATIONS_1 combo_1_count)
        list(LENGTH COMBINATIONS_2 combo_2_count)
        list(LENGTH COMBINATIONS_3 combo_3_count)
        message(STATUS "🎯 Using selective CUDA combinations: 1-type=${combo_1_count}, 2-type=${combo_2_count}, 3-type=${combo_3_count}")
    else()
        message(FATAL_ERROR "❌ Selective rendering REQUIRED for CUDA template parity - cannot proceed with fallback")
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

# UPDATED: The main template processing function with CUDA support
function(execute_template_generation)
    message(STATUS "🔧 STARTING UNIFIED CPU/CUDA TEMPLATE GENERATION")
    set(ALL_GENERATED_SOURCES "")

    if(SD_CUDA)
        message(STATUS "🚀 Processing CUDA templates with selective rendering...")

        # Process legacy CUDA compilation units (existing system)
        file(GLOB_RECURSE CUDA_COMPILATION_UNITS
                ./include/loops/cuda/compilation_units/*.cu.in
                ./include/ops/impl/compilation_units/*.cpp.in)
        foreach(FL_ITEM ${CUDA_COMPILATION_UNITS})
            genCompilation(${FL_ITEM} ALL_GENERATED_SOURCES)
        endforeach()

        set(CUDA_INST_DIR "${CMAKE_BINARY_DIR}/cuda_instantiations")
        file(MAKE_DIRECTORY "${CUDA_INST_DIR}")

        if(NOT DEFINED UNIFIED_COMBINATIONS_2 OR NOT DEFINED UNIFIED_COMBINATIONS_3)
            message(FATAL_ERROR "❌ CUDA template processing requires selective rendering combinations!")
        endif()

        # Log the filtering being applied
        list(LENGTH UNIFIED_COMBINATIONS_2 cuda_2_count)
        list(LENGTH UNIFIED_COMBINATIONS_3 cuda_3_count)
        message(STATUS "🎯 CUDA using selective combinations: 2-type=${cuda_2_count}, 3-type=${cuda_3_count}")

        process_cuda_comb_templates("${CUDA_INST_DIR}" ALL_GENERATED_SOURCES)
    else()
        message(STATUS "🖥️  Processing CPU templates with selective rendering...")

        # Process legacy CPU compilation units (existing system)
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

        # USE SELECTIVE RENDERING COMBINATIONS
        if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
            message(STATUS "✅ Using selective rendering combinations")
            set(COMBINATIONS_2 ${UNIFIED_COMBINATIONS_2})
            set(COMBINATIONS_3 ${UNIFIED_COMBINATIONS_3})
            list(LENGTH COMBINATIONS_2 combo_2_count)
            list(LENGTH COMBINATIONS_3 combo_3_count)
            message(STATUS "   - Using ${combo_2_count} filtered 2-type combinations")
            message(STATUS "   - Using ${combo_3_count} filtered 3-type combinations")
        else()
            message(WARNING "⚠️ Selective rendering not available, using fallback combinations")
            # Reduced fallback set - not all combinations
            set(COMBINATIONS_3
                    "0,0,0" "0,0,1" "0,1,0" "0,1,1"
                    "1,0,0" "1,0,1" "1,1,0" "1,1,1"
                    "0,0,2" "1,1,2" "2,2,2")
            set(COMBINATIONS_2 "0,0" "0,1" "1,0" "1,1" "2,2")
            list(LENGTH COMBINATIONS_2 fallback_2_count)
            list(LENGTH COMBINATIONS_3 fallback_3_count)
            message(STATUS "   - Using ${fallback_2_count} fallback 2-type combinations")
            message(STATUS "   - Using ${fallback_3_count} fallback 3-type combinations")
        endif()

        set(CPU_INST_DIR "${CMAKE_BINARY_DIR}/cpu_instantiations")
        file(MAKE_DIRECTORY "${CPU_INST_DIR}")

        # Process 3-type templates with direct instantiation
        set(TEMPLATES_3
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/scalar_instantiation_template_3.cpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/broadcast_instantiation_template_3.cpp.in")

        foreach(TEMPLATE_FILE ${TEMPLATES_3})
            create_direct_instantiation_file("${TEMPLATE_FILE}" "${COMBINATIONS_3}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
        endforeach()

        # Process 2-type templates with direct instantiation (NOT macro system)
        set(TEMPLATES_2_DIRECT
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/indexreduction_instantiation_template_2.cpp.in")
        foreach(TEMPLATE_FILE ${TEMPLATES_2_DIRECT})
            create_direct_instantiation_file("${TEMPLATE_FILE}" "${COMBINATIONS_2}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
        endforeach()

        # Process specials templates with direct instantiation (NOT macro system)
        set(SPECIALS_DOUBLE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_double.cpp.in")
        if(EXISTS "${SPECIALS_DOUBLE_TEMPLATE}")
            create_direct_instantiation_file("${SPECIALS_DOUBLE_TEMPLATE}" "${COMBINATIONS_2}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
        endif()

        set(SPECIALS_SINGLE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_single.cpp.in")
        if(EXISTS "${SPECIALS_SINGLE_TEMPLATE}")
            # For single templates, we need to create 1-type combinations from 2-type combinations
            set(COMBINATIONS_1 "")
            foreach(combination ${COMBINATIONS_2})
                string(REPLACE "," ";" combo_parts "${combination}")
                list(GET combo_parts 0 comb1)
                list(FIND COMBINATIONS_1 "${comb1}" found_idx)
                if(found_idx EQUAL -1)
                    list(APPEND COMBINATIONS_1 "${comb1}")
                endif()
            endforeach()
            create_direct_instantiation_file("${SPECIALS_SINGLE_TEMPLATE}" "${COMBINATIONS_1}" "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
        endif()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} CACHE INTERNAL "Template-generated source files" FORCE)
    list(LENGTH ALL_GENERATED_SOURCES total_count)

    # Enhanced final reporting
    if(SD_CUDA)
        message(STATUS "✅ CUDA template processing complete: ${total_count} files generated")
        if(DEFINED UNIFIED_COMBINATIONS_3)
            list(LENGTH UNIFIED_COMBINATIONS_3 final_combo_count)
            math(EXPR reduction_percent "100 - (100 * ${final_combo_count} / 125)")
            message(STATUS "🎯 CUDA template instantiation reduction achieved: ${reduction_percent}% fewer templates")
            message(STATUS "🎯 Generated ${final_combo_count} CUDA combinations instead of 125")
        endif()
    else()
        message(STATUS "✅ CPU template processing complete: ${total_count} files generated")
        if(DEFINED UNIFIED_COMBINATIONS_3)
            list(LENGTH UNIFIED_COMBINATIONS_3 final_combo_count)
            math(EXPR reduction_percent "100 - (100 * ${final_combo_count} / 125)")
            message(STATUS "🎯 CPU template instantiation reduction achieved: ${reduction_percent}% fewer templates")
            message(STATUS "🎯 Generated ${final_combo_count} CPU combinations instead of 125")
        endif()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} PARENT_SCOPE)
endfunction()

# Legacy compatibility function - called from MainBuildFlow.cmake
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

message(STATUS "✅ Enhanced TemplateProcessing.cmake loaded with CUDA/CPU parity - functions defined, execution deferred")