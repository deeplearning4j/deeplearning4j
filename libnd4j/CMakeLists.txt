cmake_minimum_required(VERSION 3.15)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(libnd4j)



set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")

#ensure we create lib files
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_BUILD_TESTS "Build tests" OFF)
option(SD_STATIC_LIB "Build static library" OFF)
option(SD_SHARED_LIB "Build shared library" ON)
option(SD_SANITIZE "Enable Address Sanitizer" OFF)
option(SD_USE_LTO "Use link time optimization" OFF)
option(PRINT_INDICES "Print indices" OFF)
option(PRINT_MATH "Print math operations" OFF)
option(SD_PREPROCESSOR "Use preprocessor" OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)
# GCC specific flag: 	 -finstrument-functions enables call stack logging. Useful for debugging segfaults.
option(SD_GCC_FUNCTRACE "Use call traces" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF)

message("PRINT_INDICES: ${PRINT_INDICES}")
if("${PRINT_INDICES}" STREQUAL "ON")
    message("Added print indices compile definition")
    add_compile_definitions(PRINT_INDICES)
endif()

message("PRINT_MATH: ${PRINT_MATH}")
if("${PRINT_MATH}" STREQUAL "ON")
    message("Added print indices compile definition")
    add_compile_definitions(SD_PRINT_MATH)
endif()


if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
    message("Set optimization for functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL   "0")

else()
    message("Set optimization level for no functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL  "3")
endif()

message("Set default optimization level ${SD_OPTIMIZATION_LEVEL}")
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)
# note we may or may not use a build type called "none" to prevent default injection
# of flags from cmake. We do this when using functrace so we can add symbols
# to a binary but still run from java without freezing.
# Normally, we would just want to use debug build. Running a debug build
# via JNI seems to just freeze though. The goal is to just use tools like
# valgrind or compute-sanitizer or even address sanitizer with symbols
# embedded in a binary but still run code from java.
message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

print_all_variables()

# Define ARM Compute Library URLs based on architecture
# For ARM 32-bit (Raspberry Pi)
set(ARM_COMPUTE_URL_ARMV7 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-armv7a-cpu-bin.tar.gz")
# For ARM 64-bit (Raspberry Pi, Odroid N2)
set(ARM_COMPUTE_URL_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-aarch64-cpu-bin.tar.gz")
# For ARM 64-bit Android
set(ARM_COMPUTE_URL_ANDROID_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-android-aarch64-cpu-bin.tar.gz")

# Set the appropriate URL based on the architecture
# Set the appropriate URL based on the architecture
if(${SD_ARCH} MATCHES "armv7")
    set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
    message("Using ARM Compute Library for ARMv7 32-bit")
    # Corrected block starts here
elseif(${SD_ARCH} MATCHES "armv8-a") # First condition
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (matched armv8-a)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched armv8-a)")
    endif()
elseif(${SD_ARCH} MATCHES "arm64") # Second condition (separate elseif)
    # Same logic as above
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (matched arm64)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched arm64)")
    endif()
    # Original endif for the whole if/elseif chain
endif()

# Find the section related to CUDA configuration around line 167
if(SD_CUDA)
    message("Building with CUDA support")

    set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler")
    set(CMAKE_CUDA_COMPILER_ID_FLAGS "--allow-unsupported-compiler")

    message("Setting CUDA flags to allow unsupported compiler: ${CMAKE_CUDA_FLAGS}")

    add_definitions(-D__CUDABLAS__=true)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "")
    endif()

    enable_language(CUDA)

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message("CUDA include directory: ${CUDA_INCLUDE_DIRS} with CXX compiler ${CMAKE_CXX_COMPILER_ID} SD_GCC_FUNCTRACE=${SD_GCC_FUNCTRACE}")
        include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")

        set(CMAKE_CUDA_FLAGS_DEBUG " -g")
        message("CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

        if("${SD_PTXAS}" STREQUAL "ON")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
        endif()

        if(SD_KEEP_NVCC_OUTPUT)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SD_GCC_FUNCTRACE STREQUAL "ON")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Werror -Wall   -Wno-return-type  -Wno-unknown-pragmas  -Wno-unused-variable -Wno-unused-parameter  -Wreturn-type -W -ggdb3 -fPIC -DSD_GCC_FUNCTRACE=1 -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -finstrument-functions  -O0")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G")
                add_compile_definitions(SD_GCC_FUNCTRACE)
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
            endif()
        endif()

        if(WIN32)
            message("Configuring CUDA libraries for Windows")
            if(NOT DEFINED CUDA_cublas_LIBRARY)
                set(CUDA_cublas_LIBRARY "${CUDA_HOME}/lib/x64/cublas.lib")
            endif()
            if(NOT DEFINED CUDA_cusolver_LIBRARY)
                set(CUDA_cusolver_LIBRARY "${CUDA_HOME}/lib/x64/cusolver.lib")
            endif()
        endif()

        if("${SD_ARCH}" MATCHES "armv8-a" AND UNIX)
            message("Adding Jetson Nano specific settings")
            if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CUDA_cublas_LIBRARY}" MATCHES ".*NOTFOUND.*" )
                message("Setting cublas library manually")
                set(CUDA_cublas_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so" CACHE STRING "CUDA CUBLAS LIB" FORCE)
                unset(CUDA_cublas-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            if(NOT DEFINED CUDA_cusolver_LIBRARY OR CUDA_cusolver_LIBRARY MATCHES ".*NOTFOUND.*")
                message("Setting cusolver library manually for Jetson Nano")
                set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so" CACHE STRING "CUDA CUSOLVER LIB" FORCE)
                unset(CUDA_cusolver-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            message("Jetson Nano cublas library: ${CUDA_cublas_LIBRARY}, cusolver library: ${CUDA_cusolver_LIBRARY}")
        endif()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=128")

        string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
        if(COMPUTE_CMP STREQUAL "all")
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
                set(CUDA_ARCH_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80")
            else()
                set(CUDA_ARCH_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90")
            endif()
            set(CMAKE_CUDA_ARCHITECTURES "${CUDA_DEFAULT_ARCHITECTURES}")
        elseif(COMPUTE_CMP STREQUAL "auto")
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
            set(CMAKE_CUDA_ARCHITECTURES "${CUDA_DEFAULT_ARCHITECTURES}")
        elseif(COMPUTE_CMP MATCHES "^[0-9]+$")
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE}")
            set(CMAKE_CUDA_ARCHITECTURES "${COMPUTE}")
        else()
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
        endif()

        string(REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(NOT "${CUDA_ARCH_FLAGS}" STREQUAL "")
                set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS} CACHE STRING "CUDA architectures" FORCE)
            else()
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
                    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 CACHE STRING "CUDA architectures for CUDA 11.x compilers" FORCE)
                else()
                    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 CACHE STRING "CUDA architectures for nvcc 12 and up" FORCE)
                endif()
                add_compile_definitions(CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
            endif()
        endif()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudart -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}")

        message("Final CUDA flags: ${CMAKE_CUDA_FLAGS}")

        find_library(CUDA_DRIVER_LIBRARY
                NAMES cuda
                PATHS ${CUDA_TOOLKIT_ROOT_DIR}
                PATH_SUFFIXES lib64 lib/x64)

        add_compile_definitions(SD_CUDA=true)
        if("$ENV{CLION_IDE}")
            add_compile_definitions(__CUDA_CC__)
        endif()

        message("Set CUDA C++ Standard to ${CMAKE_CUDA_STANDARD}")

        set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")

        file(GLOB_RECURSE PERF_SOURCES ../include/performance/*.cpp ../include/performance/*.h)
        file(GLOB_RECURSE EXCEPTIONS_SOURCES ../include/exceptions/*.cpp ../include/exceptions/*.h)
        file(GLOB_RECURSE EXEC_SOURCES ../include/execution/impl/*.cpp ../include/execution/cuda/*.cu ../include/execution/cuda/*.h ../include/execution/*.cu ../include/execution/*.h)
        file(GLOB_RECURSE TYPES_SOURCES ../include/types/*.cpp ../include/types/*.h)
        file(GLOB_RECURSE ARRAY_SOURCES ../include/array/cuda/*.cu ../include/array/cuda/*.chpp ../include/array/impl/*.cpp ../include/array/cuda/*.cu ../include/array/*.h)
        file(GLOB_RECURSE MEMORY_SOURCES ../include/memory/impl/*.cpp ../include/memory/cuda/*.cu ../include/memory/*.h)
        file(GLOB_RECURSE GRAPH_SOURCES ../include/graph/*.cpp ../include/graph/*.cu ../include/graph/*.h)
        file(GLOB_RECURSE CUSTOMOPS_SOURCES ../include/ops/declarable/generic/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ../include/ops/declarable/helpers/cuda/*.cu ../include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE OPS_SOURCES ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp ../include/ops/*.h)
        file(GLOB_RECURSE HELPERS_SOURCES
                ../include/build_info.cpp
                ../include/ConstMessages.cpp
                ../include/helpers/*.cpp
                ../include/helpers/cuda/*.cu
                ../include/helpers/*.h)
        file(GLOB CPU_HELPERS_TO_EXCLUDE
                ../include/helpers/cpu/*.cpp)
        list(REMOVE_ITEM HELPERS_SOURCES ${CPU_HELPERS_TO_EXCLUDE})
        file(GLOB_RECURSE INDEXING_SOURCES ../include/indexing/*.cpp ../include/indexing/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES ../include/loops/impl/*.cpp ../include/loops/*.h)
        file(GLOB_RECURSE LEGACY_SOURCES ../include/legacy/impl/*.cpp ../include/legacy/*.cu ../include/legacy/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA  ../include/loops/*.cu ../include/loops/cuda/**/*.cu)
        file(GLOB_RECURSE COMPILATION_UNITS ../include/loops/cuda/compilation_units/*.cu.in ../include/ops/impl/compilation_units/*.cpp.in)
        file(GLOB_RECURSE COMPILATION_UNITS ../include/loops/cuda/compilation_units/*.cu.in  ../include/ops/impl/compilation_units/*.cpp.in)

        foreach(FL_ITEM ${COMPILATION_UNITS})
            genCompilation(${FL_ITEM})
        endforeach()
        set(ALL_SOURCES
                ${CUDA_GENERATED_SOURCES}
                ${PERF_SOURCES}
                ${EXCEPTIONS_SOURCES}
                ${EXEC_SOURCES}
                ${TYPES_SOURCES}
                ${ARRAY_SOURCES}
                ${MEMORY_SOURCES}
                ${GRAPH_SOURCES}
                ${CUSTOMOPS_SOURCES}
                ${CUSTOMOPS_ONEDNN_SOURCES}
                ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
                ${CUSTOMOPS_GENERIC_SOURCES}
                ${OPS_SOURCES}
                ${HELPERS_SOURCES}
                ${INDEXING_SOURCES}
                ${LOOPS_SOURCES}
                ${LEGACY_SOURCES}
                ${LOOPS_SOURCES_CUDA}
        )

        if(HAVE_CUDNN)
            message("cuDNN included")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ../include/ops/declarable/platform/cudnn/*.cu)
        endif()

        if(SD_GCC_FUNCTRACE STREQUAL "ON")
            add_library(samediff_obj OBJECT
                    ${CUDA_GENERATED_SOURCES}
                    ${LEGACY_SOURCES}
                    ${HELPERS_CPP}
                    ${CUSTOMOPS_HELPERS_SOURCES}
                    ${HELPERS_SOURCES}
                    ${EXEC_SOURCES}
                    ${LOOPS_SOURCES}
                    ${ARRAY_SOURCES}
                    ${TYPES_SOURCES}
                    ${MEMORY_SOURCES}
                    ${GRAPH_SOURCES}
                    ${CUSTOMOPS_SOURCES}
                    ${INDEXING_SOURCES}
                    ${EXCEPTIONS_SOURCES}
                    ${OPS_SOURCES}
                    ${PERF_SOURCES}
                    ${CUSTOMOPS_CUDNN_SOURCES}
                    ${CUSTOMOPS_ONEDNN_SOURCES}
                    ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
                    ${CUSTOMOPS_GENERIC_SOURCES}
                    ${LOOPS_SOURCES_CUDA}
            )
        else()
            add_library(samediff_obj OBJECT
                    ${CUDA_GENERATED_SOURCES}
                    ${LOOPS_SOURCES_CUDA}
                    ${LEGACY_SOURCES}
                    ${HELPERS_CPP}
                    ${CUSTOMOPS_HELPERS_SOURCES}
                    ${HELPERS_SOURCES}
                    ${EXEC_SOURCES}
                    ${LOOPS_SOURCES}
                    ${ARRAY_SOURCES}
                    ${TYPES_SOURCES}
                    ${MEMORY_SOURCES}
                    ${GRAPH_SOURCES}
                    ${CUSTOMOPS_SOURCES}
                    ${INDEXING_SOURCES}
                    ${EXCEPTIONS_SOURCES}
                    ${OPS_SOURCES}
                    ${PERF_SOURCES}
                    ${CUSTOMOPS_CUDNN_SOURCES}
                    ${CUSTOMOPS_ONEDNN_SOURCES}
                    ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
                    ${CUSTOMOPS_GENERIC_SOURCES}
                    ${LOOPS_SOURCES_CUDA}
            )
        endif()

        target_include_directories(samediff_obj PUBLIC ${EXTERNAL_INCLUDE_DIRS})

        set(MSVC_RT_LIB "MultiThreadedDLL")
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        if(WIN32)
            message("Enabling /EHsc for CUDA on Windows")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /std:c++14")
        endif()

        if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        endif()

        if(SD_STATIC_LIB AND SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}static DESTINATION .)
        elseif(SD_STATIC_LIB)
            add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
        endif()

        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                ${CUDA_LIBRARIES}
                ${CUDA_DRIVER_LIBRARY}
                ${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${CUDNN}
        )
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cuda")

        install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
    endif()
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
endif()





# The following existing code section will now use the optimized COMBINATIONS_3:
foreach(TEMPLATE_FILE ${CUDA_TEMPLATE_FILES})
    foreach(COMBINATION ${COMBINATIONS_3})
        genSingleFunctionCuda(${TEMPLATE_FILE} ${COMBINATION} "${CMAKE_BINARY_DIR}/cuda_generated")
    endforeach()
endforeach()




function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
    # Split the COMBINATION string into a list
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    # Extract combination values
    list(GET COMB_LIST 0 COMB1)
    list(GET COMB_LIST 1 COMB2)
    list(GET COMB_LIST 2 COMB3)

    # Get the base name from the template file
    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)

    # Read the template content
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Extract class and method names using regex
    string(REGEX MATCH "([a-zA-Z0-9_:]+),[ \n\t]*::([a-zA-Z0-9_]+)" FUNCTION_MATCH "${TEMPLATE_CONTENT}")
    set(CLASS_NAME ${CMAKE_MATCH_1})
    set(METHOD_NAME ${CMAKE_MATCH_2})

    # Clean class name for file naming
    string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")

    # Extract function signature to create a hash
    string(REGEX MATCH "::${METHOD_NAME}\\(([^;]+)\\);" FUNC_ARGS_MATCH "${TEMPLATE_CONTENT}")
    set(FUNCTION_ARGS "${CMAKE_MATCH_1}")

    # Create a signature identifier based on parameter count and types
    set(PARAM_COUNT 0)
    set(SIGNATURE_ID "")

    # Split the function arguments and count them
    string(REPLACE "," ";" ARGS_LIST "${FUNCTION_ARGS}")
    list(LENGTH ARGS_LIST PARAM_COUNT)

    # Create a hash of the signature
    foreach(ARG ${ARGS_LIST})
        # Extract just the type name from the parameter
        string(REGEX MATCH "^[^*& \t]+" TYPE_NAME "${ARG}")
        if(TYPE_NAME)
            # Append to the signature ID
            string(APPEND SIGNATURE_ID "_${TYPE_NAME}")
        endif()
    endforeach()

    # Create a shorter hash if the signature is too long
    if(SIGNATURE_ID MATCHES ".{30,}")
        string(MD5 SIGNATURE_HASH "${SIGNATURE_ID}")
        string(SUBSTRING "${SIGNATURE_HASH}" 0 8 SIGNATURE_ID)
        set(SIGNATURE_ID "_h${SIGNATURE_ID}")
    endif()

    # Output filename with signature identifier
    set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}${SIGNATURE_ID}_${COMB1}_${COMB2}_${COMB3}.cu")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    # Check if this file already exists - if so, no need to regenerate
    if(EXISTS "${GENERATED_FILE}")
        # Add to CUDA_GENERATED_SOURCES without regenerating
        list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
        set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
        return()
    endif()

    # Extract just the necessary header information
    set(START_MARKER "ITERATE_COMBINATIONS_3")
    string(FIND "${TEMPLATE_CONTENT}" "${START_MARKER}" START_POS)
    if(START_POS EQUAL -1)
        message(FATAL_ERROR "Could not find ITERATE_COMBINATIONS_3 in template file ${TEMPLATE_FILE}")
    endif()

    string(SUBSTRING "${TEMPLATE_CONTENT}" 0 ${START_POS} HEADER_CONTENT)

    # Create streamlined content with just the single instantiation needed
    set(NEW_CONTENT "${HEADER_CONTENT}\n\n// Single function instantiation for ${CLASS_NAME}::${METHOD_NAME}\n")
    string(APPEND NEW_CONTENT "template void ${CLASS_NAME}::${METHOD_NAME}<SD_SINGLE_TYPE_${COMB1}, SD_SINGLE_TYPE_${COMB2}, SD_SINGLE_TYPE_${COMB3}>(${FUNCTION_ARGS});\n")

    # Create directory if needed
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # Write the processed content to the output file
    file(WRITE "${GENERATED_FILE}" "${NEW_CONTENT}")

    # Set properties and add to sources
    set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA)
    list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)

    message(STATUS "Generated: ${GENERATED_FILE}")
endfunction()

# Define genPartitionCombination Function
function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
    # TEMPLATE_FILE: Path to the .cpp.in template file
    # COMBINATION_TYPE: "3" for ITERATE_COMBINATIONS_3, "2" for ITERATE_COMBINATIONS
    # COMBINATION: Comma-separated string of indices, e.g., "0,0,0" or "0,1"
    # OUTPUT_DIR: Directory to place the generated .cpp file

    # Split the COMBINATION string into a list
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    # Determine the number of elements in the combination
    list(LENGTH COMB_LIST COMB_COUNT)

    # Validate combination length
    if(NOT (COMBINATION_TYPE EQUAL 3 OR COMBINATION_TYPE EQUAL 2))
        message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 3 or 2.")
    endif()

    if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR
    (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
        message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).")
    endif()

    # Define placeholders based on combination type
    if(COMBINATION_TYPE EQUAL 3)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        list(GET COMB_LIST 2 COMB3)
        set(PLACEHOLDER1 "@COMB1@")
        set(PLACEHOLDER2 "@COMB2@")
        set(PLACEHOLDER3 "@COMB3@")
    elseif(COMBINATION_TYPE EQUAL 2)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        set(PLACEHOLDER1 "@COMB1@")
        set(PLACEHOLDER2 "@COMB2@")
    endif()

    # Read the template content
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Perform placeholder replacements
    if(COMBINATION_TYPE EQUAL 3)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" TEMP_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" TEMP_CONTENT "${TEMP_CONTENT}")
        string(REPLACE "${PLACEHOLDER3}" "${COMB3}" FINAL_CONTENT "${TEMP_CONTENT}")
    elseif(COMBINATION_TYPE EQUAL 2)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" FINAL_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}")
    endif()

    # Define the output file name based on combination
    if(COMBINATION_TYPE EQUAL 3)
        set(OUTPUT_FILE "pairwise_instantiation_${COMB1}_${COMB2}_${COMB3}.cpp")
    elseif(COMBINATION_TYPE EQUAL 2)
        set(OUTPUT_FILE "pairwise_instantiation_${COMB1}_${COMB2}.cpp")
    endif()

    # Define the full path for the generated file
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    # Write the processed content to the output file
    file(WRITE "${GENERATED_FILE}" "${FINAL_CONTENT}")

    # Optionally, collect generated sources for further processing
    list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")

    # Output a message for verification
    message(STATUS "Generated Instantiation File: ${GENERATED_FILE}")
endfunction()


set(GEN_COMPILATION_TEMPLATES
        "${CMAKE_CURRENT_SOURCE_DIR}/loops/cpu/compilation_units/pairwise_instantiation_template.cpp.in"
        # Add other template files as needed
)
# After (modified to add our template):
set(INSTANTIATION_TEMPLATES_2
        "${CMAKE_CURRENT_SOURCE_DIR}/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in"
)



# Example Usage of genPartitionCombination
# Define combinations for ITERATE_COMBINATIONS_3
set(COMBINATIONS_3
        "0,0,0"
        "0,0,2"
        "0,1,2"
        "0,2,0"
        "0,2,1"
        "0,2,2"
        "1,0,0"
        "1,1,1"
        "1,1,2"
        "1,2,0"
        "1,2,1"
        "1,2,2"
        "2,0,0"
        "2,0,1"
        "2,0,2"
        "2,1,0"
        "2,1,1"
        "2,1,2"
        "2,2,0"
        "2,2,1"
        "2,2,2"
)

# Define combinations for ITERATE_COMBINATIONS
set(COMBINATIONS_2
        "0,0"
        "0,1"
        "1,0"
        "1,1"
        "0,2"
        "2,0"
        "1,2"
        "2,1"
        "2,2"
)

# Define the list of instantiation templates under comb_compilation_units
set(INSTANTIATION_TEMPLATES_3
        "${CMAKE_CURRENT_SOURCE_DIR}/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"
)

set(INSTANTIATION_TEMPLATES_2
        "${CMAKE_CURRENT_SOURCE_DIR}/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in"
)


#///////////////////////////////////////////////////////////////////////////////
# genCompilation: Generates cpp, cu files
# INPUT:
# $FILE_ITEM template-configuration that utilizes libnd4j type, macros helpers
# defined inside { include/types/types.h, include/system/type_boilerplate.h}
# OUTPUT:
# $CUSTOMOPS_GENERIC_SOURCES  generated files will be added into this List
#////////////////////////////////////////////////////////////////////////////////
#  A simple template-configuration file example:
# // hints and defines what types will be generated
# #cmakedefine SD_COMMON_TYPES_GEN
# #cmakedefine SD_FLOAT_TYPES_GEN
# // below if defines blocks are needed for correctly handling multiple types
# #if defined(SD_COMMON_TYPES_GEN) && defined(SD_COMMON_TYPES_@FL_TYPE_INDEX@)
#  BUILD_DOUBLE_TEMPLATE(template void someFunc, (arg_list,..),
#                          SD_COMMON_TYPES_@FL_TYPE_INDEX@, SD_INDEXING_TYPES);
# #endif
# #if defined(SD_FLOAT_TYPES_GEN) && defined(SD_FLOAT_TYPES_@FL_TYPE_INDEX@)
#  BUILD_SINGLE_TEMPLATE(template class SomeClass,, SD_FLOAT_TYPES_@FL_TYPE_INDEX@);
# #endif
#////////////////////////////////////////////////////////////////////////////////

set_property(GLOBAL PROPERTY JOB_POOLS one_jobs=1 two_jobs=2)

# removeFileIfExcluded: removes the file from the given glob list
# used for filtering files that shouldn't be included in compilation
function(removeFileIfExcluded)
    cmake_parse_arguments(
            PARSED_ARGS # prefix of output variables
            ""          # list of names of the boolean arguments (only defined ones will be true)
            "FILE_ITEM" # list of names of mono-valued arguments
            "LIST_ITEM" # list of names of multi-valued arguments (expects list *name*)
            ${ARGN}     # arguments of the function to parse, here we take the all original ones
    )
    file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
    string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)

    # Check if the "NOT_EXCLUDED" marker is present
    if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
        # Get the list content into a local variable for safe modification
        set(local_list ${${PARSED_ARGS_LIST_ITEM}})
        set(file_removed FALSE) # Flag to track if the file was removed

        foreach(OP ${SD_OPS_LIST})
            # *** FIX 1: Use correct variable FILE_CONTENTS ***
            string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)

            # If the specific OP exclusion is NOT found, it means the file *should* be removed (excluded)
            if(${NOT_EXCLUDED_OP_IDX} LESS 0)
                list(REMOVE_ITEM local_list "${PARSED_ARGS_FILE_ITEM}")
                set(file_removed TRUE)
                # Once removed for one missing OP exclusion, no need to check further for this file
                break()
            endif()
        endforeach()

        # *** FIX 2: Propagate the modified list back to the parent scope IF it was changed ***
        if(file_removed)
            set(${PARSED_ARGS_LIST_ITEM} ${local_list} PARENT_SCOPE)
        endif()
    endif()
    # If "NOT_EXCLUDED" is not found at all, the file is included by default, do nothing.

endfunction()

function(genCompilation FILE_ITEM)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)

    set(EXTENSION "cpp")

    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)
    #check content for types

    #set all to false
    set (SD_FLOAT_TYPES_GEN     0)
    set (SD_INTEGER_TYPES_GEN       0)
    set (SD_COMMON_TYPES_GEN   0)
    set (SD_PAIRWISE_TYPES_GEN  0)
    set (RANGE_STOP         -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    #STOP=COUNT(SD_COMMON_TYPES)-1
    set(SD_INTEGER_TYPES_END 7)
    set(SD_COMMON_TYPES_END 12)
    set(SD_FLOAT_TYPES_END 3)
    set(SD_PAIRWISE_TYPES_END 12)

    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set (SD_INTEGER_TYPES_GEN  1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set (SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set (SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set (SD_PAIRWISE_TYPES_GEN  1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()

    endforeach()

    if(RANGE_STOP GREATER -1)
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            # set OFF if the index is above
            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set (SD_FLOAT_TYPES_GEN     0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set (SD_INTEGER_TYPES_GEN     0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set (SD_COMMON_TYPES_GEN   0)
            endif()
            set(GENERATED_SOURCE  "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            configure_file(  "${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
            LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE} )
        endforeach()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()


# Define a function to generate individual CUDA files for each type combination
function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
    # Split the COMBINATION string into a list
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    # Extract combination values
    list(GET COMB_LIST 0 COMB1)
    list(GET COMB_LIST 1 COMB2)
    list(GET COMB_LIST 2 COMB3)

    # Get the base name from the template file
    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)

    # Read the template content
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Extract class and method names using regex
    string(REGEX MATCH "([a-zA-Z0-9_:]+),[ \n\t]*::([a-zA-Z0-9_]+)" FUNCTION_MATCH "${TEMPLATE_CONTENT}")
    set(CLASS_NAME ${CMAKE_MATCH_1})
    set(METHOD_NAME ${CMAKE_MATCH_2})

    # Clean class name for file naming
    string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")

    # Output filename
    set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}_${COMB1}_${COMB2}_${COMB3}.cu")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    # Replace placeholders with actual values
    string(REPLACE "@COMB1@" "${COMB1}" MODIFIED_CONTENT "${TEMPLATE_CONTENT}")
    string(REPLACE "@COMB2@" "${COMB2}" MODIFIED_CONTENT "${MODIFIED_CONTENT}")
    string(REPLACE "@COMB3@" "${COMB3}" MODIFIED_CONTENT "${MODIFIED_CONTENT}")

    # Create directory if needed
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # Write the content
    file(WRITE "${GENERATED_FILE}" "${MODIFIED_CONTENT}")

    # Set properties and add to sources
    set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA)
    list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)

    message(STATUS "Generated: ${GENERATED_FILE}")
endfunction()

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

set(SD_X86_BUILD false)
# Determine platform type more accurately
set(SD_X86_BUILD false)
set(SD_ARM_BUILD false) # Add similar flag for ARM if needed

if(SD_ANDROID_BUILD)
    # For Android, trust ANDROID_ABI
    if(ANDROID_ABI MATCHES "x86_64")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86-64") # Explicitly set SD_ARCH too
    elseif(ANDROID_ABI MATCHES "x86")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86") # Explicitly set SD_ARCH too
    elseif(ANDROID_ABI MATCHES "arm64-v8a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "arm64-v8a") # Explicitly set SD_ARCH too
    elseif(ANDROID_ABI MATCHES "armeabi-v7a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "armv7-a") # Explicitly set SD_ARCH too
    endif()
elseif(NOT SD_IOS_BUILD)
    # Non-Android, Non-iOS logic (can use CMAKE_SYSTEM_PROCESSOR or keep original checks)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_X86_BUILD true)
        # Potentially set SD_ARCH here too if not passed externally
        if(NOT DEFINED SD_ARCH)
            set(SD_ARCH "x86-64")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
        set(SD_ARM_BUILD true)
        # Potentially set SD_ARCH here too if not passed externally
        # Add other non-Android/iOS platforms (PowerPC, etc.) if needed
    endif()
    # Add iOS logic if necessary
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")


# -fsanitize=address
# -fsanitize=leak
if (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "armv8-a")
        set(SD_ARCH armv8-a)
        set(SD_X86_BUILD false)
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()


    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
    set(SD_X86_BUILD true)
    if (SD_CUDA)
        set(CMAKE_CXX_FLAGS_RELEASE  "-D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  "  /FS /EHsc")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
    endif()
else()
    # set this as a separate flag. profiler won't work with anything higher than -O0
    # we also have to do this for release because libnd4j has issues when working with java
    # ideally debug builds would work but since it's not consistent it's better to just
    # set this for release flags so we can debug code even from the java level
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        # note we may or may not use a build type called "none" to prevent default injection
        # of flags from cmake. We do this when using functrace so we can add symbols
        # to a binary but still run from java without freezing.
        # Normally, we would just want to use debug build. Running a debug build
        # via JNI seems to just freeze though. The goal is to just use tools like
        # valgrind or compute-sanitizer or even address sanitizer with symbols
        # embedded in a binary but still run code from java.
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
    # note on ftls model: https://github.com/microsoft/mimalloc/issues/147 tsanitize sometimes throws errors
    if (SD_SANITIZE)
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIE   -lpthread -ftls-model=local-dynamic  -static-libasan  -fsanitize=${SD_SANITIZERS}  -fno-sanitize-recover=all")
        message("Using sanitizers: ${SD_SANITIZERS} - note you can not use both thread and address sanitizer at the same time. Be careful what sanitizers you specify.
         Note that address and undefined can not be used at the same time or an address overlap error will occur.  See: https://github.com/google/sanitizers/issues/856
         FOR THREADS USE: thread,undefined,float-divide-by-zero,float-cast-overflow
         FOR ADDRESS USE: address,undefined,float-divide-by-zero,float-cast-overflow")
        if(SD_CPU)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${SANITIZE_FLAGS}")
        endif()
        if(SD_CUDA)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${SANITIZE_FLAGS} -lpthread -ftls-model=local-dynamic --relocatable-device-code=true")
        endif()
        # adds stack size to prevent misc errors with address sanitizer
    endif()
endif()

if(SD_NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*")
        set(SD_X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()


if(NOT SD_CUDA)
    # we need this definition to avoid global memory use within onednn
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if ("${OPENBLAS_PATH}" STREQUAL "")
        #we don't want OpenBLAS on Apple
        if (NOT APPLE)
            # note: this is not a typo
            set(BLA_VENDOR "OpenBLAS")
        endif()

        # look around for system blas instead, see: https://cmake.org/cmake/help/latest/module/FindBLAS.html
        find_package(BLAS REQUIRED)
        if (BLAS_FOUND)
            message("Found external BLAS implementation: ${BLAS_LIBRARIES} ")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        endif()
    else()
        # if we have externally provided OPENBLAS_PATH - let's use it
        set(HAVE_OPENBLAS 1)
        message("Setting openblas")
        include_directories(${OPENBLAS_PATH}/include/)
        link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
        set(OPENBLAS_LIBRARIES openblas)
    endif()

endif()


#arm-compute entry
# Root CMakeLists.txt adjustment

# Initialize variables to indicate ARM Compute is not yet configured
set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)

# Only attempt ARM Compute configuration if the helper flag is set
if(${HELPERS_armcompute})
    message(STATUS "ARM Compute helper is active, attempting configuration...")

    # First, try to find an existing installation
    find_package(ARMCOMPUTE QUIET) # Use QUIET to avoid messages if not found initially

    if(ARMCOMPUTE_FOUND)
        # Found via find_package
        set(HAVE_ARMCOMPUTE 1)
        include_directories(${ARMCOMPUTE_INCLUDE_DIRS}) # Use standard variable from FindARMCOMPUTE
        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIBRARIES}) # Standard variable from FindARMCOMPUTE
        message(STATUS "Found ARM Compute via find_package. Include: ${ARMCOMPUTE_INCLUDE_DIRS}, Libs: ${ARMCOMPUTE_LIBRARIES}")
    else()
        # Not found via find_package, attempt download if URL is defined and arch matches
        message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
        if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7" OR ${SD_ARCH} MATCHES "armv8-a" OR ${SD_ARCH} MATCHES "arm64"))
            message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")

            # Configure the download project
            configure_file(./CMakeLists.txt.armcompute.in armcompute-download/CMakeLists.txt @ONLY)

            # Execute the CMake configure step for the download
            execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result_config
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
                    OUTPUT_QUIET ERROR_QUIET)

            if(NOT result_config EQUAL 0)
                message(WARNING "CMake configuration step for ARM Compute Library download failed: ${result_config}")
            else()
                # Execute the CMake build step for the download
                execute_process(COMMAND ${CMAKE_COMMAND} --build .
                        RESULT_VARIABLE result_build
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
                        OUTPUT_QUIET ERROR_QUIET)

                if(NOT result_build EQUAL 0)
                    message(WARNING "Build step for ARM Compute Library download failed: ${result_build}")
                else()
                    # Download and build successful, set up paths and variables
                    set(ARMCOMPUTE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/armcompute-src") # Set the root directory

                    # Check if the expected library files exist after download/build
                    set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so")
                    set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so")

                    if(EXISTS "${ARMCOMPUTE_LIB_PATH}" AND EXISTS "${ARMCOMPUTE_CORE_LIB_PATH}")
                        add_definitions(-DARMCOMPUTENEON_ENABLED) # Add necessary definition
                        include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include) # Include directories
                        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIB_PATH} ${ARMCOMPUTE_CORE_LIB_PATH}) # Set the libraries variable
                        set(HAVE_ARMCOMPUTE 1) # Mark ARM Compute as available
                        message(STATUS "ARM Compute Library successfully configured via download at: ${ARMCOMPUTE_ROOT}")
                    else()
                        message(WARNING "ARM Compute download/build step completed, but library files not found at expected location: ${ARMCOMPUTE_ROOT}/lib")
                    endif()
                endif()
            endif()
        else()
            message(STATUS "ARM Compute helper active, but library not found and download conditions not met (URL: ${ARM_COMPUTE_URL}, Arch: ${SD_ARCH}).")
        endif()
    endif()
endif()

# --- The rest of your root CMakeLists.txt continues below ---
# Make sure the 'else' block that previously defined ARMCOMPUTE_LIBRARIES incorrectly is removed.


if (${HELPERS_onednn})
    message("Going to pull & build onednn")
    set(HAVE_ONEDNN 1)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE)

    configure_file(./CMakeLists.txt.onednn.in onednn-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND}  "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
    if(result)
        message(FATAL_ERROR "CMake step for onednn failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
    if(result)
        message(FATAL_ERROR "Build step for onednn failed: ${result}")
    endif()

    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/onednn-src
            ${CMAKE_CURRENT_BINARY_DIR}/onednn-build
            EXCLUDE_FROM_ALL)

    set(onednn_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/onednn-build)
    set(onednn_EXT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onednn-src)
    set(ONEDNN_PATH "${onednn_SOURCE_DIR}")
    include_directories(${onednn_SOURCE_DIR}/include ${onednn_EXT_DIR}/include ${onednn_SOURCE_DIR})
    set(ONEDNN dnnl)
endif()


if (${HELPERS_cudnn})
    if (NOT SD_CUDA)
        message(FATAL_ERROR "Can't build cuDNN on non-CUDA platform")
    endif()

    #set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

    SET(CUDNN_LIBNAME "cudnn")

    if(DEFINED ENV{CUDNN_ROOT_DIR})
        message("Using cudnn root directory from environment")
        set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
    endif()
    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        message("Using cuda root directory from environment")
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
    endif()
    message("Cudnn root dir ${CUDNN_ROOT_DIR} CUDA TOOLKIT ROOT DIR ${CUDA_TOOLKIT_ROOT_DIR}")
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES cuda/include include)

    find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 cuda/lib/x64 lib/x64)


    if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message("CUDNN include dir is ${CUDNN_INCLUDE_DIR}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true)
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find cuDNN, no library found. Please ensure cudnn is installed under the cuda toolkit directory.")
    endif()
endif()

# Before the flatbuffers download section
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    set(FLATBUFFERS_BUILD_FLATC "ON" CACHE STRING "Enable flatc build" FORCE)
else()
    set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Disable flatc build" FORCE)
endif()

# Existing flatbuffers download section
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker")
    configure_file(CMakeLists.txt.in flatbuffers-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download)
    if(result)
        message(FATAL_ERROR "CMake step for flatbuffers failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download)
    if(result)
        message(FATAL_ERROR "Build step for flatbuffers failed: ${result}")
    endif()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker" "Download complete")
else()
    message(STATUS "Flatbuffers already downloaded, skipping download step")
endif()

# Add flatbuffers directly to build
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src
        ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
        EXCLUDE_FROM_ALL)

# After flatbuffers is added, handle generation if enabled
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    # First, ensure flatc is built
    execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build --target flatc
            RESULT_VARIABLE FLATC_BUILD_RESULT
    )
    if(FLATC_BUILD_RESULT)
        message(FATAL_ERROR "Failed to build flatc: ${FLATC_BUILD_RESULT}")
    endif()


    set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")
    message("Using flatc from: ${FLATC_EXECUTABLE}")

    # Create required directories
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated
            ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd
    )

    # Run flatc generation
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
            bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
            RESULT_VARIABLE FLATC_RESULT
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(FLATC_RESULT)
        message(FATAL_ERROR "Flatbuffer generation failed: ${FLATC_RESULT}")
    endif()

    # Copy Java files
    execute_process(
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
            RESULT_VARIABLE COPY_RESULT
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(COPY_RESULT)
        message(FATAL_ERROR "Java file copying failed: ${COPY_RESULT}")
    endif()
endif()
include_directories(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include)

configure_file(include/config.h.in include/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(blas)
if(SD_BUILD_TESTS)
    include(CTest)
    # tests are always compiled with all ops included
    set(SD_ALL_OPS true)
    enable_testing()
    add_subdirectory(tests_cpu)
endif()


set (CMAKE_INSTALL_PREFIX $ENV{ND4J_HOME}/nd4j-native-parent/nd4j-native/src/main/resources)

# Set package information
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "agibsonccc <adam@skymind.global>")
set(CPACK_PACKAGE_VENDOR "Eclipse")
set(CPACK_SETDESTDIR "false")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/lib")
set(CPACK_PACKAGE_NAME "libnd4j")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "8")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "libnd4j")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Determine distribution and release — may require redhat-lsb-core installed on CentOS / RH
execute_process(COMMAND lsb_release -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND lsb_release -sc OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND uname -i OUTPUT_VARIABLE ARCHITECTURE)

# Set package name and type (deb vs rpm)
if(DISTRIBUTION STREQUAL "Ubuntu")

    # Set Ubuntu-specific information (see http://www.cmake.org/Wiki/CMake:CPackPackageGenerators)
    if(ARCHITECTURE MATCHES ".*x86_64.*")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    else()
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
    endif()
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "agibsonccc")
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "cuda")
    # For Ubuntu <= 12, libatlas3gf-base, liblapack3gf
    # Build deps: libatlas3-base liblapack3 libopenblas-dev libatlas-dev liblapack-dev gcc-5 g++-5
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/deeplearning4j/deeplearning4j")
    set(CPACK_GENERATOR "DEB")
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${RELEASE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm;" )

elseif(DISTRIBUTION STREQUAL "CentOS")

    # Set Fedora-specific information (see http://www.cmake.org/Wiki/CMake:CPackPackageGenerators)
    execute_process(COMMAND lsb_release -sr OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ARCHITECTURE MATCHES ".*x86_64.*")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    else()
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "i686")
    endif()
    set(CPACK_PACKAGE_CONTACT "agibsonccc")
    set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
    set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    set(CPACK_RPM_PACKAGE_SUGGESTS "cuda")
    set(CPACK_RPM_PACKAGE_REQUIRES "")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/deeplearning4j/deeplearning4j/libnd4j")
    set(CPACK_GENERATOR "RPM")
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.fc${RELEASE}.${CPACK_RPM_PACKAGE_ARCHITECTURE})
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst")
    set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm")
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib")

endif()

include(CPack)