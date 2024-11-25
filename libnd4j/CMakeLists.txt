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



if (SD_CUDA)
    message("Enabling cuda")

    if("${SD_ARCH}" MATCHES "armv8-a" AND UNIX)
        message("Adding jetson nano specific settings")
        # Need to manually specify stubbed links in order
        # for cublas and cusolver to be resolved
        if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CUDA_cublas_LIBRARY}" MATCHES ".*NOTFOUND.*" )
            message("Setting cublas library manually")
            set(CUDA_cublas_LIBRARY  "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so" CACHE STRING "CUDA CUBLAS LIB" FORCE)
            set(CUDA_cublas_LIBRARY  "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so")
            # See: https://stackoverflow.com/questions/47948473/what-this-cmake-error-means-my-variables-are-set-to-notfound
            unset(CUDA_cublas-NOTFOUND CACHE)
            unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
            unset(CUDA_cublas_LIBRARY-NOTFOUND PARENT_SCOPE)


        endif()

        if(NOT DEFINED CUDA_cusolver_LIBRARY OR "${CUDA_cusolver_LIBRARY}" MATCHES ".*NOTFOUND.*")
            message("Setting cusolver library manually")
            set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so" CACHE STRING "CUDA CUSOLVER LIB" FORCE)
            set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so")
            #See: https://stackoverflow.com/questions/47948473/what-this-cmake-error-means-my-variables-are-set-to-notfound
            unset(CUDA_cusolver-NOTFOUND CACHE)
            unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
            unset(CUDA_cusolver_LIBRARY-NOTFOUND PARENT_SCOPE)

        endif()

        message("Jetson nano cublas library is ${CUDA_cublas_LIBRARY} and CuSolver library ${CUDA_cusolver_LIBRARY}")
    endif()


    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   -allow-unsupported-compiler --ptxas-options=-v  -Xptxas -O1")
    if(SD_KEEP_NVCC_OUTPUT)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --keep ")
    endif()
    enable_language(CUDA)
    # see: https://stackoverflow.com/questions/2876357/determine-the-line-of-code-that-causes-a-segmentation-fault
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)
    # cuda driver library for device validator
    find_library(CUDA_DRIVER_LIBRARY
            NAMES cuda
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib64 lib/x64)

    # Add a platform agnostic way of showing we are compiling for cuda
    # The normal way of using __NVCC__ and __CUDACC__ does not show up
    # for cpp files which makes it difficult to add guards for specific functions.
    add_compile_definitions(SD_CUDA=true)
    if("$ENV{CLION_IDE}")
        add_compile_definitions(__CUDA_CC__)
    endif()
    if(CMAKE_CXX_COMPILER MATCHES "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.29" )
        message("Working around windows build failure with visual studio. Visual studio 16.10 introduced a compiler bug with compiling CUDA code with C++14. Set the cuda standard to 17 as a workaround.")
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CUDA_STANDARD 11)
    endif()

    message("Set CUDA C++ Standard to ${CMAKE_CUDA_STANDARD}")

    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")

endif()





# Define Combinations
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
            "" # list of names of the boolean arguments (only defined ones will be true)
            "FILE_ITEM" # list of names of mono-valued arguments
            "LIST_ITEM" # list of names of multi-valued arguments (output variables are lists)
            ${ARGN} # arguments of the function to parse, here we take the all original ones
    )
    file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
    string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)
    # found an exclusion, figure out if relevant
    if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
        foreach(OP ${SD_OPS_LIST})
            string(FIND PARSED_ARGS_FILE_CONTENTS "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
            # figure out if the exclusion contains the given op if not exclude the file
            if(${NOT_EXCLUDED_OP_IDX} LESS 0)
                list(REMOVE_ITEM "${PARSED_ARGS_LIST_ITEM}" "${PARSED_ARGS_FILE_ITEM}")
            endif()

        endforeach()
    endif()

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

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

set(SD_X86_BUILD false)

if (NOT SD_IOS_BUILD AND NOT SD_ANDROID_BUILD AND NOT ${SD_ARCH} MATCHES "power*" AND NOT ${SD_ARCH} MATCHES "arm*")
    set(SD_X86_BUILD true)
endif()

# -fsanitize=address
# -fsanitize=leak
if (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "armv8-a")
        set(SD_ARCH armv8-a)
        set(SD_X86_BUILD false)
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()


    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
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
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIE   -lpthread -ftls-model=local-dynamic -fsanitize=${SD_SANITIZERS}  -fno-sanitize-recover=all")
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
if(${HELPERS_armcompute})
    find_package(ARMCOMPUTE REQUIRED)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE ld_version)
    if ("${ld_version}" MATCHES "GNU gold")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold ")
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_link_options("-Wl,--long-plt")
        endif()
    endif()

    if(ARMCOMPUTE_FOUND)
        message("Found ARMCOMPUTE: ${ARMCOMPUTE_LIBRARIES}")
        set(HAVE_ARMCOMPUTE 1)
        # Add preprocessor definition for ARM Compute NEON
        add_definitions(-DARMCOMPUTENEON_ENABLED)
        include_directories(${ARMCOMPUTE_INCLUDE})
        message("----${ARMCOMPUTE_INCLUDE}---")
    endif()


endif()






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

# Download and unpack flatbuffers at configure time
configure_file(CMakeLists.txt.in flatbuffers-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND}  "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download )
if(result)
    message(FATAL_ERROR "CMake step for flatbuffers failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download )
if(result)
    message(FATAL_ERROR "Build step for flatbuffers failed: ${result}")
endif()

# Add flatbuffers directly to our build.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src
        ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
        EXCLUDE_FROM_ALL)

set(HAVE_FLATBUFFERS 1)
set(FLATBUFFERS_PATH ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src)
include_directories(${FLATBUFFERS_PATH}/include)


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
