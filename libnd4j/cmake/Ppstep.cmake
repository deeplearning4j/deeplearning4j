# cmake/Ppstep.cmake
# Builds ppstep preprocessor tool with robust include path discovery

if(BUILD_PPSTEP)
    message(STATUS "🔧 PPSTEP BUILD MODE - Building ppstep tool only")
    message(STATUS "Setting up ppstep build in: ${CMAKE_BINARY_DIR}")
    
    # Check for Boost (required for ppstep)
    find_package(Boost COMPONENTS system filesystem program_options thread wave QUIET)
    
    if(NOT Boost_FOUND)
        message(FATAL_ERROR "❌ Boost not found - cannot build ppstep. Install with: apt-get install libboost-all-dev")
    endif()
    
    include(ExternalProject)
    
    # Configure ppstep as external project
    set(PPSTEP_PREFIX "${CMAKE_BINARY_DIR}/ppstep_external")
    set(PPSTEP_SOURCE_DIR "${PPSTEP_PREFIX}/src/ppstep")
    set(PPSTEP_BUILD_DIR "${PPSTEP_PREFIX}/build")
    
    # Clean up any previous failed attempts
    if(EXISTS "${PPSTEP_PREFIX}/src/ppstep_build-stamp")
        file(REMOVE_RECURSE "${PPSTEP_PREFIX}/src/ppstep_build-stamp")
    endif()
    
    message(STATUS "Cloning and building ppstep...")
    
    # Set up CMAKE_ARGS conditionally to avoid empty values
    set(PPSTEP_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
    )
    
    if(CMAKE_CXX_COMPILER)
        list(APPEND PPSTEP_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
    endif()
    
    if(CMAKE_C_COMPILER)
        list(APPEND PPSTEP_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
    endif()
    
    if(BOOST_ROOT)
        list(APPEND PPSTEP_CMAKE_ARGS -DBOOST_ROOT=${BOOST_ROOT})
    endif()
    
    ExternalProject_Add(ppstep_build
        PREFIX            "${PPSTEP_PREFIX}"
        GIT_REPOSITORY    "https://github.com/agibsonccc/ppstep.git"
        GIT_TAG           "master"
        GIT_SHALLOW       TRUE
        SOURCE_DIR        "${PPSTEP_SOURCE_DIR}"
        BINARY_DIR        "${PPSTEP_BUILD_DIR}"
        STAMP_DIR         "${PPSTEP_PREFIX}/stamp"
        TMP_DIR           "${PPSTEP_PREFIX}/tmp"
        DOWNLOAD_DIR      "${PPSTEP_PREFIX}/download"
        CMAKE_ARGS        ${PPSTEP_CMAKE_ARGS}
        BUILD_COMMAND     make
        INSTALL_COMMAND   ""
        BUILD_BYPRODUCTS  "${PPSTEP_BUILD_DIR}/ppstep"
        LOG_DOWNLOAD      TRUE
        LOG_CONFIGURE     TRUE
        LOG_BUILD         TRUE
    )
    
    # ============================================================================
    # ROBUST INCLUDE PATH DISCOVERY AT CMAKE CONFIGURE TIME
    # ============================================================================
    
    message(STATUS "Discovering system include paths...")
    
    set(DISCOVERED_INCLUDE_PATHS "")
    
    # Method 1: Use CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES
    foreach(dir ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        if(EXISTS "${dir}")
            list(APPEND DISCOVERED_INCLUDE_PATHS "${dir}")
        endif()
    endforeach()
    
    # Method 2: Get from compiler using execute_process
    if(CMAKE_CXX_COMPILER)
        # Get verbose output from compiler
        execute_process(
            COMMAND echo ""
            COMMAND ${CMAKE_CXX_COMPILER} -E -x c++ -Wp,-v -
            OUTPUT_VARIABLE COMPILER_OUTPUT
            ERROR_VARIABLE COMPILER_ERROR
            INPUT_FILE /dev/null
            RESULT_VARIABLE COMPILER_RESULT
        )
        
        # Parse the output to extract include paths
        string(REGEX MATCHALL "#include [<\"].*[>\"] search starts here:.*End of search list" INCLUDE_SECTION "${COMPILER_ERROR}")
        if(INCLUDE_SECTION)
            string(REGEX REPLACE "#include [<\"].*[>\"] search starts here:" "" INCLUDE_SECTION "${INCLUDE_SECTION}")
            string(REGEX REPLACE "End of search list" "" INCLUDE_SECTION "${INCLUDE_SECTION}")
            string(REGEX REPLACE "\n" ";" INCLUDE_LINES "${INCLUDE_SECTION}")
            
            foreach(line ${INCLUDE_LINES})
                string(STRIP "${line}" line)
                if(EXISTS "${line}")
                    list(APPEND DISCOVERED_INCLUDE_PATHS "${line}")
                endif()
            endforeach()
        endif()
        
        # Also try cpp directly
        find_program(CPP_EXECUTABLE cpp)
        if(CPP_EXECUTABLE)
            execute_process(
                COMMAND ${CPP_EXECUTABLE} -x c++ -v
                OUTPUT_VARIABLE CPP_OUTPUT
                ERROR_VARIABLE CPP_ERROR
                INPUT_FILE /dev/null
                RESULT_VARIABLE CPP_RESULT
            )
            
            string(REGEX MATCHALL "/[^\n]*include[^\n]*" INCLUDE_DIRS "${CPP_ERROR}")
            foreach(dir ${INCLUDE_DIRS})
                string(STRIP "${dir}" dir)
                if(EXISTS "${dir}" AND IS_DIRECTORY "${dir}")
                    list(APPEND DISCOVERED_INCLUDE_PATHS "${dir}")
                endif()
            endforeach()
        endif()
    endif()
    
    # Method 3: Get GCC-specific paths
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=include
        OUTPUT_VARIABLE GCC_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(EXISTS "${GCC_INCLUDE_DIR}")
        list(APPEND DISCOVERED_INCLUDE_PATHS "${GCC_INCLUDE_DIR}")
    endif()
    
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-search-dirs
        OUTPUT_VARIABLE GCC_SEARCH_DIRS
        ERROR_QUIET
    )
    if(GCC_SEARCH_DIRS)
        string(REGEX MATCH "install: ([^\n]*)" INSTALL_LINE "${GCC_SEARCH_DIRS}")
        if(CMAKE_MATCH_1)
            string(STRIP "${CMAKE_MATCH_1}" INSTALL_DIR)
            if(EXISTS "${INSTALL_DIR}/include")
                list(APPEND DISCOVERED_INCLUDE_PATHS "${INSTALL_DIR}/include")
            endif()
        endif()
    endif()
    
    # Method 4: Get target-specific paths
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
        OUTPUT_VARIABLE GCC_TARGET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(GCC_TARGET AND GCC_VERSION)
        # Extract major version
        string(REGEX MATCH "^([0-9]+)" GCC_MAJOR "${GCC_VERSION}")
        
        # Check common GCC installation paths
        foreach(base_path "/usr/lib/gcc" "/usr/lib64/gcc" "/usr/local/lib/gcc")
            foreach(version ${GCC_VERSION} ${GCC_MAJOR})
                foreach(subdir "include" "include-fixed")
                    set(test_path "${base_path}/${GCC_TARGET}/${version}/${subdir}")
                    if(EXISTS "${test_path}")
                        list(APPEND DISCOVERED_INCLUDE_PATHS "${test_path}")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endif()
    
    # Method 5: Common system paths
    set(COMMON_INCLUDE_PATHS
        /usr/include
        /usr/local/include
        /usr/include/linux
        /usr/include/x86_64-linux-gnu
        /usr/include/i386-linux-gnu
        /usr/include/aarch64-linux-gnu
        /opt/local/include
        /opt/include
    )
    
    foreach(path ${COMMON_INCLUDE_PATHS})
        if(EXISTS "${path}")
            list(APPEND DISCOVERED_INCLUDE_PATHS "${path}")
        endif()
    endforeach()
    
    # Add C++ standard library paths
    foreach(version RANGE 5 15)
        set(cpp_paths
            "/usr/include/c++/${version}"
            "/usr/include/c++/${version}/x86_64-linux-gnu"
            "/usr/include/c++/${version}/x86_64-redhat-linux"
            "/usr/include/c++/${version}/aarch64-linux-gnu"
            "/usr/include/c++/${version}/backward"
        )
        foreach(path ${cpp_paths})
            if(EXISTS "${path}")
                list(APPEND DISCOVERED_INCLUDE_PATHS "${path}")
            endif()
        endforeach()
    endforeach()
    
    # Method 6: Try to compile a test file and extract includes
    file(WRITE "${CMAKE_BINARY_DIR}/test_includes.cpp" "
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
int main() { return 0; }
")
    
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -H -E "${CMAKE_BINARY_DIR}/test_includes.cpp"
        OUTPUT_VARIABLE HEADER_OUTPUT
        ERROR_VARIABLE HEADER_ERROR
        RESULT_VARIABLE HEADER_RESULT
    )
    
    # Parse the -H output to find include directories
    string(REGEX MATCHALL "\\. /[^\n]*\\.h" HEADER_LINES "${HEADER_ERROR}")
    foreach(line ${HEADER_LINES})
        string(REGEX REPLACE "^\\. " "" header_file "${line}")
        get_filename_component(header_dir "${header_file}" DIRECTORY)
        if(EXISTS "${header_dir}")
            list(APPEND DISCOVERED_INCLUDE_PATHS "${header_dir}")
        endif()
    endforeach()
    
    # Method 7: Check if clang is available and get its paths
    find_program(CLANG_CXX clang++)
    if(CLANG_CXX)
        execute_process(
            COMMAND ${CLANG_CXX} -print-resource-dir
            OUTPUT_VARIABLE CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(EXISTS "${CLANG_RESOURCE_DIR}/include")
            list(APPEND DISCOVERED_INCLUDE_PATHS "${CLANG_RESOURCE_DIR}/include")
        endif()
    endif()
    
    # Remove duplicates
    list(REMOVE_DUPLICATES DISCOVERED_INCLUDE_PATHS)
    
    # Validate that we found critical headers
    set(CRITICAL_HEADERS limits.h stdio.h stdlib.h stddef.h stdint.h)
    set(MISSING_HEADERS "")
    
    foreach(header ${CRITICAL_HEADERS})
        set(HEADER_FOUND FALSE)
        foreach(dir ${DISCOVERED_INCLUDE_PATHS})
            if(EXISTS "${dir}/${header}")
                set(HEADER_FOUND TRUE)
                break()
            endif()
        endforeach()
        
        if(NOT HEADER_FOUND)
            list(APPEND MISSING_HEADERS ${header})
            message(WARNING "Critical header ${header} not found in discovered paths")
            
            # Try to find it
            execute_process(
                COMMAND find /usr -name ${header} -type f -print -quit
                OUTPUT_VARIABLE FOUND_HEADER
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                TIMEOUT 5
            )
            if(FOUND_HEADER)
                get_filename_component(FOUND_DIR "${FOUND_HEADER}" DIRECTORY)
                message(STATUS "  Found ${header} in ${FOUND_DIR}")
                list(APPEND DISCOVERED_INCLUDE_PATHS "${FOUND_DIR}")
            endif()
        endif()
    endforeach()
    
    # Remove duplicates again after adding missing headers
    list(REMOVE_DUPLICATES DISCOVERED_INCLUDE_PATHS)
    
    # Convert to -I flags
    set(SYSTEM_INCLUDE_FLAGS "")
    foreach(dir ${DISCOVERED_INCLUDE_PATHS})
        set(SYSTEM_INCLUDE_FLAGS "${SYSTEM_INCLUDE_FLAGS} -I${dir}")
    endforeach()
    
    message(STATUS "Found ${CMAKE_WORDS_BIGENDIAN} system include directories")
    message(STATUS "System include flags: ${SYSTEM_INCLUDE_FLAGS}")
    
    # Write the discovered paths to a file for debugging
    file(WRITE "${CMAKE_BINARY_DIR}/discovered_includes.txt" "${SYSTEM_INCLUDE_FLAGS}")
    
    # ============================================================================
    # GENERATE WRAPPER SCRIPT WITH DISCOVERED PATHS
    # ============================================================================
    
    set(PPSTEP_WRAPPER "${CMAKE_BINARY_DIR}/ppstep-nd4j")
    
    # Generate wrapper with the discovered paths baked in
    file(WRITE ${PPSTEP_WRAPPER}
"#!/bin/bash
# Auto-generated ppstep wrapper for libnd4j
# Generated at CMake configure time with discovered include paths

PPSTEP_BIN=\"${PPSTEP_BUILD_DIR}/ppstep\"

if [ ! -f \"\$PPSTEP_BIN\" ]; then
    echo \"Error: ppstep not built yet. Run 'make ppstep_build' first\"
    exit 1
fi

# System includes discovered at CMake configure time
SYSTEM_INCLUDES=\"${SYSTEM_INCLUDE_FLAGS}\"

# Additional runtime discovery (fallback)
if [ -z \"\$SYSTEM_INCLUDES\" ] || [ \"\$1\" = \"--rediscover\" ]; then
    echo \"Warning: No includes from CMake or rediscovery requested\" >&2
    
    # Try runtime discovery
    RUNTIME_INCLUDES=\"\"
    
    # Method 1: cpp -v
    if command -v cpp &> /dev/null; then
        while IFS= read -r dir; do
            [ -d \"\$dir\" ] && RUNTIME_INCLUDES=\"\$RUNTIME_INCLUDES -I\$dir\"
        done < <(cpp -x c++ -v < /dev/null 2>&1 | awk '
            /^#include <...> search starts here:/ { flag=1; next }
            /^End of search list./ { flag=0 }
            flag { gsub(/^[ \t]+/, \"\"); print }
        ')
    fi
    
    # Method 2: Direct paths
    for dir in /usr/include /usr/local/include /usr/include/linux; do
        [ -d \"\$dir\" ] && RUNTIME_INCLUDES=\"\$RUNTIME_INCLUDES -I\$dir\"
    done
    
    SYSTEM_INCLUDES=\"\$RUNTIME_INCLUDES\"
fi

# Include paths for libnd4j
INCLUDES=\"-I${CMAKE_CURRENT_SOURCE_DIR}/include\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_BINARY_DIR}/include\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/array\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/blas\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/cnpy\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/execution\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/flatbuffers\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/generated\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/graph\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/helpers\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/indexing\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/legacy\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/loops\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/math\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/memory\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/ops\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/system\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/types\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/performance\"

# Add OpenBLAS if available
if [ -n \"${OPENBLAS_PATH}\" ] && [ -d \"${OPENBLAS_PATH}/include\" ]; then
    INCLUDES=\"\$INCLUDES -I${OPENBLAS_PATH}/include\"
fi

# Basic defines
DEFINES=\"-D__CPUBLAS__=1 -DSD_CPU=1\"
DEFINES=\"\$DEFINES -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS\"
DEFINES=\"\$DEFINES -D_GNU_SOURCE\"

# Handle special arguments
case \"\$1\" in
    --debug|--show-includes)
        echo \"System includes:\"
        echo \$SYSTEM_INCLUDES | tr ' ' '\\n' | grep '^-I' | while read inc; do
            dir=\${inc#-I}
            echo \"  \$inc\"
            for h in limits.h stdio.h stdlib.h stddef.h; do
                [ -f \"\$dir/\$h\" ] && echo \"    ✓ has \$h\"
            done
        done
        echo \"\"
        echo \"Project includes:\"
        echo \$INCLUDES | tr ' ' '\\n' | grep '^-I'
        echo \"\"
        echo \"Defines:\"
        echo \$DEFINES | tr ' ' '\\n'
        exit 0
        ;;
    --version)
        echo \"ppstep wrapper generated by CMake\"
        echo \"Binary: \$PPSTEP_BIN\"
        \$PPSTEP_BIN --version 2>/dev/null || echo \"ppstep version unknown\"
        exit 0
        ;;
    -h|--help)
        echo \"Usage: \$0 [options] <source_file.cpp>\"
        echo \"Options:\"
        echo \"  --debug, --show-includes  Show include paths and exit\"
        echo \"  --rediscover             Force runtime include discovery\"
        echo \"  --version                Show version information\"
        echo \"  -h, --help              Show this help\"
        exit 0
        ;;
esac

# Remove special arguments if rediscovery was requested
[ \"\$1\" = \"--rediscover\" ] && shift

# Run ppstep
exec \"\$PPSTEP_BIN\" \$SYSTEM_INCLUDES \$INCLUDES \$DEFINES \"\$@\"
")
    
    # Make wrapper executable
    file(CHMOD ${PPSTEP_WRAPPER}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
                    GROUP_READ GROUP_EXECUTE 
                    WORLD_READ WORLD_EXECUTE)
    
    # Also generate a CMake include file with the discovered paths
    file(WRITE "${CMAKE_BINARY_DIR}/ppstep_includes.cmake"
"# Auto-generated by Ppstep.cmake
# System include directories discovered during configuration

set(PPSTEP_SYSTEM_INCLUDE_DIRS
")
    foreach(dir ${DISCOVERED_INCLUDE_PATHS})
        file(APPEND "${CMAKE_BINARY_DIR}/ppstep_includes.cmake" "    \"${dir}\"\n")
    endforeach()
    file(APPEND "${CMAKE_BINARY_DIR}/ppstep_includes.cmake" ")\n")
    
    # Add custom target
    add_custom_target(ppstep ALL
        DEPENDS ppstep_build
        COMMAND ${CMAKE_COMMAND} -E echo "✅ ppstep built successfully"
        COMMAND ${CMAKE_COMMAND} -E echo "   Executable: ${PPSTEP_BUILD_DIR}/ppstep"
        COMMAND ${CMAKE_COMMAND} -E echo "   Wrapper: ${PPSTEP_WRAPPER}"
        COMMAND ${CMAKE_COMMAND} -E echo "   Include paths: ${CMAKE_BINARY_DIR}/discovered_includes.txt"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "Usage:"
        COMMAND ${CMAKE_COMMAND} -E echo "   ${PPSTEP_WRAPPER} [source_file.cpp]"
        COMMAND ${CMAKE_COMMAND} -E echo "   ${PPSTEP_WRAPPER} --debug  # Show include paths"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMENT "ppstep preprocessor tool ready"
    )
    
    # Custom target to regenerate wrapper if needed
    add_custom_target(ppstep-regenerate
        COMMAND ${CMAKE_COMMAND} -E echo "Regenerating ppstep wrapper..."
        COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}
        COMMENT "Regenerating ppstep wrapper with updated paths"
    )
    
    message(STATUS "✅ ppstep target configured")
    message(STATUS "Exiting after ppstep setup - run 'make' to build ppstep")
    
    # CRITICAL: EARLY EXIT - Don't process the rest of CMakeLists.txt
    return()
endif()