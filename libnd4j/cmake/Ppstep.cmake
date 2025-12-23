# ==============================================================================
# cmake/Ppstep.cmake
# Builds ppstep preprocessor debugger tool
# ==============================================================================

if(BUILD_PPSTEP)
    message(STATUS "🔧 PPSTEP BUILD MODE - Building ppstep tool only")
    message(STATUS "Setting up ppstep build in: ${CMAKE_BINARY_DIR}")
    
    # Detect linuxbrew Boost installation - pick the newest version
    file(GLOB LINUXBREW_BOOST_DIRS "/home/linuxbrew/.linuxbrew/Cellar/boost/*")
    if(LINUXBREW_BOOST_DIRS)
        list(SORT LINUXBREW_BOOST_DIRS)
        list(REVERSE LINUXBREW_BOOST_DIRS)
        list(GET LINUXBREW_BOOST_DIRS 0 BOOST_ROOT)
        message(STATUS "🍺 Using linuxbrew Boost at: ${BOOST_ROOT}")
        
        set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
        set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
        
        message(STATUS "   Include dir: ${BOOST_INCLUDEDIR}")
        message(STATUS "   Library dir: ${BOOST_LIBRARYDIR}")
    else()
        message(FATAL_ERROR "❌ No Boost found in /home/linuxbrew/.linuxbrew/Cellar/boost/")
    endif()
    
    include(ExternalProject)
    
    # Configure ppstep as external project
    set(PPSTEP_PREFIX "${CMAKE_BINARY_DIR}/ppstep_external")
    set(PPSTEP_SOURCE_DIR "${PPSTEP_PREFIX}/src/ppstep")
    set(PPSTEP_BUILD_DIR "${PPSTEP_PREFIX}/build")
    
    if(EXISTS "${PPSTEP_PREFIX}/src/ppstep_build-stamp")
        file(REMOVE_RECURSE "${PPSTEP_PREFIX}/src/ppstep_build-stamp")
    endif()
    
    message(STATUS "Cloning and building ppstep...")
    
    set(PPSTEP_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DBOOST_ROOT=${BOOST_ROOT}
        -DBOOST_INCLUDEDIR=${BOOST_INCLUDEDIR}
        -DBOOST_LIBRARYDIR=${BOOST_LIBRARYDIR}
    )
    
    if(CMAKE_CXX_COMPILER)
        list(APPEND PPSTEP_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
    endif()
    
    if(CMAKE_C_COMPILER)
        list(APPEND PPSTEP_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
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
    # INCLUDE PATH DISCOVERY
    # ============================================================================
    
    message(STATUS "Discovering system include paths...")
    
    set(DISCOVERED_INCLUDE_PATHS "")
    
    foreach(dir ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        if(EXISTS "${dir}")
            list(APPEND DISCOVERED_INCLUDE_PATHS "${dir}")
        endif()
    endforeach()
    
    if(CMAKE_CXX_COMPILER)
        execute_process(
            COMMAND echo ""
            COMMAND ${CMAKE_CXX_COMPILER} -E -x c++ -Wp,-v -
            OUTPUT_VARIABLE COMPILER_OUTPUT
            ERROR_VARIABLE COMPILER_ERROR
            INPUT_FILE /dev/null
            RESULT_VARIABLE COMPILER_RESULT
        )
        
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
    endif()
    
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=include
        OUTPUT_VARIABLE GCC_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(EXISTS "${GCC_INCLUDE_DIR}")
        list(APPEND DISCOVERED_INCLUDE_PATHS "${GCC_INCLUDE_DIR}")
    endif()
    
    set(COMMON_INCLUDE_PATHS
        /usr/include
        /usr/local/include
        /usr/include/linux
        /usr/include/x86_64-linux-gnu
    )
    
    foreach(path ${COMMON_INCLUDE_PATHS})
        if(EXISTS "${path}")
            list(APPEND DISCOVERED_INCLUDE_PATHS "${path}")
        endif()
    endforeach()
    
    foreach(version RANGE 5 15)
        set(cpp_paths
            "/usr/include/c++/${version}"
            "/usr/include/c++/${version}/x86_64-linux-gnu"
            "/usr/include/c++/${version}/backward"
        )
        foreach(path ${cpp_paths})
            if(EXISTS "${path}")
                list(APPEND DISCOVERED_INCLUDE_PATHS "${path}")
            endif()
        endforeach()
    endforeach()
    
    list(REMOVE_DUPLICATES DISCOVERED_INCLUDE_PATHS)
    
    set(SYSTEM_INCLUDE_FLAGS "")
    foreach(dir ${DISCOVERED_INCLUDE_PATHS})
        set(SYSTEM_INCLUDE_FLAGS "${SYSTEM_INCLUDE_FLAGS} -I${dir}")
    endforeach()
    
    list(LENGTH DISCOVERED_INCLUDE_PATHS NUM_INCLUDE_DIRS)
    message(STATUS "Found ${NUM_INCLUDE_DIRS} system include directories")
    
    file(WRITE "${CMAKE_BINARY_DIR}/discovered_includes.txt" "${SYSTEM_INCLUDE_FLAGS}")
    
    # ============================================================================
    # GENERATE WRAPPER SCRIPT
    # ============================================================================
    
    set(PPSTEP_WRAPPER "${CMAKE_BINARY_DIR}/ppstep-nd4j")
    
    file(WRITE ${PPSTEP_WRAPPER}
"#!/bin/bash
PPSTEP_BIN=\"${PPSTEP_BUILD_DIR}/ppstep\"

if [ ! -f \"\$PPSTEP_BIN\" ]; then
    echo \"Error: ppstep not built yet. Run 'make ppstep_build' first\"
    exit 1
fi

SYSTEM_INCLUDES=\"${SYSTEM_INCLUDE_FLAGS}\"

INCLUDES=\"-I${CMAKE_CURRENT_SOURCE_DIR}/include\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_BINARY_DIR}/include\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/array\"
INCLUDES=\"\$INCLUDES -I${CMAKE_CURRENT_SOURCE_DIR}/include/ops\"

DEFINES=\"-D__CPUBLAS__=1 -DSD_CPU=1\"
DEFINES=\"\$DEFINES -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -D_GNU_SOURCE\"

case \"\$1\" in
    --debug|--show-includes)
        echo \"System includes:\"
        echo \$SYSTEM_INCLUDES | tr ' ' '\\n' | grep '^-I'
        echo \"\"
        echo \"Project includes:\"
        echo \$INCLUDES | tr ' ' '\\n' | grep '^-I'
        exit 0
        ;;
    -h|--help)
        echo \"ppstep-nd4j - Preprocessor debugger for libnd4j\"
        echo \"USAGE: ppstep-nd4j [options] <source_file.cpp>\"
        exit 0
        ;;
esac

exec \"\$PPSTEP_BIN\" \$SYSTEM_INCLUDES \$INCLUDES \$DEFINES \"\$@\"
")
    
    file(CHMOD ${PPSTEP_WRAPPER}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
                    GROUP_READ GROUP_EXECUTE 
                    WORLD_READ WORLD_EXECUTE)
    
    add_custom_target(ppstep ALL
        DEPENDS ppstep_build
        COMMAND ${CMAKE_COMMAND} -E echo "✅ ✅ ppstep build complete"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "📍 PPSTEP LOCATIONS:"
        COMMAND ${CMAKE_COMMAND} -E echo "   Executable: ${PPSTEP_BUILD_DIR}/ppstep"
        COMMAND ${CMAKE_COMMAND} -E echo "   Wrapper: ${PPSTEP_WRAPPER}"
        COMMAND ${CMAKE_COMMAND} -E echo "   Include paths file: ${CMAKE_BINARY_DIR}/discovered_includes.txt"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "📖 USAGE:"
        COMMAND ${CMAKE_COMMAND} -E echo "   ${PPSTEP_WRAPPER} source_file.cpp"
        COMMAND ${CMAKE_COMMAND} -E echo "   ${PPSTEP_WRAPPER} --debug  # Show include paths"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "ℹ️  EXAMPLES:"
        COMMAND ${CMAKE_COMMAND} -E echo "   # Preprocess a single file"
        COMMAND ${CMAKE_COMMAND} -E echo "   ${PPSTEP_WRAPPER} include/ops/declarable/generic/parity_ops/unique.cpp"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "   # Debug mode - show all include paths being used"
        COMMAND ${CMAKE_COMMAND} -E echo "   ${PPSTEP_WRAPPER} --debug"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMENT "ppstep preprocessor tool ready"
    )
    
    message(STATUS "✅ ppstep target configured")
    message(STATUS "Run 'make' to build ppstep")
    
    return()
endif()
