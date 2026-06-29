# ============================================================================
# PartialLinking.cmake - Configurable Linker for Large Binaries
# ============================================================================
#
# PROBLEM:
# When the total object file size exceeds ~2GB, .eh_frame sections contain
# PC32 relocations that overflow. These relocations reference local symbols
# within object files, and when the final binary layout places sections too
# far apart, the 32-bit relative offsets overflow.
#
# SOLUTION:
# Use a modern linker (mold, gold, lld) that handles large binaries properly.
# The linker can be configured via SD_LINKER option.
#
# ============================================================================

include_guard(GLOBAL)

# ============================================================================
# Option: SD_LINKER
# Specify which linker to use: auto, mold, gold, lld, bfd
# ============================================================================
set(SD_LINKER "auto" CACHE STRING "Linker to use for large binaries (auto, mold, gold, lld, bfd)")
set_property(CACHE SD_LINKER PROPERTY STRINGS auto mold gold lld bfd)

# ============================================================================
# Function: sd_detect_best_linker
# Auto-detect the best available linker for large binaries
# ============================================================================
function(sd_detect_best_linker RESULT_VAR RESULT_FLAG_VAR)
    # Priority order for large binary handling:
    # 1. mold - fastest, best large binary support
    # 2. lld - LLVM linker, good large binary support
    # 3. gold - GNU gold, decent large binary support
    # 4. bfd - GNU ld, fallback (may fail on >2GB)

    find_program(MOLD_FOUND mold)
    find_program(LLD_FOUND ld.lld)
    find_program(GOLD_FOUND ld.gold)
    find_program(BFD_FOUND ld.bfd)

    if(MOLD_FOUND)
        set(${RESULT_VAR} "mold" PARENT_SCOPE)
        set(${RESULT_FLAG_VAR} "-fuse-ld=mold" PARENT_SCOPE)
        message(STATUS "Auto-detected linker: mold (best for large binaries)")
    elseif(LLD_FOUND)
        set(${RESULT_VAR} "lld" PARENT_SCOPE)
        set(${RESULT_FLAG_VAR} "-fuse-ld=lld" PARENT_SCOPE)
        message(STATUS "Auto-detected linker: lld (good for large binaries)")
    elseif(GOLD_FOUND)
        set(${RESULT_VAR} "gold" PARENT_SCOPE)
        set(${RESULT_FLAG_VAR} "-fuse-ld=gold" PARENT_SCOPE)
        message(STATUS "Auto-detected linker: gold (acceptable for large binaries)")
    elseif(BFD_FOUND)
        set(${RESULT_VAR} "bfd" PARENT_SCOPE)
        set(${RESULT_FLAG_VAR} "-fuse-ld=bfd" PARENT_SCOPE)
        message(WARNING "Auto-detected linker: bfd (may fail on >2GB binaries)")
    else()
        set(${RESULT_VAR} "default" PARENT_SCOPE)
        set(${RESULT_FLAG_VAR} "" PARENT_SCOPE)
        message(WARNING "No specific linker found, using system default")
    endif()
endfunction()

# ============================================================================
# Function: sd_get_linker_config
# Get the linker configuration based on SD_LINKER setting
# ============================================================================
function(sd_get_linker_config LINKER_NAME_VAR LINKER_FLAG_VAR LINKER_EXTRA_FLAGS_VAR)
    if(SD_LINKER STREQUAL "auto")
        sd_detect_best_linker(DETECTED_LINKER DETECTED_FLAG)
        set(${LINKER_NAME_VAR} "${DETECTED_LINKER}" PARENT_SCOPE)
        set(${LINKER_FLAG_VAR} "${DETECTED_FLAG}" PARENT_SCOPE)
    elseif(SD_LINKER STREQUAL "mold")
        set(${LINKER_NAME_VAR} "mold" PARENT_SCOPE)
        set(${LINKER_FLAG_VAR} "-fuse-ld=mold" PARENT_SCOPE)
    elseif(SD_LINKER STREQUAL "gold")
        set(${LINKER_NAME_VAR} "gold" PARENT_SCOPE)
        set(${LINKER_FLAG_VAR} "-fuse-ld=gold" PARENT_SCOPE)
    elseif(SD_LINKER STREQUAL "lld")
        set(${LINKER_NAME_VAR} "lld" PARENT_SCOPE)
        set(${LINKER_FLAG_VAR} "-fuse-ld=lld" PARENT_SCOPE)
    elseif(SD_LINKER STREQUAL "bfd")
        set(${LINKER_NAME_VAR} "bfd" PARENT_SCOPE)
        set(${LINKER_FLAG_VAR} "-fuse-ld=bfd" PARENT_SCOPE)
    else()
        message(WARNING "Unknown linker '${SD_LINKER}', using auto-detection")
        sd_detect_best_linker(DETECTED_LINKER DETECTED_FLAG)
        set(${LINKER_NAME_VAR} "${DETECTED_LINKER}" PARENT_SCOPE)
        set(${LINKER_FLAG_VAR} "${DETECTED_FLAG}" PARENT_SCOPE)
    endif()

    # Set linker-specific extra flags
    if(${LINKER_NAME_VAR} STREQUAL "mold" OR DETECTED_LINKER STREQUAL "mold")
        # Mold-specific flags for large binaries
        set(${LINKER_EXTRA_FLAGS_VAR} "-Wl,--threads -Wl,-z,notext -Wl,--hash-style=gnu" PARENT_SCOPE)
    elseif(${LINKER_NAME_VAR} STREQUAL "gold" OR DETECTED_LINKER STREQUAL "gold")
        # Gold-specific flags for large binaries
        set(${LINKER_EXTRA_FLAGS_VAR} "-Wl,--no-keep-memory -Wl,-z,notext -Wl,--no-relax" PARENT_SCOPE)
    elseif(${LINKER_NAME_VAR} STREQUAL "lld" OR DETECTED_LINKER STREQUAL "lld")
        # LLD-specific flags for large binaries
        set(${LINKER_EXTRA_FLAGS_VAR} "-Wl,-z,notext -Wl,--hash-style=gnu" PARENT_SCOPE)
    else()
        # Default/bfd flags
        set(${LINKER_EXTRA_FLAGS_VAR} "-Wl,-z,notext" PARENT_SCOPE)
    endif()
endfunction()

# ============================================================================
# Function: sd_create_library_with_partial_linking
# Creates the shared library using configurable linker for large binaries
# ============================================================================
function(sd_create_library_with_partial_linking LIB_NAME OBJECT_LIB_NAME)
    # Get linker configuration
    sd_get_linker_config(LINKER_NAME LINKER_FLAG LINKER_EXTRA_FLAGS)

    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════════╗")
    message(STATUS "║  LARGE BINARY (>2GB) LINKING CONFIGURATION                       ║")
    message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
    message(STATUS "║  Linker: ${LINKER_NAME}")
    message(STATUS "║  Flag: ${LINKER_FLAG}")
    message(STATUS "║  Extra flags: ${LINKER_EXTRA_FLAGS}")
    message(STATUS "╚══════════════════════════════════════════════════════════════════╝")
    message(STATUS "")

    # Get the object directory
    set(OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${OBJECT_LIB_NAME}.dir")
    set(LINK_DIR "${CMAKE_CURRENT_BINARY_DIR}/direct_link")
    file(MAKE_DIRECTORY "${LINK_DIR}")

    # Create the shared library output path
    set(LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/lib${LIB_NAME}.so")

    # Find the CUDA toolkit for library paths
    if(SD_CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}")
        message(STATUS "Large binary linking: CUDA library dir: ${CUDA_LIB_DIR}")
    endif()

    # Build the link libraries list
    set(LINK_LIBS "")
    if(SD_CUDA AND CUDA_LIB_DIR)
        list(APPEND LINK_LIBS "-L${CUDA_LIB_DIR}")
        list(APPEND LINK_LIBS "-Wl,-rpath,${CUDA_LIB_DIR}")
        list(APPEND LINK_LIBS "-lcudart")
        list(APPEND LINK_LIBS "-lcublas")
        list(APPEND LINK_LIBS "-lcusolver")
        list(APPEND LINK_LIBS "-lcusparse")
        list(APPEND LINK_LIBS "-lcurand")
        # Add cuDNN if available
        if(HAVE_CUDNN)
            list(APPEND LINK_LIBS "-lcudnn")
        endif()
    endif()

    # Add other required libraries
    if(SD_GCC_FUNCTRACE AND NOT WIN32)
        list(APPEND LINK_LIBS "-ldw")
        list(APPEND LINK_LIBS "-lelf")
        list(APPEND LINK_LIBS "-lbfd")
        list(APPEND LINK_LIBS "-ldl")
        list(APPEND LINK_LIBS "-lpthread")
    endif()

    # Add OpenBLAS if available
    if(OPENBLAS_LIBRARIES)
        list(APPEND LINK_LIBS "${OPENBLAS_LIBRARIES}")
    endif()

    # Convert CMake list (semicolon-separated) to space-separated for bash
    string(REPLACE ";" " " LINK_LIBS_STR "${LINK_LIBS}")

    # Create the linking script with configurable linker
    set(LINK_SCRIPT "${LINK_DIR}/link_library.sh")

    file(WRITE "${LINK_SCRIPT}" "#!/bin/bash
# Auto-generated linking script for large binary (>2GB) support
# Linker: ${LINKER_NAME}
# Generated by CMake with SD_LINKER=${SD_LINKER}
set -e

echo \"\"
echo \"╔══════════════════════════════════════════════════════════════════╗\"
echo \"║  LARGE BINARY LINKING                                            ║\"
echo \"║  Linker: ${LINKER_NAME}\"
echo \"╚══════════════════════════════════════════════════════════════════╝\"
echo \"\"

OBJ_DIR=\"${OBJ_DIR}\"
LINK_DIR=\"${LINK_DIR}\"
OUTPUT=\"${LIB_OUTPUT}\"
LINKER_NAME=\"${LINKER_NAME}\"
LINKER_FLAG=\"${LINKER_FLAG}\"

# Verify linker is available
case \"\$LINKER_NAME\" in
    mold)
        if ! command -v mold &> /dev/null; then
            echo \"ERROR: mold linker not found. Install with: sudo apt install mold\"
            echo \"Or set SD_LINKER to a different linker (gold, lld, bfd)\"
            exit 1
        fi
        echo \"Using mold linker: \$(which mold)\"
        echo \"Mold version: \$(mold --version 2>/dev/null || echo 'unknown')\"
        ;;
    gold)
        if ! command -v ld.gold &> /dev/null; then
            echo \"ERROR: gold linker not found. Install with: sudo apt install binutils\"
            exit 1
        fi
        echo \"Using gold linker: \$(which ld.gold)\"
        ;;
    lld)
        if ! command -v ld.lld &> /dev/null; then
            echo \"ERROR: lld linker not found. Install with: sudo apt install lld\"
            exit 1
        fi
        echo \"Using lld linker: \$(which ld.lld)\"
        echo \"LLD version: \$(ld.lld --version 2>/dev/null || echo 'unknown')\"
        ;;
    bfd)
        if ! command -v ld.bfd &> /dev/null; then
            echo \"WARNING: ld.bfd not found, using default ld\"
            LINKER_FLAG=\"\"
        else
            echo \"Using bfd linker: \$(which ld.bfd)\"
        fi
        echo \"WARNING: bfd linker may fail on binaries >2GB\"
        ;;
    *)
        echo \"Using system default linker\"
        ;;
esac

# Collect all object files
echo \"\"
echo \"Collecting object files from: \$OBJ_DIR\"
OBJECTS_FILE=\"\$LINK_DIR/objects.txt\"
find \"\$OBJ_DIR\" -name '*.o' 2>/dev/null | sort > \"\$OBJECTS_FILE\"

OBJ_COUNT=\$(wc -l < \"\$OBJECTS_FILE\")
echo \"Found \$OBJ_COUNT object files\"

if [ \$OBJ_COUNT -eq 0 ]; then
    echo \"ERROR: No object files found\"
    exit 1
fi

# Calculate total size of object files
if command -v bc &> /dev/null; then
    TOTAL_SIZE=\$(find \"\$OBJ_DIR\" -name '*.o' -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1)
    TOTAL_SIZE_GB=\$(echo \"scale=2; \$TOTAL_SIZE / 1073741824\" | bc)
    echo \"Total object file size: \${TOTAL_SIZE_GB}GB\"

    # Warn if using bfd with large binary
    if [ \"\$LINKER_NAME\" = \"bfd\" ] || [ -z \"\$LINKER_FLAG\" ]; then
        SIZE_CHECK=\$(echo \"\$TOTAL_SIZE > 2147483648\" | bc)
        if [ \"\$SIZE_CHECK\" -eq 1 ]; then
            echo \"\"
            echo \"WARNING: Binary size exceeds 2GB and using bfd/default linker!\"
            echo \"This may cause relocation overflow errors.\"
            echo \"Consider using: cmake -DSD_LINKER=mold ..\"
            echo \"\"
        fi
    fi
fi

# Use response file for object list (command line may be too long)
RESPONSE_FILE=\"\$LINK_DIR/objects.rsp\"
while IFS= read -r obj; do
    echo \"\$obj\"
done < \"\$OBJECTS_FILE\" > \"\$RESPONSE_FILE\"

echo \"\"
echo \"Linking with \$LINKER_NAME...\"
echo \"Command: ${CMAKE_CXX_COMPILER} \$LINKER_FLAG -shared -o \$OUTPUT ...\"
echo \"\"

# Run the linker
${CMAKE_CXX_COMPILER} \$LINKER_FLAG -shared \\
    -o \"\$OUTPUT\" \\
    ${CMAKE_SHARED_LINKER_FLAGS} \\
    @\"\$RESPONSE_FILE\" \\
    ${LINK_LIBS_STR} \\
    ${LINKER_EXTRA_FLAGS}

echo \"\"
echo \"Linking complete!\"
echo \"Output: \$OUTPUT\"
ls -lh \"\$OUTPUT\"

# Show some stats about the library
echo \"\"
echo \"Library sections:\"
size \"\$OUTPUT\" 2>/dev/null || true
")

    file(CHMOD "${LINK_SCRIPT}"
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE
                     WORLD_READ WORLD_EXECUTE)

    # Create custom target for object collection (depends on object library being built)
    add_custom_target(${LIB_NAME}_collect_objects
        DEPENDS ${OBJECT_LIB_NAME}
        COMMENT "Objects ready for linking ${LIB_NAME}"
    )

    add_custom_command(
        OUTPUT "${LIB_OUTPUT}"
        COMMAND "${LINK_SCRIPT}"
        DEPENDS ${LIB_NAME}_collect_objects
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Linking ${LIB_NAME} with ${LINKER_NAME} linker"
        VERBATIM
    )

    add_custom_target(${LIB_NAME}_final ALL
        DEPENDS "${LIB_OUTPUT}"
    )

    # Create an imported target so other targets can depend on it
    add_library(${LIB_NAME} SHARED IMPORTED GLOBAL)
    set_target_properties(${LIB_NAME} PROPERTIES
        IMPORTED_LOCATION "${LIB_OUTPUT}"
    )
    add_dependencies(${LIB_NAME} ${LIB_NAME}_final)

    # Set up INTERFACE include directories for the imported target
    set_target_properties(${LIB_NAME} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${CMAKE_CURRENT_BINARY_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/include/array;${CMAKE_CURRENT_SOURCE_DIR}/include/execution;${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions;${CMAKE_CURRENT_SOURCE_DIR}/include/graph;${CMAKE_CURRENT_SOURCE_DIR}/include/helpers;${CMAKE_CURRENT_SOURCE_DIR}/include/loops;${CMAKE_CURRENT_SOURCE_DIR}/include/memory;${CMAKE_CURRENT_SOURCE_DIR}/include/ops;${CMAKE_CURRENT_SOURCE_DIR}/include/types;${CMAKE_CURRENT_SOURCE_DIR}/include/system;${CMAKE_CURRENT_SOURCE_DIR}/include/legacy;${CMAKE_CURRENT_SOURCE_DIR}/include/performance;${CMAKE_CURRENT_SOURCE_DIR}/include/indexing;${CMAKE_CURRENT_SOURCE_DIR}/include/generated;${CMAKE_BINARY_DIR}/compilation_units;${CMAKE_BINARY_DIR}/cuda_instantiations"
    )

    # Mark that this target was created via this module (IMPORTED target)
    # This signals to MainBuildFlow.cmake to skip operations that don't work on IMPORTED targets
    set(${LIB_NAME}_PARTIAL_LINKED TRUE PARENT_SCOPE)
    set(SD_PARTIAL_LINKED_TARGET TRUE PARENT_SCOPE)

    message(STATUS "Large binary linking configured for ${LIB_NAME}")
    message(STATUS "Linker: ${LINKER_NAME} (${LINKER_FLAG})")
    message(STATUS "Output: ${LIB_OUTPUT}")
endfunction()

# ============================================================================
# Function: sd_should_use_partial_linking
# Determines if special large binary linking should be used
# ============================================================================
function(sd_should_use_partial_linking RESULT_VAR)
    if(SD_GCC_FUNCTRACE AND SD_CUDA)
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()
