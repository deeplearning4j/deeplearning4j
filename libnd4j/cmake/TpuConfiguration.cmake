################################################################################
# TPU/PJRT Configuration Functions
# Functions for TPU-specific build configuration using PJRT (Portable Runtime)
################################################################################

# Enhanced PJRT toolkit detection with proper include path setup
function(setup_pjrt_paths)
    message(STATUS "Setting up PJRT/TPU paths...")

    set(PJRT_SEARCH_PATHS
            # Environment variables
            $ENV{PJRT_PATH}
            $ENV{PJRT_ROOT}
            $ENV{XLA_PATH}
            $ENV{TPU_LIBRARY_PATH}

            # Common installation paths
            /usr/local/lib/python3.10/dist-packages/jax_plugins/xla_tpu
            /usr/local/lib/python3.11/dist-packages/jax_plugins/xla_tpu
            /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_tpu

            # Google Cloud TPU paths
            /opt/google-cloud-tpu
            /usr/share/tpu

            # System paths
            /usr/local
            /usr
            /opt
    )

    # Search for PJRT C API header
    find_path(PJRT_INCLUDE_DIR
            NAMES pjrt_c_api.h
            HINTS ${PJRT_SEARCH_PATHS}
            PATH_SUFFIXES
            include
            include/xla/pjrt/c
            xla/pjrt/c
            pjrt/c
            NO_DEFAULT_PATH
    )

    # If not found, try system paths
    if(NOT PJRT_INCLUDE_DIR)
        find_path(PJRT_INCLUDE_DIR
                NAMES pjrt_c_api.h
                PATHS /usr/include /usr/local/include
                PATH_SUFFIXES xla/pjrt/c pjrt/c
        )
    endif()

    # Search for PJRT library
    find_library(PJRT_LIBRARY
            NAMES pjrt_c_api libtpu tpu_client pjrt
            HINTS ${PJRT_SEARCH_PATHS}
            PATH_SUFFIXES
            lib
            lib64
            NO_DEFAULT_PATH
    )

    # If not found, try system paths
    if(NOT PJRT_LIBRARY)
        find_library(PJRT_LIBRARY
                NAMES pjrt_c_api libtpu tpu_client pjrt
                PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
        )
    endif()

    message(STATUS "PJRT search results:")
    message(STATUS "   PJRT_INCLUDE_DIR: ${PJRT_INCLUDE_DIR}")
    message(STATUS "   PJRT_LIBRARY: ${PJRT_LIBRARY}")

    # Check if we found both header and library
    if(PJRT_INCLUDE_DIR AND PJRT_LIBRARY)
        message(STATUS "PJRT found!")

        # Create imported target if it doesn't exist
        if(NOT TARGET PJRT::pjrt)
            add_library(PJRT::pjrt UNKNOWN IMPORTED)
            set_target_properties(PJRT::pjrt PROPERTIES
                    IMPORTED_LOCATION "${PJRT_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${PJRT_INCLUDE_DIR}"
            )
        endif()

        message(STATUS "PJRT configuration:")
        message(STATUS "   Include: ${PJRT_INCLUDE_DIR}")
        message(STATUS "   Library: ${PJRT_LIBRARY}")

        # Set all the variables that might be needed
        set(HAVE_PJRT TRUE PARENT_SCOPE)
        set(PJRT_FOUND TRUE PARENT_SCOPE)
        set(PJRT_INCLUDE_DIR "${PJRT_INCLUDE_DIR}" PARENT_SCOPE)
        set(PJRT_LIBRARIES "${PJRT_LIBRARY}" PARENT_SCOPE)
        set(PJRT_LIBRARY "${PJRT_LIBRARY}" PARENT_SCOPE)

        return()
    endif()

    # Try package manager detection as fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_PJRT QUIET pjrt)
        if(PC_PJRT_FOUND)
            message(STATUS "Found PJRT via pkg-config")
            set(HAVE_PJRT TRUE PARENT_SCOPE)
            set(PJRT_INCLUDE_DIR "${PC_PJRT_INCLUDE_DIRS}" PARENT_SCOPE)
            set(PJRT_LIBRARIES "${PC_PJRT_LIBRARIES}" PARENT_SCOPE)
            return()
        endif()
    endif()

    message(STATUS "PJRT not found. Searched extensively in:")
    message(STATUS "   Environment variables: PJRT_PATH, PJRT_ROOT, XLA_PATH, TPU_LIBRARY_PATH")
    message(STATUS "   System paths: /usr, /usr/local, /opt")
    message(STATUS "   Google Cloud TPU: /opt/google-cloud-tpu")
    message(STATUS "")
    message(STATUS "To fix this issue:")
    message(STATUS "   1. Install JAX with TPU support: pip install jax[tpu]")
    message(STATUS "   2. Set PJRT_PATH to your PJRT/XLA installation")
    message(STATUS "   3. Or disable TPU with -DSD_TPU=OFF")

    set(HAVE_PJRT FALSE PARENT_SCOPE)
endfunction()

# Setup TPU compiler flags
function(build_tpu_compiler_flags)
    set(LOCAL_TPU_FLAGS "")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(LOCAL_TPU_FLAGS "${LOCAL_TPU_FLAGS} -fPIC")
        if(SD_GCC_FUNCTRACE)
            set(LOCAL_TPU_FLAGS "${LOCAL_TPU_FLAGS} -g -O0")
        else()
            set(LOCAL_TPU_FLAGS "${LOCAL_TPU_FLAGS} -O3")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(LOCAL_TPU_FLAGS "${LOCAL_TPU_FLAGS} -fPIC -O3")
    endif()

    set(TPU_COMPILER_FLAGS "${LOCAL_TPU_FLAGS}" PARENT_SCOPE)
    message(STATUS "TPU Compiler Flags: ${LOCAL_TPU_FLAGS}")
endfunction()

# Configure TPU linking for a target
function(configure_tpu_linking main_target_name)
    # Setup PJRT paths first
    setup_pjrt_paths()

    # Add PJRT include directories
    if(PJRT_INCLUDE_DIR)
        target_include_directories(${main_target_name} PUBLIC ${PJRT_INCLUDE_DIR})
        message(STATUS "Added PJRT include directories to ${main_target_name}: ${PJRT_INCLUDE_DIR}")
    endif()

    # Link against PJRT
    if(HAVE_PJRT AND TARGET PJRT::pjrt)
        message(STATUS "Linking with modern PJRT::pjrt target")
        target_link_libraries(${main_target_name} PUBLIC PJRT::pjrt)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_PJRT=1)
    elseif(HAVE_PJRT AND PJRT_LIBRARIES)
        message(STATUS "Linking with PJRT libraries: ${PJRT_LIBRARIES}")
        target_link_libraries(${main_target_name} PUBLIC ${PJRT_LIBRARIES})
        target_include_directories(${main_target_name} PUBLIC ${PJRT_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_PJRT=1)
    else()
        message(STATUS "Building without PJRT support")
        target_compile_definitions(${main_target_name} PUBLIC HAVE_PJRT=0)
    endif()

    target_link_libraries(${main_target_name} PUBLIC flatbuffers_interface)
    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

# Main TPU build setup function
function(setup_tpu_build)
    message(STATUS "=== TPU BUILD CONFIGURATION ===")

    if(NOT SD_TPU)
        message(STATUS "TPU build not enabled (SD_TPU=OFF)")
        return()
    endif()

    # Setup PJRT paths
    setup_pjrt_paths()

    if(NOT HAVE_PJRT)
        message(WARNING "PJRT not found. TPU support will be limited.")
    endif()

    # Build compiler flags
    build_tpu_compiler_flags()

    # Set TPU-specific compile definitions
    add_compile_definitions(SD_TPU=true)
    set(DEFAULT_ENGINE "samediff::ENGINE_TPU" PARENT_SCOPE)

    message(STATUS "=== TPU BUILD CONFIGURATION COMPLETE ===")
endfunction()

# Debug configuration function
function(debug_tpu_configuration)
    message(STATUS "=== TPU Configuration Debug Info ===")
    message(STATUS "SD_TPU: ${SD_TPU}")
    message(STATUS "HELPERS_pjrt: ${HELPERS_pjrt}")
    message(STATUS "TPU_VERSION: ${TPU_VERSION}")
    message(STATUS "HAVE_PJRT: ${HAVE_PJRT}")
    if(HAVE_PJRT)
        message(STATUS "PJRT_INCLUDE_DIR: ${PJRT_INCLUDE_DIR}")
        message(STATUS "PJRT_LIBRARIES: ${PJRT_LIBRARIES}")
    endif()
    message(STATUS "=== End TPU Debug Info ===")
endfunction()

# Function to ensure TPU paths are available at configure time
function(ensure_tpu_paths_available)
    if(NOT SD_TPU)
        return()
    endif()

    message(STATUS "Ensuring TPU/PJRT paths are available...")

    # Set up paths immediately
    setup_pjrt_paths()

    # Export to parent scope for immediate use
    set(PJRT_INCLUDE_DIR "${PJRT_INCLUDE_DIR}" PARENT_SCOPE)
    set(PJRT_LIBRARIES "${PJRT_LIBRARIES}" PARENT_SCOPE)
    set(HAVE_PJRT "${HAVE_PJRT}" PARENT_SCOPE)

    if(HAVE_PJRT)
        message(STATUS "TPU/PJRT paths configured and available")
    else()
        message(WARNING "TPU/PJRT paths not available - some features may be disabled")
    endif()
endfunction()
