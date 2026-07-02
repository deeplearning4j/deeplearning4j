################################################################################
# TPU/PJRT Configuration Functions
# Functions for TPU-specific build configuration using PJRT (Portable Runtime)
################################################################################

# Enhanced PJRT toolkit detection with proper include path setup
function(setup_pjrt_paths)
    message(STATUS "Setting up PJRT/TPU paths...")

    # Vendored header path — ships in-tree, no python/pip required at build time.
    # Pinned commit: 109c47c1bd003dc856fcfa940c1291700a4addb3 (openxla/xla)
    # $PJRT_PATH / $PJRT_ROOT still override when a runtime libtpu.so is available.
    set(_VENDORED_PJRT_INCLUDE "${CMAKE_SOURCE_DIR}/include/external/pjrt")

    set(PJRT_SEARCH_PATHS
            # Environment variables (override vendored copy when set)
            $ENV{PJRT_PATH}
            $ENV{PJRT_ROOT}
            $ENV{XLA_PATH}
            $ENV{TPU_LIBRARY_PATH}

            # In-tree vendored copy — always available, no external dependencies
            ${_VENDORED_PJRT_INCLUDE}

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

    # Search for PJRT C API header (vendored copy is tried via HINTS before NO_DEFAULT_PATH paths)
    find_path(PJRT_INCLUDE_DIR
            NAMES pjrt_c_api.h
            HINTS ${PJRT_SEARCH_PATHS}
            PATH_SUFFIXES
            include
            include/xla/pjrt/c
            xla/pjrt/c
            pjrt/c
            .
    )

    # If not found yet, try system paths
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

    # Header is the primary requirement.  The native library (libtpu.so) is
    # loaded at RUNTIME via dlopen()/dlsym() inside PjrtClientManager.cpp, so
    # it is NOT a link-time dependency.  We only need it to exist at cmake
    # configure time so we can create a proper imported CMake target (optional).
    if(PJRT_INCLUDE_DIR)
        message(STATUS "PJRT header found at: ${PJRT_INCLUDE_DIR}")

        if(PJRT_LIBRARY)
            message(STATUS "PJRT runtime library found at: ${PJRT_LIBRARY} (will be linked)")
            # Create imported target for optional link-time binding
            if(NOT TARGET PJRT::pjrt)
                add_library(PJRT::pjrt UNKNOWN IMPORTED)
                set_target_properties(PJRT::pjrt PROPERTIES
                        IMPORTED_LOCATION "${PJRT_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES "${PJRT_INCLUDE_DIR}"
                )
            endif()
            set(PJRT_LIBRARIES "${PJRT_LIBRARY}" PARENT_SCOPE)
            set(PJRT_LIBRARY "${PJRT_LIBRARY}" PARENT_SCOPE)
        else()
            message(STATUS "PJRT runtime library NOT found — header-only mode.")
            message(STATUS "   libtpu.so will be loaded at runtime via dlopen(). Build succeeds without it.")
            message(STATUS "   To enable link-time binding: set PJRT_PATH to a dir containing libtpu.so")
            set(PJRT_LIBRARIES "" PARENT_SCOPE)
            set(PJRT_LIBRARY "" PARENT_SCOPE)
        endif()

        set(HAVE_PJRT TRUE PARENT_SCOPE)
        set(PJRT_FOUND TRUE PARENT_SCOPE)
        set(PJRT_INCLUDE_DIR "${PJRT_INCLUDE_DIR}" PARENT_SCOPE)
        return()
    endif()

    # Try package manager detection as fallback (library only — header already searched above)
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

    message(STATUS "PJRT header (pjrt_c_api.h) not found anywhere. Searched:")
    message(STATUS "   Vendored:          ${CMAKE_SOURCE_DIR}/include/external/pjrt/")
    message(STATUS "   Env vars:          PJRT_PATH, PJRT_ROOT, XLA_PATH, TPU_LIBRARY_PATH")
    message(STATUS "   System paths:      /usr, /usr/local, /opt")
    message(STATUS "   Google Cloud TPU:  /opt/google-cloud-tpu")
    message(STATUS "The vendored copy should always be present in-tree.")
    message(STATUS "If missing, run: libnd4j/scripts/vendor-pjrt-header.sh")

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
