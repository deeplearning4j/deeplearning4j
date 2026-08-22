################################################################################
# TPU/PJRT configuration
#
# The PJRT C API header is a build-time ABI definition. The actual TPU plugin is
# always selected and loaded at runtime by PjrtClientManager; it is never linked
# into libnd4jtpu.
################################################################################

function(setup_pjrt_paths)
    set(_VENDORED_PJRT_INCLUDE "${CMAKE_SOURCE_DIR}/include/external/pjrt")

    if(NOT EXISTS "${_VENDORED_PJRT_INCLUDE}/pjrt_c_api.h")
        message(FATAL_ERROR
            "Pinned PJRT 0.113 C API header is missing: "
            "${_VENDORED_PJRT_INCLUDE}/pjrt_c_api.h")
    endif()
    # PjrtClientManager includes external/pjrt/pjrt_c_api.h directly. Keep one
    # pinned ABI authority instead of accepting an ambient header that may not
    # match the generated bindings or runtime validation.
    set(PJRT_INCLUDE_DIR "${_VENDORED_PJRT_INCLUDE}")

    # Keep this as a cache value because MainBuildFlow and packaging inspect it
    # after Options.cmake initialized all HAVE_* values to OFF.
    set(HAVE_PJRT ON CACHE BOOL "PJRT C API availability" FORCE)
    set(PJRT_FOUND ON CACHE BOOL "PJRT C API availability" FORCE)
    set(PJRT_INCLUDE_DIR "${PJRT_INCLUDE_DIR}" CACHE PATH
        "PJRT C API include directory" FORCE)
    set(PJRT_LIBRARIES "" CACHE STRING
        "PJRT runtime libraries (intentionally empty; plugins are dlopen-only)" FORCE)

    set(HAVE_PJRT ON PARENT_SCOPE)
    set(PJRT_FOUND ON PARENT_SCOPE)
    set(PJRT_INCLUDE_DIR "${PJRT_INCLUDE_DIR}" PARENT_SCOPE)
    set(PJRT_LIBRARIES "" PARENT_SCOPE)

    message(STATUS "PJRT C API: ${PJRT_INCLUDE_DIR}/pjrt_c_api.h")
    message(STATUS "PJRT runtime: dlopen-only (no link-time plugin dependency)")
endfunction()

function(build_tpu_compiler_flags)
    set(_TPU_FLAGS "-fPIC")
    if(SD_GCC_FUNCTRACE)
        string(APPEND _TPU_FLAGS " -g -O0")
    else()
        string(APPEND _TPU_FLAGS " -O3")
    endif()
    set(TPU_COMPILER_FLAGS "${_TPU_FLAGS}" PARENT_SCOPE)
endfunction()

function(debug_tpu_configuration)
    message(STATUS "=== TPU Configuration ===")
    message(STATUS "SD_TPU: ${SD_TPU}")
    message(STATUS "HAVE_PJRT: ${HAVE_PJRT}")
    message(STATUS "PJRT_INCLUDE_DIR: ${PJRT_INCLUDE_DIR}")
    message(STATUS "PJRT runtime linking: disabled (runtime plugin loading)")
endfunction()

function(ensure_tpu_paths_available)
    if(SD_TPU)
        setup_pjrt_paths()
    endif()
endfunction()
