# cmake/VulkanConfiguration.cmake
# Configuration checks for the standalone Vulkan chip build (SD_VULKAN,
# libnd4jvulkan).
#
# Dependency discovery and bootstrap live in Dependencies.cmake's
# setup_vulkan(): it locates vulkan/vulkan.h (auto-downloading the Khronos
# Vulkan-Headers release when absent, dep-cache aware) and the Vulkan loader
# (libvulkan.so / libvulkan.so.1 / vulkan-1.dll), verifies the Linux ELF
# class and target machine, and sets HAVE_VULKAN / VULKAN_INCLUDE_DIR / VULKAN_LIBRARY.
# MainBuildFlow.cmake calls setup_vulkan() only for SD_VULKAN. By the time
# this file is included, the standalone backend has completed discovery.
# CPU and CUDA builds never probe, bootstrap, compile, or link Vulkan here.
# A libnd4jvulkan.so without Vulkan support is therefore a hard error.

function(verify_vulkan_chip_requirements)
    message(STATUS "=== VULKAN CHIP REQUIREMENTS ===")

    if(NOT LIBND4J_ENABLE_VULKAN)
        message(FATAL_ERROR
            "SD_VULKAN chip build requested but LIBND4J_ENABLE_VULKAN=OFF. "
            "Remove -DLIBND4J_ENABLE_VULKAN=OFF — the Vulkan chip build "
            "requires the Vulkan compute backend.")
    endif()

    if(ANDROID)
        set(_vulkan_android_api "${SD_ANDROID_TARGET_API_LEVEL}")
        if(NOT _vulkan_android_api MATCHES "^[0-9]+$")
            message(FATAL_ERROR
                "Android Vulkan requires a numeric target API; configured value is '${_vulkan_android_api}'.")
        endif()
        if(_vulkan_android_api LESS 24)
            message(FATAL_ERROR
                "Android Vulkan requires API level 24 or newer; configured API is ${_vulkan_android_api}.")
        endif()
    endif()

    # MLIR is an optional development-time SPIR-V compiler. Production and
    # mobile builds load AOT SPIR-V from the SDX bundle and must not ship the
    # compiler toolchain. Vulkan discovery remains mandatory below.
    if(HAVE_MLIR AND MLIR_ENABLE_VULKAN)
        set(_vulkan_compiler_mode "MLIR SPIR-V JIT available (development mode)")
    else()
        set(_vulkan_compiler_mode "AOT SPIR-V only (runtime JIT unavailable)")
    endif()

    if(NOT HAVE_VULKAN)
        message(FATAL_ERROR
            "SD_VULKAN chip build requested but Vulkan was not found.\n"
            "The Vulkan headers bootstrap automatically, so this almost "
            "always means the Vulkan loader library is missing.\n"
            "Install it with your package manager:\n"
            "  Fedora/RHEL:   dnf install vulkan-loader vulkan-loader-devel\n"
            "  Debian/Ubuntu: apt-get install libvulkan-dev\n"
            "  Arch:          pacman -S vulkan-icd-loader\n"
            "  Windows/macOS: install the LunarG Vulkan SDK (VULKAN_SDK env)\n"
            "Then re-run the build.")
    endif()

    if(NOT DEFINED VULKAN_LIBRARY OR NOT EXISTS "${VULKAN_LIBRARY}")
        message(FATAL_ERROR
            "HAVE_VULKAN is set but VULKAN_LIBRARY ('${VULKAN_LIBRARY}') "
            "does not exist — stale CMake cache? Delete the configured build directory "
            "('${CMAKE_BINARY_DIR}') and reconfigure.")
    endif()

    if(NOT DEFINED VULKAN_INCLUDE_DIR OR NOT EXISTS "${VULKAN_INCLUDE_DIR}/vulkan/vulkan.h")
        message(FATAL_ERROR
            "HAVE_VULKAN is set but vulkan/vulkan.h was not found under "
            "VULKAN_INCLUDE_DIR ('${VULKAN_INCLUDE_DIR}') — stale CMake "
            "cache? Delete the configured build directory ('${CMAKE_BINARY_DIR}') "
            "and reconfigure.")
    endif()

    message(STATUS "  Vulkan headers: ${VULKAN_INCLUDE_DIR}")
    message(STATUS "  Vulkan loader:  ${VULKAN_LIBRARY}")
    message(STATUS "  Replay:         native Vulkan command buffers (Triton compiler not required)")
    message(STATUS "  Compiler:       ${_vulkan_compiler_mode}")
    message(STATUS "=== VULKAN CHIP REQUIREMENTS OK ===")
endfunction()

function(debug_vulkan_configuration)
    message(STATUS "=== Vulkan Configuration ===")
    message(STATUS "  HAVE_VULKAN:        ${HAVE_VULKAN}")
    message(STATUS "  VULKAN_INCLUDE_DIR: ${VULKAN_INCLUDE_DIR}")
    message(STATUS "  VULKAN_LIBRARY:     ${VULKAN_LIBRARY}")
    message(STATUS "  MLIR_ENABLE_VULKAN: ${MLIR_ENABLE_VULKAN}")
    message(STATUS "============================")
endfunction()
