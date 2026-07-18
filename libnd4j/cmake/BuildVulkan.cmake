# cmake/BuildVulkan.cmake
# Configures the standalone Vulkan compute native library (nd4jvulkan).
#
# MainBuildFlow.cmake (included before this file) creates the shared NativeOps
# targets from an explicit common-control-plane plus Vulkan-owned source set.
# setup_vulkan() has resolved the loader before this function runs. This file:
#   1. Verifies the required Vulkan device/ABI sources are in the object target.
#   2. Stamps the chip identity (SD_VULKAN=1) on the targets.
#   3. Makes the Vulkan include dir + loader linkage explicit for the chip.
#
# DEFAULT_ENGINE is ENGINE_VULKAN. DSP replay and descriptor/hash-driven eager
# execution both route through Vulkan-owned device implementations. There is no
# CPU execution fallback in this artifact.

function(setup_vulkan_build)
    message(STATUS "=== VULKAN BUILD CONFIGURATION ===")

    if(NOT SD_VULKAN)
        message(STATUS "Vulkan build not enabled (SD_VULKAN=OFF)")
        return()
    endif()

    # -----------------------------------------------------------------
    # 1. Determine existing target names (set by CMakeLists.txt before
    #    MainBuildFlow.cmake was included).
    # -----------------------------------------------------------------
    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    set(MAIN_LIB_NAME   "${SD_LIBRARY_NAME}")

    if(NOT TARGET ${OBJECT_LIB_NAME})
        message(FATAL_ERROR "Expected target '${OBJECT_LIB_NAME}' not found. "
                            "MainBuildFlow.cmake must be included before BuildVulkan.cmake.")
    endif()

    # -----------------------------------------------------------------
    # 2. Verify the explicit Vulkan backend source boundary. These files cover
    #    replay, descriptor/hash eager dispatch, stream/DSP routing, device
    #    allocation, dynamic-shape ownership, and the NativeOps ABI.
    # -----------------------------------------------------------------
    get_target_property(_vulkan_obj_sources ${OBJECT_LIB_NAME} SOURCES)
    set(_vulkan_required_sources
        VulkanDeviceContext.cpp
        VulkanReplayHandle.cpp
        VulkanEagerExecutor.cpp
        VulkanKernelEmitterCatalog.cpp
        VulkanPipelineCache.cpp
        VulkanSegmentRecorder.cpp
        VulkanSpirvDiskCache.cpp
        VulkanExecutionStream.cpp
        VulkanDspDispatch.cpp
        VulkanResourceBinder.cpp
        VulkanMemoryPool.cpp
        NativeDynamicShapePlan.cpp
        legacy/vulkan/NativeOps.cpp
        legacy/vulkan/NativeOps_indexing.cpp
        legacy/vulkan/NativeOps_dsp_plan.cpp
        legacy/vulkan/NativeOps_dsp_runtime.cpp
        legacy/vulkan/NativeOps_dsp_collectives.cpp
        array/vulkan/DataBuffer.cpp
        helpers/vulkan/TadCalculator.cpp)
    foreach(_vulkan_required_source IN LISTS _vulkan_required_sources)
        string(FIND "${_vulkan_obj_sources}" "${_vulkan_required_source}" _vulkan_src_idx)
        if(_vulkan_src_idx EQUAL -1)
            message(FATAL_ERROR
                "Required Vulkan backend source '${_vulkan_required_source}' is missing "
                "from target '${OBJECT_LIB_NAME}'. The explicit SD_VULKAN source "
                "boundary is incomplete.")
        endif()
    endforeach()

    # -----------------------------------------------------------------
    # 3. Compile definitions. DEFAULT_ENGINE / SD_VULKAN are already set at
    #    directory scope by CMakeLists.txt; repeating them as PUBLIC target
    #    definitions matches the Hexagon/TPU chip convention and propagates
    #    the chip identity to anything linking the object library.
    # -----------------------------------------------------------------
    target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC
        SD_VULKAN=1
        HAVE_VULKAN=1
        DEFAULT_ENGINE=samediff::ENGINE_VULKAN
    )
    target_compile_definitions(${MAIN_LIB_NAME} PUBLIC
        SD_VULKAN=1
        HAVE_VULKAN=1
    )

    # -----------------------------------------------------------------
    # 4. Include dir + linkage. setup_vulkan() already did both at directory
    #    scope / on the main target; being explicit here keeps the chip
    #    build correct even if the helper wiring changes. CMake dedupes.
    # -----------------------------------------------------------------
    target_include_directories(${OBJECT_LIB_NAME} SYSTEM PRIVATE "${VULKAN_INCLUDE_DIR}")
    target_link_libraries(${MAIN_LIB_NAME} PUBLIC
        "${VULKAN_LIBRARY}"
        ${CMAKE_DL_LIBS}
    )

    # The backend classifier carries the project-managed compiler runtimes
    # explicitly selected by this native target configuration. Loader-relative
    # lookup keeps independent versioned LLVM/MLIR installations coexistent.
    set(_vulkan_shared_runtimes "")

    if(HAVE_MLIR)
        foreach(_mlir_runtime_target IN ITEMS MLIR MLIRExecutionEngineShared LLVM)
            if(NOT TARGET ${_mlir_runtime_target})
                message(FATAL_ERROR
                    "Vulkan MLIR requires shared runtime target ${_mlir_runtime_target}")
            endif()
            list(APPEND _vulkan_shared_runtimes
                "$<TARGET_FILE:${_mlir_runtime_target}>")
        endforeach()
    endif()

    if(HAVE_TRITON)
        foreach(_triton_runtime_target IN ITEMS triton_mlir_shared triton_llvm_shared)
            if(NOT TARGET ${_triton_runtime_target})
                message(FATAL_ERROR
                    "Vulkan Triton requires shared runtime target ${_triton_runtime_target}")
            endif()
            list(APPEND _vulkan_shared_runtimes
                "$<TARGET_FILE:${_triton_runtime_target}>")
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _vulkan_shared_runtimes)
    if(_vulkan_shared_runtimes AND UNIX)
        if(APPLE)
            set(_vulkan_runtime_rpath "@loader_path")
        else()
            set(_vulkan_runtime_rpath "$ORIGIN")
        endif()
        set_target_properties(${MAIN_LIB_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${_vulkan_runtime_rpath}"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()

    # Always refresh the manifest and build-toolchain metadata. This also clears
    # compiler runtimes left by a previous MLIR/Triton-enabled configuration.
    add_custom_command(TARGET ${MAIN_LIB_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DRUNTIME_LIBRARIES_PIPE=$<JOIN:${_vulkan_shared_runtimes},|>"
            "-DREADELF=${CMAKE_READELF}"
            "-DOTOOL=${CMAKE_OTOOL}"
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DOUTPUT_DIR=$<TARGET_FILE_DIR:${MAIN_LIB_NAME}>"
            -P "${CMAKE_SOURCE_DIR}/cmake/StageSharedRuntime.cmake"
        VERBATIM)

    message(STATUS "Vulkan build configured:")
    message(STATUS "   Library name: ${MAIN_LIB_NAME}")
    message(STATUS "   Vulkan loader: ${VULKAN_LIBRARY}")
    message(STATUS "   MLIR SPIR-V JIT: HAVE_MLIR=${HAVE_MLIR}, MLIR_ENABLE_VULKAN=${MLIR_ENABLE_VULKAN}")
    message(STATUS "=== VULKAN BUILD CONFIGURATION COMPLETE ===")

endfunction()
