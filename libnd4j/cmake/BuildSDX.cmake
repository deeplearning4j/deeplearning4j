################################################################################
#
# BuildSDX.cmake — Standalone SDX runtime library target.
#
# Builds libsdx_cpu.so or libsdx_cuda.so: a self-contained shared library
# that exports only the sdx* C API (dsp_runtime_c.h), with no JVM dependency.
#
# Usage:
#   cmake -DSD_BUILD_SDX_STANDALONE=ON ...
#   make sdx_cpu -j12        # or sdx_cuda for CUDA builds
#
# Each kernel backend can be independently toggled.  All default ON when the
# parent build detected them, so a plain -DSD_BUILD_SDX_STANDALONE=ON produces
# a fully-featured runtime.  Override any to OFF for a smaller deployment:
#
#   -DSDX_INCLUDE_TRITON=OFF     Triton/LLVM JIT compiler
#   -DSDX_INCLUDE_ONEDNN=OFF     OneDNN (MKL-DNN) CPU primitives
#   -DSDX_INCLUDE_MLIR=OFF       MLIR JIT compiler
#   -DSDX_INCLUDE_OPENVINO=OFF   OpenVINO CPU graph backend
#
################################################################################

# --- SDX kernel backend options (default: match parent build) ---
option(SDX_INCLUDE_TRITON   "Include Triton JIT compiler in SDX"   ${HAVE_TRITON})
option(SDX_INCLUDE_ONEDNN   "Include OneDNN CPU primitives in SDX" ${HAVE_ONEDNN})
option(SDX_INCLUDE_MLIR     "Include MLIR JIT compiler in SDX"     ${HAVE_MLIR})
option(SDX_INCLUDE_OPENVINO "Include OpenVINO graph backend in SDX" ${HAVE_OPENVINO})

function(build_sdx_library)
    # --- Determine target name ---
    if(SD_CUDA)
        set(SDX_LIB_NAME "sdx_cuda")
    else()
        set(SDX_LIB_NAME "sdx_cpu")
    endif()

    # --- Re-use object files from the main build ---
    # The main build already compiled ALL_SOURCES into ${SD_LIBRARY_NAME}_object.
    # We re-link those SAME object files into a separate .so with:
    #   - No libjvm dependency
    #   - Symbol visibility restricted to sdx* only (via linker version script)
    # This avoids double compilation entirely — zero additional compile time.
    set(MAIN_OBJECT_LIB "${SD_LIBRARY_NAME}_object")
    if(NOT TARGET ${MAIN_OBJECT_LIB})
        message(FATAL_ERROR "SDX standalone requires the main library target. "
                "Ensure create_and_link_library() runs before build_sdx_library().")
    endif()

    message(STATUS "SDX standalone: re-using objects from ${MAIN_OBJECT_LIB} (no recompilation)")

    # --- Create SHARED library directly from the main object library ---
    # JNI entry points are included in the objects but are made invisible by
    # the linker version script (sdx_exports.lds) which exports only sdx*.
    # JNI functions use JNIEnv* function pointer tables, not direct libjvm calls,
    # so they don't generate unsatisfied external references without libjvm.
    add_library(${SDX_LIB_NAME} SHARED $<TARGET_OBJECTS:${MAIN_OBJECT_LIB}>)
    add_dependencies(${SDX_LIB_NAME} ${MAIN_OBJECT_LIB})
    set_target_properties(${SDX_LIB_NAME} PROPERTIES
        OUTPUT_NAME ${SDX_LIB_NAME}
        POSITION_INDEPENDENT_CODE ON
    )

    # --- Link dependencies (NO JVM) ---
    if(SD_CUDA)
        configure_sdx_cuda_linking(${SDX_LIB_NAME})
    else()
        configure_sdx_cpu_linking(${SDX_LIB_NAME})
    endif()

    # --- Symbol visibility: export only sdx* ---
    if(UNIX AND NOT APPLE)
        set(_sdx_version_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdx_exports.lds")
        if(EXISTS "${_sdx_version_script}")
            target_link_options(${SDX_LIB_NAME} PRIVATE
                -Wl,--version-script=${_sdx_version_script})
            message(STATUS "SDX standalone: using linker version script for symbol visibility")
        endif()
    elseif(APPLE)
        # macOS: use -exported_symbols_list equivalent
        target_link_options(${SDX_LIB_NAME} PRIVATE
            -Wl,-exported_symbol,_sdx*)
    endif()

    message(STATUS "SDX standalone target: ${SDX_LIB_NAME} (no JVM)")
    message(STATUS "  Triton:   ${SDX_INCLUDE_TRITON}")
    message(STATUS "  OneDNN:   ${SDX_INCLUDE_ONEDNN}")
    message(STATUS "  MLIR:     ${SDX_INCLUDE_MLIR}")
    message(STATUS "  OpenVINO: ${SDX_INCLUDE_OPENVINO}")

    # Make the target name available to the parent scope for sdx_runtime_sdk
    set(SDX_STANDALONE_TARGET ${SDX_LIB_NAME} PARENT_SCOPE)
endfunction()


# ---------------------------------------------------------------------------
# CPU linking for SDX — same as configure_cpu_linking() minus JVM_LIBRARY.
# Each kernel backend is gated on its SDX_INCLUDE_* toggle.
# ---------------------------------------------------------------------------
function(configure_sdx_cpu_linking main_target_name)
    target_link_libraries(${main_target_name} PUBLIC
            ${OPENBLAS_LIBRARIES} ${BLAS_LIBRARIES} flatbuffers_interface ${CMAKE_DL_LIBS})

    # OneDNN
    if(SDX_INCLUDE_ONEDNN AND HAVE_ONEDNN AND DEFINED ONEDNN)
        target_link_libraries(${main_target_name} PUBLIC ${ONEDNN})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ONEDNN=1)
    endif()

    # ARM Compute (always included when available — no separate toggle)
    if(HAVE_ARMCOMPUTE AND DEFINED ARMCOMPUTE_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${ARMCOMPUTE_LIBRARIES})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ARMCOMPUTE=1)
    endif()

    # MLIR
    if(SDX_INCLUDE_MLIR AND HAVE_MLIR AND DEFINED MLIR)
        target_link_libraries(${main_target_name} PUBLIC ${MLIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLIR=1)
    endif()

    # Triton
    if(SDX_INCLUDE_TRITON AND HAVE_TRITON AND DEFINED TRITON)
        target_link_libraries(${main_target_name} PUBLIC ${TRITON})
    endif()

    # OpenVINO
    if(SDX_INCLUDE_OPENVINO AND HAVE_OPENVINO AND TARGET openvino_interface)
        target_link_libraries(${main_target_name} PUBLIC openvino_interface)
    endif()

    # MLX (always included when available — Apple-only)
    if(HAVE_MLX AND DEFINED MLX)
        target_link_libraries(${main_target_name} PUBLIC ${MLX})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLX=1)
    endif()

    # NCCL (always included when available)
    if(HAVE_NCCL AND DEFINED NCCL_LIB)
        target_link_libraries(${main_target_name} PUBLIC ${NCCL_LIB})
        target_include_directories(${main_target_name} PUBLIC ${NCCL_INCLUDE_DIRS})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_NCCL=1)
    endif()

    # Dynamic kernel selection
    if(SD_DYNAMIC_KERNEL_SELECTION)
        target_compile_definitions(${main_target_name} PUBLIC SD_DYNAMIC_KERNEL_SELECTION=1)
        target_compile_definitions(${main_target_name} PUBLIC SD_KERNEL_STRATEGY="${SD_KERNEL_STRATEGY}")
    endif()

    # OpenMP
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)
    elseif(NOT APPLE)
        target_link_libraries(${main_target_name} PUBLIC "-fopenmp")
    endif()

    # zlib for SDZ DEFLATE (objects already compiled with HAVE_ZLIB from main build)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        target_link_libraries(${main_target_name} PUBLIC ZLIB::ZLIB)
    endif()

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()


# ---------------------------------------------------------------------------
# CUDA linking for SDX — same as configure_cuda_linking() minus JVM_LIBRARY.
# Each kernel backend is gated on its SDX_INCLUDE_* toggle.
# ---------------------------------------------------------------------------
function(configure_sdx_cuda_linking main_target_name)
    # CUDA toolkit
    target_link_libraries(${main_target_name} PUBLIC
            CUDA::cudart CUDA::cublas CUDA::cusolver)
    if(TARGET CUDA::nvrtc)
        target_link_libraries(${main_target_name} PUBLIC CUDA::nvrtc)
    endif()
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(${main_target_name} PUBLIC CUDA::cuda_driver)
    endif()

    # OpenBLAS / BLAS
    if(DEFINED OPENBLAS_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${OPENBLAS_LIBRARIES})
    endif()
    if(DEFINED BLAS_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${BLAS_LIBRARIES})
    endif()

    target_link_libraries(${main_target_name} PUBLIC flatbuffers_interface ${CMAKE_DL_LIBS})

    # OneDNN
    if(SDX_INCLUDE_ONEDNN AND HAVE_ONEDNN AND DEFINED ONEDNN)
        target_link_libraries(${main_target_name} PUBLIC ${ONEDNN})
    endif()

    # Triton
    if(SDX_INCLUDE_TRITON AND HAVE_TRITON AND DEFINED TRITON)
        target_link_libraries(${main_target_name} PUBLIC ${TRITON})
    endif()

    # NCCL (always included when available)
    if(HAVE_NCCL AND DEFINED NCCL_LIB)
        target_link_libraries(${main_target_name} PUBLIC ${NCCL_LIB})
        target_include_directories(${main_target_name} PUBLIC ${NCCL_INCLUDE_DIRS})
    endif()

    # OpenMP
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # zlib (objects already compiled with HAVE_ZLIB from main build)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        target_link_libraries(${main_target_name} PUBLIC ZLIB::ZLIB)
    endif()

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()
