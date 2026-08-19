################################################################################
#
# BuildSDX.cmake — Standalone SDX runtime library target.
#
# CPU and CUDA builds produce a self-contained libsdx_cpu.so or
# libsdx_cuda.so that exports only the sdx* C API (dsp_runtime_c.h), with no
# JVM dependency. Vulkan, TPU, NNAPI, and Hexagon builds reuse their central
# chip library because it is already JVM-free and contains the same SDX C ABI.
# Tensor G3's central NNAPI library also owns its planned ARM DSP replay islands;
# reusing it keeps one canonical binary and avoids a misleading libsdx_cpu.so.
#
# Usage:
#   cmake -DSD_BUILD_SDX_STANDALONE=ON ...
#   cmake --build <build-dir> --target sdx_cpu  # or sdx_cuda for CUDA builds
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

# --- SDX kernel backend options (default: match the current parent build) ---
#
# These are tri-state instead of BOOL options. A release matrix deliberately
# reuses one CMake build tree across classifiers, so a BOOL default cached by
# the base classifier (usually OFF) would otherwise remain OFF when a later
# classifier enables OneDNN, Triton, MLIR, or OpenVINO. AUTO is stable in the
# cache while its effective value is recomputed from HAVE_* on every configure.
function(sdx_resolve_feature_option option_name detected_name description output_name)
    set(${option_name} "AUTO" CACHE STRING
        "${description} (AUTO follows the parent build)")
    set_property(CACHE ${option_name} PROPERTY STRINGS AUTO ON OFF)

    string(TOUPPER "${${option_name}}" _sdx_requested)
    if(_sdx_requested STREQUAL "AUTO")
        if(DEFINED ${detected_name} AND ${detected_name})
            set(_sdx_enabled ON)
        else()
            set(_sdx_enabled OFF)
        endif()
    elseif(_sdx_requested MATCHES "^(ON|TRUE|YES|Y|1)$")
        set(_sdx_enabled ON)
    elseif(_sdx_requested MATCHES "^(OFF|FALSE|NO|N|0)$")
        set(_sdx_enabled OFF)
    else()
        message(FATAL_ERROR
            "${option_name} must be AUTO, ON, or OFF; got '${${option_name}}'")
    endif()
    set(${output_name} "${_sdx_enabled}" PARENT_SCOPE)
endfunction()

sdx_resolve_feature_option(
    SDX_INCLUDE_TRITON HAVE_TRITON "Include Triton JIT compiler in SDX" SDX_ENABLE_TRITON)
sdx_resolve_feature_option(
    SDX_INCLUDE_ONEDNN HAVE_ONEDNN "Include OneDNN CPU primitives in SDX" SDX_ENABLE_ONEDNN)
sdx_resolve_feature_option(
    SDX_INCLUDE_MLIR HAVE_MLIR "Include MLIR JIT compiler in SDX" SDX_ENABLE_MLIR)
sdx_resolve_feature_option(
    SDX_INCLUDE_OPENVINO HAVE_OPENVINO "Include OpenVINO graph backend in SDX" SDX_ENABLE_OPENVINO)

function(build_sdx_library)
    # Mobile chip libraries already expose the SDX C ABI and do not link libjvm.
    # Keep that single canonical binary instead of re-linking the same objects
    # through a second CPU artifact (which would mislabel the Tensor G3 hybrid
    # runtime and omit device loader libraries such as Vulkan or NNAPI).
    if(SD_VULKAN OR SD_TPU OR SD_HEXAGON OR SD_NNAPI_ACCELERATOR_ONLY)
        if(NOT TARGET ${SD_LIBRARY_NAME})
            message(FATAL_ERROR
                "SDX device runtime requires the central ${SD_LIBRARY_NAME} target")
        endif()
        set(SDX_STANDALONE_TARGET ${SD_LIBRARY_NAME} PARENT_SCOPE)
        message(STATUS
            "SDX standalone: reusing canonical device target ${SD_LIBRARY_NAME}")
        return()
    endif()

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
    message(STATUS "  Triton:   ${SDX_ENABLE_TRITON} (requested ${SDX_INCLUDE_TRITON})")
    message(STATUS "  OneDNN:   ${SDX_ENABLE_ONEDNN} (requested ${SDX_INCLUDE_ONEDNN})")
    message(STATUS "  MLIR:     ${SDX_ENABLE_MLIR} (requested ${SDX_INCLUDE_MLIR})")
    message(STATUS "  OpenVINO: ${SDX_ENABLE_OPENVINO} (requested ${SDX_INCLUDE_OPENVINO})")

    # Make the target name available to the parent scope for sdx_runtime_sdk
    set(SDX_STANDALONE_TARGET ${SDX_LIB_NAME} PARENT_SCOPE)
endfunction()


# ---------------------------------------------------------------------------
# Triton linking and runtime delivery shared by CPU and CUDA SDX targets.
# The standalone target consumes object files from the central library, so it
# must carry the same normalized Triton/LLVM/MLIR closure whenever those object
# files were compiled with Triton enabled.
# ---------------------------------------------------------------------------
function(configure_sdx_triton_linking main_target_name)
    if(NOT SDX_ENABLE_TRITON AND NOT SD_ZLUDA)
        return()
    endif()
    set(_sdx_runtime_targets "")
    set(_sdx_shared_runtimes "")
    set(_sdx_runtime_roots "")

    if(SDX_ENABLE_TRITON)
        if(NOT HAVE_TRITON)
            message(FATAL_ERROR
                "SDX requested Triton, but the parent build did not configure it")
        endif()
        if(NOT TARGET triton_interface)
            message(FATAL_ERROR
                "SDX Triton support requires the triton_interface target")
        endif()
        target_link_libraries(${main_target_name} PUBLIC triton_interface)
        foreach(_triton_runtime_target IN ITEMS triton_mlir_shared triton_llvm_shared)
            if(NOT TARGET ${_triton_runtime_target})
                message(FATAL_ERROR
                    "SDX Triton support requires normalized shared runtime target ${_triton_runtime_target}")
            endif()
            list(APPEND _sdx_runtime_targets ${_triton_runtime_target})
            list(APPEND _sdx_shared_runtimes
                "$<TARGET_FILE:${_triton_runtime_target}>")
        endforeach()
    endif()

    if(SD_ZLUDA)
        if(NOT ZLUDA_RUNTIME_LIBRARIES OR
           (NOT WIN32 AND NOT ROCM_HIP_RUNTIME_LIBRARY) OR
           (NOT WIN32 AND NOT ROCM_HSA_RUNTIME_LIBRARY) OR
           (NOT WIN32 AND NOT ROCM_HSAKMT_RUNTIME_LIBRARY))
            message(FATAL_ERROR
                "ZLUDA SDX target requires the platform runtime closure used by nd4jcuda")
        endif()
        list(APPEND _sdx_shared_runtimes ${ZLUDA_RUNTIME_LIBRARIES})
        # Keep SDX on the same classifier-owned CUDART as nd4jcuda. The
        # standalone target is loaded independently, so omitting this seed
        # would let its CUDA registration path fall back to host CUDART.
        if(UNIX AND NOT APPLE AND TARGET CUDA::cudart)
            list(APPEND _sdx_shared_runtimes "$<TARGET_FILE:CUDA::cudart>")
        endif()
        if(UNIX AND NOT APPLE AND TARGET CUDA::nvrtc)
            list(APPEND _sdx_shared_runtimes "$<TARGET_FILE:CUDA::nvrtc>")
        endif()
        if(ROCM_HIP_RUNTIME_LIBRARY)
            list(APPEND _sdx_shared_runtimes "${ROCM_HIP_RUNTIME_LIBRARY}")
        endif()
        if(ROCM_HSA_RUNTIME_LIBRARY)
            list(APPEND _sdx_shared_runtimes "${ROCM_HSA_RUNTIME_LIBRARY}")
        endif()
        if(ROCM_HSAKMT_RUNTIME_LIBRARY)
            list(APPEND _sdx_shared_runtimes "${ROCM_HSAKMT_RUNTIME_LIBRARY}")
        endif()
        if(HAVE_MIOPEN AND MIOPEN_LIBRARY)
            list(APPEND _sdx_shared_runtimes "${MIOPEN_LIBRARY}")
        endif()
        foreach(_sdx_runtime_root IN ITEMS
                "${ZLUDA_RUNTIME_ROOT}" "${ROCM_PATH}" "${ROCM_LIB_DIR}"
                "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_ROOT}")
            if(IS_DIRECTORY "${_sdx_runtime_root}")
                list(APPEND _sdx_runtime_roots "${_sdx_runtime_root}")
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _sdx_shared_runtimes)
    list(REMOVE_DUPLICATES _sdx_runtime_roots)
    set_property(TARGET ${main_target_name} PROPERTY
        SDX_RUNTIME_DEPENDENCY_TARGETS "${_sdx_runtime_targets}")
    list(JOIN _sdx_shared_runtimes "|" _sdx_shared_runtimes_pipe)
    list(JOIN _sdx_runtime_roots "|" _sdx_runtime_roots_pipe)

    if(APPLE)
        set_target_properties(${main_target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@loader_path"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    elseif(UNIX)
        set_target_properties(${main_target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()

    add_custom_command(TARGET ${main_target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DRUNTIME_LIBRARIES_PIPE=${_sdx_shared_runtimes_pipe}"
            "-DRUNTIME_SEARCH_ROOTS_PIPE=${_sdx_runtime_roots_pipe}"
            "-DPRIMARY_RUNTIME=$<TARGET_FILE:${main_target_name}>"
            "-DRUNTIME_POLICY=$<IF:$<BOOL:${SD_ZLUDA}>,zluda-amd,default>"
            "-DREADELF=${CMAKE_READELF}"
            "-DOBJDUMP=${CMAKE_OBJDUMP}"
            "-DOTOOL=${CMAKE_OTOOL}"
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DOUTPUT_DIR=$<TARGET_FILE_DIR:${main_target_name}>"
            "-P${CMAKE_SOURCE_DIR}/cmake/StageSharedRuntime.cmake"
        VERBATIM)
endfunction()


# ---------------------------------------------------------------------------
# CPU linking for SDX — same as configure_cpu_linking() minus JVM_LIBRARY.
# Each kernel backend is gated on its SDX_INCLUDE_* toggle.
# ---------------------------------------------------------------------------
function(configure_sdx_cpu_linking main_target_name)
    target_link_libraries(${main_target_name} PUBLIC
            ${OPENBLAS_LIBRARIES} ${BLAS_LIBRARIES} flatbuffers_interface ${CMAKE_DL_LIBS})

    # The standalone SDX library contains the same Apple MPS implementation
    # objects as the normal CPU target and therefore needs the same frameworks.
    if(HAVE_MPS)
        if(NOT APPLE OR NOT MPS_LIBRARIES)
            message(FATAL_ERROR
                "HAVE_MPS requires resolved Apple MPS frameworks for ${main_target_name}")
        endif()
        target_link_libraries(${main_target_name} PUBLIC ${MPS_LIBRARIES})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MPS=1)
    endif()

    # Android CPU-family SDX builds include the NNAPI system ABI whenever the
    # shared object contains NnapiGraphBackend.  Keep this explicit on the
    # standalone target as well as the normal CPU target; otherwise the object
    # graph compiles successfully but libsdx_cpu.so fails at link time.
    if(HAVE_NNAPI)
        if(NOT ANDROID)
            message(FATAL_ERROR
                "HAVE_NNAPI requires an Android toolchain for ${main_target_name}")
        endif()
        find_library(_sdx_nnapi_library neuralnetworks)
        if(NOT _sdx_nnapi_library)
            message(FATAL_ERROR
                "Android system libneuralnetworks was not found for ${main_target_name}")
        endif()
        target_link_libraries(${main_target_name} PUBLIC "${_sdx_nnapi_library}")
        message(STATUS
            "NNAPI SDX link boundary: ${main_target_name} -> ${_sdx_nnapi_library}")
    endif()

    # OneDNN
    if(SDX_ENABLE_ONEDNN AND HAVE_ONEDNN AND DEFINED ONEDNN)
        target_link_libraries(${main_target_name} PUBLIC ${ONEDNN})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ONEDNN=1)
    endif()

    # ARM Compute (always included when available — no separate toggle)
    if(HAVE_ARMCOMPUTE AND DEFINED ARMCOMPUTE_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${ARMCOMPUTE_LIBRARIES})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ARMCOMPUTE=1)
    endif()

    # MLIR
    if(SDX_ENABLE_MLIR AND HAVE_MLIR AND DEFINED MLIR)
        target_link_libraries(${main_target_name} PUBLIC ${MLIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_MLIR=1)
    endif()

    # Triton
    configure_sdx_triton_linking(${main_target_name})

    # OpenVINO
    if(SDX_ENABLE_OPENVINO AND HAVE_OPENVINO AND TARGET openvino_interface)
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
    if(SD_ZLUDA AND HAVE_ZLUDA)
        configure_zluda_cuda_toolkit_linking(${main_target_name})
    else()
        if(UNIX AND NOT APPLE)
            target_link_options(${main_target_name} PRIVATE "LINKER:--no-as-needed")
        endif()
        target_link_libraries(${main_target_name} PUBLIC
                CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cusparse)
        if(TARGET CUDA::nvrtc)
            target_link_libraries(${main_target_name} PUBLIC CUDA::nvrtc)
        endif()
        if(TARGET CUDA::cuda_driver)
            target_link_libraries(${main_target_name} PUBLIC CUDA::cuda_driver)
        endif()
    endif()

    # The standalone target consumes the same CUDA object library as nd4jcuda.
    # When those objects were compiled with cuDNN helpers, their cuDNN symbols
    # must be resolved here as well; the normal nd4jcuda link is not transitive.
    if(HAVE_CUDNN AND TARGET CUDNN::cudnn)
        target_link_libraries(${main_target_name} PUBLIC CUDNN::cudnn)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    elseif(HAVE_CUDNN AND CUDNN_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${CUDNN_LIBRARIES})
        if(CUDNN_INCLUDE_DIR)
            target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        endif()
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
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
    if(SDX_ENABLE_ONEDNN AND HAVE_ONEDNN AND DEFINED ONEDNN)
        target_link_libraries(${main_target_name} PUBLIC ${ONEDNN})
    endif()

    # Triton
    configure_sdx_triton_linking(${main_target_name})

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
