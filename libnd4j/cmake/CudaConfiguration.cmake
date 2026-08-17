################################################################################
# CUDA Configuration Functions
# Functions for CUDA-specific build configuration and optimization
################################################################################

# CUDA toolkit detection with proper include path setup
function(setup_cuda_toolkit_paths)
    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA toolkit not found. Please install CUDA toolkit or set CUDA_PATH environment variable.")
    endif()

    # Get CUDA include directories
    get_target_property(CUDA_INCLUDE_DIRS CUDA::toolkit INTERFACE_INCLUDE_DIRECTORIES)

    if(NOT CUDA_INCLUDE_DIRS)
        set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()

    # Still not found? Try environment variables and common paths
    if(NOT CUDA_INCLUDE_DIRS)
        set(CUDA_SEARCH_PATHS
                $ENV{CUDA_PATH}
                $ENV{CUDA_HOME}
                $ENV{CUDA_ROOT}
                ${CUDAToolkit_ROOT}
        )

        if(WIN32)
            list(APPEND CUDA_SEARCH_PATHS
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
                    "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
                    "C:/tools/cuda"
            )
        else()
            list(APPEND CUDA_SEARCH_PATHS
                    /usr/local/cuda
                    /opt/cuda
                    /usr/cuda
            )
        endif()

        foreach(search_path ${CUDA_SEARCH_PATHS})
            if(EXISTS "${search_path}/include/cuda.h")
                set(CUDA_INCLUDE_DIRS "${search_path}/include")
                break()
            endif()
        endforeach()
    endif()

    if(NOT CUDA_INCLUDE_DIRS)
        message(FATAL_ERROR "CUDA include directories not found. Please ensure CUDA toolkit is properly installed.")
    endif()

    # Verify cuda.h exists
    set(CUDA_H_FOUND FALSE)
    foreach(include_dir ${CUDA_INCLUDE_DIRS})
        if(EXISTS "${include_dir}/cuda.h")
            set(CUDA_H_FOUND TRUE)
            break()
        endif()
    endforeach()

    if(NOT CUDA_H_FOUND)
        message(FATAL_ERROR "cuda.h not found in CUDA include directories: ${CUDA_INCLUDE_DIRS}")
    endif()

    set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" PARENT_SCOPE)
endfunction()

# cuDNN detection
function(setup_modern_cudnn)
    set(HAVE_CUDNN FALSE PARENT_SCOPE)

    if(NOT (HELPERS_cudnn AND SD_CUDA))
        return()
    endif()

    find_package(CUDAToolkit REQUIRED)

    # Search paths for cuDNN
    set(CUDNN_SEARCH_PATHS
            $ENV{CUDNN_ROOT_DIR}
            $ENV{CUDNN_ROOT}
            $ENV{CUDA_PATH}
            $ENV{CUDA_HOME}
            ${CUDNN_ROOT_DIR}
            ${CUDAToolkit_ROOT}
    )

    if(WIN32)
        list(APPEND CUDNN_SEARCH_PATHS
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
                "C:/tools/cuda"
        )
    else()
        list(APPEND CUDNN_SEARCH_PATHS
                /usr/local/cuda-12.6
                /usr/local/cuda-12.5
                /usr/local/cuda-12.4
                /usr/local/cuda
                /usr/include/cudnn
                /usr/local/include/cudnn
                /opt/cuda
                /opt/cudnn
                /usr
                /usr/local
        )
    endif()

    find_path(CUDNN_INCLUDE_DIR
            NAMES cudnn.h
            HINTS ${CUDNN_SEARCH_PATHS}
            PATHS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES include targets/x86_64-linux/include targets/aarch64-linux/include
            NO_DEFAULT_PATH
    )

    if(NOT CUDNN_INCLUDE_DIR)
        find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h PATHS /usr/include /usr/local/include PATH_SUFFIXES cudnn)
    endif()

    find_library(CUDNN_LIBRARY
            NAMES cudnn libcudnn cudnn8 libcudnn8
            HINTS ${CUDNN_SEARCH_PATHS}
            PATHS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES lib64 lib lib/x64 targets/x86_64-linux/lib
            NO_DEFAULT_PATH
    )

    if(NOT CUDNN_LIBRARY)
        find_library(CUDNN_LIBRARY NAMES cudnn libcudnn PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)
    endif()

    # The CUDA SDK supplies headers while compiling a ZLUDA classifier, but its
    # NVIDIA cuDNN DSO must never enter the runtime link. Select the matching
    # AMD-backed cuDNN ABI implementation from the pinned ZLUDA bundle instead.
    if(SD_ZLUDA AND HAVE_ZLUDA AND NOT WIN32)
        if(NOT CUDNN_INCLUDE_DIR)
            message(FATAL_ERROR
                "ZLUDA cuDNN support requires build-time cuDNN headers")
        endif()
        file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" _zluda_cudnn_header)
        string(REGEX MATCH "define CUDNN_MAJOR[ \t]+([0-9]+)"
            _zluda_cudnn_major_match "${_zluda_cudnn_header}")
        set(_zluda_cudnn_major "${CMAKE_MATCH_1}")
        set(_zluda_selected_cudnn "")
        foreach(_zluda_cudnn_runtime IN LISTS ZLUDA_CUDNN_RUNTIME_LIBRARIES)
            get_filename_component(_zluda_cudnn_name
                "${_zluda_cudnn_runtime}" NAME)
            if(_zluda_cudnn_major AND
               _zluda_cudnn_name MATCHES
                   "^libcudnn\\.so\\.${_zluda_cudnn_major}($|\\.)")
                set(_zluda_selected_cudnn "${_zluda_cudnn_runtime}")
                break()
            elseif(NOT _zluda_selected_cudnn)
                set(_zluda_selected_cudnn "${_zluda_cudnn_runtime}")
            endif()
        endforeach()
        if(NOT _zluda_selected_cudnn)
            message(FATAL_ERROR
                "Pinned ZLUDA bundle has no cuDNN ABI implementation")
        endif()
        set(CUDNN_LIBRARY "${_zluda_selected_cudnn}")
        message(STATUS
            "ZLUDA cuDNN runtime: ${CUDNN_LIBRARY} (CUDA SDK headers only)")
    endif()

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        # Extract version
        if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
            file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_HEADER_CONTENTS)
            string(REGEX MATCH "define CUDNN_MAJOR[ \t]+([0-9]+)" CUDNN_VERSION_MAJOR_MATCH "${CUDNN_HEADER_CONTENTS}")
            string(REGEX MATCH "define CUDNN_MINOR[ \t]+([0-9]+)" CUDNN_VERSION_MINOR_MATCH "${CUDNN_HEADER_CONTENTS}")
            string(REGEX MATCH "define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" CUDNN_VERSION_PATCH_MATCH "${CUDNN_HEADER_CONTENTS}")

            if(CUDNN_VERSION_MAJOR_MATCH)
                string(REGEX REPLACE "define CUDNN_MAJOR[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR_MATCH}")
                string(REGEX REPLACE "define CUDNN_MINOR[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR_MATCH}")
                string(REGEX REPLACE "define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" "\\1" CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH_MATCH}")
                set(CUDNN_VERSION_STRING "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
            else()
                set(CUDNN_VERSION_STRING "Unknown")
            endif()
        else()
            set(CUDNN_VERSION_STRING "Unknown")
        endif()

        if(NOT TARGET CUDNN::cudnn)
            add_library(CUDNN::cudnn UNKNOWN IMPORTED)
            set_target_properties(CUDNN::cudnn PROPERTIES
                    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
            )
        endif()

        message(STATUS "cuDNN: ${CUDNN_VERSION_STRING} (${CUDNN_LIBRARY})")

        set(HAVE_CUDNN TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)
        set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
        set(CUDNN_LIBRARIES "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_LIBRARY "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_VERSION_STRING "${CUDNN_VERSION_STRING}" PARENT_SCOPE)
        return()
    endif()

    # Try pkg-config fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_CUDNN QUIET cudnn)
        if(PC_CUDNN_FOUND)
            set(HAVE_CUDNN TRUE PARENT_SCOPE)
            set(CUDNN_INCLUDE_DIR "${PC_CUDNN_INCLUDE_DIRS}" PARENT_SCOPE)
            set(CUDNN_LIBRARIES "${PC_CUDNN_LIBRARIES}" PARENT_SCOPE)
            set(CUDNN_VERSION_STRING "${PC_CUDNN_VERSION}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Check if cuDNN is embedded in CUDA
    if(CUDAToolkit_FOUND AND CUDAToolkit_INCLUDE_DIRS)
        foreach(cuda_include_dir ${CUDAToolkit_INCLUDE_DIRS})
            if(EXISTS "${cuda_include_dir}/cudnn.h")
                set(HAVE_CUDNN TRUE PARENT_SCOPE)
                set(CUDNN_INCLUDE_DIR "${cuda_include_dir}" PARENT_SCOPE)
                set(CUDNN_LIBRARIES "" PARENT_SCOPE)
                set(CUDNN_VERSION_STRING "Embedded" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    message(STATUS "cuDNN: Not found (disable with -DHELPERS_cudnn=OFF)")
endfunction()

# Link a CUDA SDK archive while tolerating FindCUDAToolkit modules that predate
# the corresponding imported target. CUDA 12 development images can contain the
# archive even when the host CMake does not define (for example)
# CUDA::nvrtc_static. Required archives still fail closed when genuinely absent.
function(link_zluda_cuda_static_library main_target_name imported_target library_name required)
    if(TARGET ${imported_target})
        target_link_libraries(${main_target_name} PUBLIC ${imported_target})
        return()
    endif()

    unset(_zluda_cuda_static_library CACHE)
    find_library(_zluda_cuda_static_library
        NAMES ${library_name}
        HINTS
            "${CUDAToolkit_LIBRARY_DIR}"
            "${CUDAToolkit_LIBRARY_ROOT}"
            "${CUDAToolkit_ROOT}"
            "${CUDA_TOOLKIT_ROOT_DIR}"
        PATH_SUFFIXES
            lib64 lib
            targets/x86_64-linux/lib
            targets/aarch64-linux/lib
        NO_DEFAULT_PATH)
    if(NOT _zluda_cuda_static_library)
        find_library(_zluda_cuda_static_library NAMES ${library_name})
    endif()

    if(_zluda_cuda_static_library)
        message(STATUS
            "Resolved ${imported_target} compatibility archive: ${_zluda_cuda_static_library}")
        target_link_libraries(${main_target_name} PUBLIC
            "${_zluda_cuda_static_library}")
    elseif(required)
        message(FATAL_ERROR
            "ZLUDA build requires ${imported_target} (${library_name}); install the complete CUDA build toolkit")
    endif()
    unset(_zluda_cuda_static_library CACHE)
endfunction()

# Resolve and link a CUDA SDK shared runtime when a ZLUDA classifier cannot
# safely embed the corresponding prebuilt archive. CUDA's NVRTC archive is
# compiled with the small code model; embedding it in the all-ops ZLUDA DSO
# places its read-only tables beyond the R_X86_64_PC32 range. The shared
# implementation is ABI-identical and is staged into the classifier below.
function(link_zluda_cuda_shared_library main_target_name imported_target library_name required)
    if(TARGET ${imported_target})
        target_link_libraries(${main_target_name} PUBLIC ${imported_target})
        return()
    endif()

    unset(_zluda_cuda_shared_library CACHE)
    find_library(_zluda_cuda_shared_library
        NAMES ${library_name} lib${library_name}
        HINTS
            "${CUDAToolkit_LIBRARY_DIR}"
            "${CUDAToolkit_LIBRARY_ROOT}"
            "${CUDAToolkit_ROOT}"
            "${CUDA_TOOLKIT_ROOT_DIR}"
        PATH_SUFFIXES
            lib64 lib
            targets/x86_64-linux/lib
            targets/aarch64-linux/lib
        NO_DEFAULT_PATH)
    if(NOT _zluda_cuda_shared_library)
        find_library(_zluda_cuda_shared_library NAMES ${library_name} lib${library_name})
    endif()

    if(_zluda_cuda_shared_library)
        message(STATUS
            "Resolved ${imported_target} shared compatibility library: ${_zluda_cuda_shared_library}")
        target_link_libraries(${main_target_name} PUBLIC
            "${_zluda_cuda_shared_library}")
    elseif(required)
        message(FATAL_ERROR
            "ZLUDA build requires ${imported_target} (${library_name}); install the CUDA runtime toolkit")
    endif()
    unset(_zluda_cuda_shared_library CACHE)
endfunction()

# Link CUDA implementation pieces that ZLUDA does not implement as CUDA ABI
# replacements into the backend at build time. Linux NVRTC is kept as an
# explicitly packaged shared runtime because NVIDIA's prebuilt static compiler
# archive cannot participate in a >2 GiB all-ops ELF link.
function(configure_zluda_cuda_toolkit_linking main_target_name)
    if(NOT (SD_ZLUDA AND HAVE_ZLUDA))
        message(FATAL_ERROR
            "configure_zluda_cuda_toolkit_linking requires an active ZLUDA build")
    endif()

    configure_zluda_linking(${main_target_name})

    link_zluda_cuda_static_library(
        ${main_target_name} CUDA::cudart_static cudart_static TRUE)
    # ZLUDA does not implement the cuSolver ABI. Solver-backed operations are
    # compiled out for this target, so retaining NVIDIA's static cuSolver/LAPACK
    # archives would make the supposedly AMD-only classifier toolkit-dependent.
    if(UNIX AND NOT APPLE)
        link_zluda_cuda_shared_library(
            ${main_target_name} CUDA::nvrtc nvrtc TRUE)
    else()
        link_zluda_cuda_static_library(
            ${main_target_name} CUDA::nvrtc_static nvrtc_static TRUE)

        # CUDA 12 splits these implementation archives from NVRTC. Older CMake
        # versions/toolkit layouts may expose them transitively instead.
        link_zluda_cuda_static_library(
            ${main_target_name} CUDA::nvrtc_builtins_static nvrtc-builtins_static FALSE)
        link_zluda_cuda_static_library(
            ${main_target_name} CUDA::nvJitLink_static nvJitLink_static FALSE)
        link_zluda_cuda_static_library(
            ${main_target_name} CUDA::nvptxcompiler_static nvptxcompiler_static FALSE)
    endif()

    if(WIN32)
        # Official Windows ZLUDA distributions carry DLLs with the CUDA ABI
        # names. SDK import libraries are used only while linking this build.
        foreach(_zluda_windows_import_target IN ITEMS
                CUDA::cublas CUDA::cublasLt CUDA::cusparse CUDA::cuda_driver)
            if(NOT TARGET ${_zluda_windows_import_target})
                message(FATAL_ERROR
                    "Windows ZLUDA build requires CUDA SDK import target ${_zluda_windows_import_target}")
            endif()
            target_link_libraries(${main_target_name} PUBLIC
                ${_zluda_windows_import_target})
        endforeach()
    endif()
endfunction()

function(configure_cuda_linking main_target_name)
    setup_cuda_toolkit_paths()
    find_package(CUDAToolkit REQUIRED)
    setup_modern_cudnn()

    if(CUDA_INCLUDE_DIRS)
        target_include_directories(${main_target_name} PUBLIC ${CUDA_INCLUDE_DIRS})
    endif()

    # RPATH Configuration
    if(NOT WIN32 AND NOT APPLE)
        set_target_properties(${main_target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH FALSE
            BUILD_RPATH_USE_ORIGIN TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    elseif(APPLE)
        set_target_properties(${main_target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH FALSE
            INSTALL_RPATH_USE_LINK_PATH TRUE
            MACOSX_RPATH TRUE
        )
    endif()

    if(SD_ZLUDA AND HAVE_ZLUDA)
        configure_zluda_cuda_toolkit_linking(${main_target_name})
    else()
        # Normal CUDA classifiers deliberately retain the toolkit's shared
        # runtime dependencies and Bytedeco platform resources.  CUDA helper
        # objects reference cuBLASLt through generated CUDA host objects; GNU
        # ld may otherwise discard libcublasLt under its default --as-needed
        # policy before those references are resolved.
        if(UNIX AND NOT APPLE)
            target_link_options(${main_target_name} PRIVATE "LINKER:--no-as-needed")
        endif()
        target_link_libraries(${main_target_name} PUBLIC CUDA::toolkit CUDA::cudart)
        foreach(_sd_cuda_lib CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cusparse CUDA::nvrtc CUDA::cuda_driver)
            if(TARGET ${_sd_cuda_lib})
                target_link_libraries(${main_target_name} PUBLIC ${_sd_cuda_lib})
            else()
                message(WARNING "CUDA imported target ${_sd_cuda_lib} not found for ${main_target_name}; "
                                "relying on CUDA::toolkit umbrella (verify the toolkit provides it).")
            endif()
        endforeach()
    endif()

    # SD_GCC_FUNCTRACE: Link libdw for stack traces
    if(SD_GCC_FUNCTRACE AND NOT WIN32)
        find_library(LIBDW_LIBRARY NAMES dw)
        find_library(LIBELF_LIBRARY NAMES elf)
        find_path(LIBDW_INCLUDE_DIR NAMES elfutils/libdw.h)

        if(LIBDW_LIBRARY AND LIBELF_LIBRARY)
            target_link_libraries(${main_target_name} PUBLIC ${LIBDW_LIBRARY} ${LIBELF_LIBRARY})
            if(LIBDW_INCLUDE_DIR)
                target_include_directories(${main_target_name} PUBLIC ${LIBDW_INCLUDE_DIR})
            endif()
            target_compile_definitions(${main_target_name} PUBLIC BACKWARD_HAS_DW=1)
        else()
            message(WARNING "SD_GCC_FUNCTRACE enabled but libdw/libelf not found. Install elfutils-devel.")
        endif()
    endif()

    # Link cuDNN if found
    if(HAVE_CUDNN AND TARGET CUDNN::cudnn)
        target_link_libraries(${main_target_name} PUBLIC CUDNN::cudnn)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    elseif(HAVE_CUDNN AND CUDNN_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${CUDNN_LIBRARIES})
        target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    elseif(HAVE_CUDNN AND CUDNN_INCLUDE_DIR)
        target_include_directories(${main_target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=1)
    else()
        target_compile_definitions(${main_target_name} PUBLIC HAVE_CUDNN=0)
    endif()

    target_link_libraries(${main_target_name} PUBLIC flatbuffers_interface)


    # Link OpenBLAS for CUDA builds (needed by BlasHelper.cpp for openblas_set_num_threads)
    if(OPENBLAS_LIBRARIES)
        target_link_libraries(${main_target_name} PUBLIC ${OPENBLAS_LIBRARIES})
        message(STATUS "✅ Linking CUDA build with OpenBLAS: ${OPENBLAS_LIBRARIES}")
    endif()

    # Project-managed shared runtime closure for CUDA-family classifiers.
    # ZLUDA is a runtime distribution, not merely a build-time link input: ship
    # its complete library set plus the ROCm/HIP/MIOpen user-space closure so the
    # resulting JavaCPP classifier does not require ZLUDA_PATH or a ROCm SDK.
    set(_cuda_shared_runtimes "")
    set(_cuda_shared_runtime_roots "")
    if(SD_ZLUDA)
        if(NOT HAVE_ZLUDA OR NOT ZLUDA_RUNTIME_LIBRARIES OR
           NOT IS_DIRECTORY "${ZLUDA_RUNTIME_ROOT}")
            message(FATAL_ERROR
                "ZLUDA classifier requires a complete resolved runtime bundle")
        endif()
        list(APPEND _cuda_shared_runtimes ${ZLUDA_RUNTIME_LIBRARIES})
        list(APPEND _cuda_shared_runtime_roots "${ZLUDA_RUNTIME_ROOT}")

        # Linux ZLUDA links the CUDA NVRTC shared implementation to avoid the
        # prebuilt static archive's small-model relocation overflow. Seed that
        # exact target and its toolkit root so the runtime walker packages the
        # complete CUDA compiler dependency closure.
        if(UNIX AND NOT APPLE AND TARGET CUDA::nvrtc)
            list(APPEND _cuda_shared_runtimes "$<TARGET_FILE:CUDA::nvrtc>")
        endif()
        foreach(_cuda_runtime_root IN ITEMS
                "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_ROOT}")
            if(IS_DIRECTORY "${_cuda_runtime_root}")
                list(APPEND _cuda_shared_runtime_roots "${_cuda_runtime_root}")
            endif()
        endforeach()

        if(UNIX AND NOT APPLE)
            find_program(_cuda_patchelf_executable NAMES patchelf)
            if(NOT _cuda_patchelf_executable)
                message(FATAL_ERROR
                    "Linux ZLUDA classifier packaging requires patchelf so every "
                    "bundled CUDA-ABI and AMD runtime can use a relocatable $ORIGIN RUNPATH")
            endif()
        endif()

        # Linux classifiers bundle the project-pinned ROCm user-space
        # closure. Official Windows ZLUDA packages instead target AMD's Windows
        # HIP SDK: no HIP headers or amdhip64 import/runtime library participate
        # in this CUDA-ABI build, and the release plan deliberately carries no
        # Linux ROCm SDK contract for that shard.
        if(ZLUDA_TARGET_BACKEND STREQUAL "AMD" AND NOT WIN32)
            if(NOT ROCM_HIP_RUNTIME_LIBRARY OR
               NOT EXISTS "${ROCM_HIP_RUNTIME_LIBRARY}")
                message(FATAL_ERROR
                    "AMD ZLUDA classifier requires libamdhip64 from the selected ROCm SDK")
            endif()
            if(NOT ROCM_HSA_RUNTIME_LIBRARY OR
               NOT EXISTS "${ROCM_HSA_RUNTIME_LIBRARY}")
                message(FATAL_ERROR
                    "AMD ZLUDA classifier requires the matching libhsa-runtime64 from the selected ROCm SDK")
            endif()
            if(NOT ROCM_HSAKMT_RUNTIME_LIBRARY OR
               NOT EXISTS "${ROCM_HSAKMT_RUNTIME_LIBRARY}")
                message(FATAL_ERROR
                    "AMD ZLUDA classifier requires the matching libhsakmt from the selected ROCm SDK")
            endif()
            list(APPEND _cuda_shared_runtimes
                "${ROCM_HIP_RUNTIME_LIBRARY}"
                "${ROCM_HSA_RUNTIME_LIBRARY}"
                "${ROCM_HSAKMT_RUNTIME_LIBRARY}")
            if(HAVE_MIOPEN)
                if(NOT MIOPEN_LIBRARY OR NOT EXISTS "${MIOPEN_LIBRARY}")
                    message(FATAL_ERROR
                        "HAVE_MIOPEN is set but its runtime library is unavailable")
                endif()
                list(APPEND _cuda_shared_runtimes "${MIOPEN_LIBRARY}")
            endif()
            foreach(_rocm_runtime_root IN ITEMS "${ROCM_PATH}" "${ROCM_LIB_DIR}")
                if(IS_DIRECTORY "${_rocm_runtime_root}")
                    list(APPEND _cuda_shared_runtime_roots
                        "${_rocm_runtime_root}")
                endif()
            endforeach()
        endif()
    endif()

    # Triton GPU Compiler linking (for CUDA builds)
    if(HAVE_TRITON AND TARGET triton_interface)
        target_link_libraries(${main_target_name} PUBLIC triton_interface)
        # HAVE_TRITON is provided via generated config.h, not as a global -D flag.
        message(STATUS "🔗 Linking Triton GPU compiler backend to ${main_target_name}")

        # The classifier ships the pinned shared LLVM/MLIR runtimes explicitly
        # selected by the native target configuration.
        foreach(_triton_runtime_target IN ITEMS triton_mlir_shared triton_llvm_shared)
            if(NOT TARGET ${_triton_runtime_target})
                message(FATAL_ERROR
                    "Triton requires normalized shared runtime target ${_triton_runtime_target}")
            endif()
            list(APPEND _cuda_shared_runtimes
                "$<TARGET_FILE:${_triton_runtime_target}>")
        endforeach()

    elseif(HAVE_TRITON)
        message(FATAL_ERROR
            "HAVE_TRITON=${HAVE_TRITON}, but the required triton_interface target is missing")
    endif()

    if(APPLE AND (HAVE_TRITON OR SD_ZLUDA))
        set_target_properties(${main_target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@loader_path"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    elseif(UNIX AND (HAVE_TRITON OR SD_ZLUDA))
        set_target_properties(${main_target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()

    list(REMOVE_DUPLICATES _cuda_shared_runtimes)
    list(REMOVE_DUPLICATES _cuda_shared_runtime_roots)
    list(JOIN _cuda_shared_runtimes "|"
        _cuda_shared_runtimes_pipe)
    list(JOIN _cuda_shared_runtime_roots "|"
        _cuda_shared_runtime_roots_pipe)
    set(_cuda_runtime_stage_args
        "-DRUNTIME_LIBRARIES_PIPE=${_cuda_shared_runtimes_pipe}"
        "-DRUNTIME_SEARCH_ROOTS_PIPE=${_cuda_shared_runtime_roots_pipe}")
    if(WIN32)
        # A pipe-delimited value is safe as a CMake argument on POSIX, but it is
        # a cmd.exe metacharacter. Passing the resolved generator-expression
        # values through generated files keeps the post-link command valid on
        # Windows even when paths contain spaces or other shell punctuation.
        set(_cuda_runtime_libraries_file
            "${CMAKE_CURRENT_BINARY_DIR}/${main_target_name}-runtime-libraries.txt")
        set(_cuda_runtime_search_roots_file
            "${CMAKE_CURRENT_BINARY_DIR}/${main_target_name}-runtime-search-roots.txt")
        file(GENERATE OUTPUT "${_cuda_runtime_libraries_file}"
            CONTENT "$<JOIN:${_cuda_shared_runtimes},\n>")
        file(GENERATE OUTPUT "${_cuda_runtime_search_roots_file}"
            CONTENT "$<JOIN:${_cuda_shared_runtime_roots},\n>")
        set(_cuda_runtime_stage_args
            "-DRUNTIME_LIBRARIES_FILE=${_cuda_runtime_libraries_file}"
            "-DRUNTIME_SEARCH_ROOTS_FILE=${_cuda_runtime_search_roots_file}")
    endif()
    set(_cuda_classifier_runtime_dir "")
    if(SD_ZLUDA)
        set(_cuda_classifier_runtime_dir
            "$<TARGET_FILE_DIR:${main_target_name}>/zluda-runtime-package")
    endif()

    # Always refresh the manifest and build-toolchain metadata. This also clears
    # compiler runtimes left by a previous Triton-enabled configuration.
    add_custom_command(TARGET ${main_target_name} POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
            ${_cuda_runtime_stage_args}
            "-DPRIMARY_RUNTIME=$<TARGET_FILE:${main_target_name}>"
            "-DRUNTIME_POLICY=$<IF:$<BOOL:${SD_ZLUDA}>,zluda-amd,default>"
            "-DREADELF=${CMAKE_READELF}"
            "-DOBJDUMP=${CMAKE_OBJDUMP}"
            "-DOTOOL=${CMAKE_OTOOL}"
            "-DPATCHELF_EXECUTABLE=${_cuda_patchelf_executable}"
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DOUTPUT_DIR=$<TARGET_FILE_DIR:${main_target_name}>"
            "-DPACKAGE_DIR=${_cuda_classifier_runtime_dir}"
            "-P${CMAKE_SOURCE_DIR}/cmake/StageSharedRuntime.cmake"
        VERBATIM)

    # JVM library
    if(JVM_LIBRARY)
        target_link_libraries(${main_target_name} PUBLIC ${JVM_LIBRARY})
    endif()

    # OpenMP
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${main_target_name} PUBLIC OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(${main_target_name} PUBLIC "-fopenmp")
    endif()

    # SDZ SameDiff archives are ZIP/DEFLATE by default, so link zlib when available.
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        set(_sdx_object_target "${main_target_name}_object")
        if(TARGET ${_sdx_object_target})
            target_compile_definitions(${_sdx_object_target} PUBLIC HAVE_ZLIB=1)
            target_link_libraries(${_sdx_object_target} PUBLIC ZLIB::ZLIB)
        endif()
        target_link_libraries(${main_target_name} PUBLIC ZLIB::ZLIB)
        target_compile_definitions(${main_target_name} PUBLIC HAVE_ZLIB=1)
        message(STATUS "🔗 Linking zlib for SDZ DEFLATE support")
    else()
        message(WARNING "⚠️ zlib not found - SDZ reader supports STORED ZIP entries only")
    endif()

    install(TARGETS ${main_target_name} DESTINATION .)
endfunction()

function(setup_cuda_architectures_early)
    if(NOT SD_CUDA)
        return()
    endif()

    # Fix missing _CMAKE_CUDA_WHOLE_FLAG
    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        if(WIN32)
            set(_CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE INTERNAL "CUDA whole archive flag")
        else()
            set(_CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE INTERNAL "CUDA whole archive flag")
        endif()
    endif()

    if(NOT DEFINED CMAKE_CUDA_WHOLE_FLAG)
        if(WIN32)
            set(CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE STRING "CUDA whole archive flag")
        else()
            set(CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE STRING "CUDA whole archive flag")
        endif()
    endif()

    # ZLUDA translates PTX at runtime. Disable CMake's implicit real+virtual
    # sm_50 pair so every translation unit embeds one PTX payload rather than
    # PTX plus native NVIDIA SASS. This is also required to keep the all-ops
    # Windows DLL below the PE image-size limit.
    if(SD_ZLUDA)
        set(CUDA_ARCHITECTURES "OFF" PARENT_SCOPE)
        set(CMAKE_CUDA_ARCHITECTURES "OFF" PARENT_SCOPE)
        return()
    endif()

    if(DEFINED COMPUTE)
        string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
        if(COMPUTE_CMP STREQUAL "all" OR COMPUTE_CMP STREQUAL "auto")
            set(CUDA_ARCHITECTURES "86" PARENT_SCOPE)
        else()
            string(REPLACE "," ";" ARCH_LIST "${COMPUTE}")
            set(PARSED_ARCHS "")
            foreach(ARCH ${ARCH_LIST})
                string(REPLACE "." "" ARCH_CLEAN "${ARCH}")
                if(ARCH_CLEAN MATCHES "^[0-9][0-9]$")
                    list(APPEND PARSED_ARCHS "${ARCH_CLEAN}")
                endif()
            endforeach()
            if(PARSED_ARCHS)
                set(CUDA_ARCHITECTURES "${PARSED_ARCHS}" PARENT_SCOPE)
            else()
                set(CUDA_ARCHITECTURES "86" PARENT_SCOPE)
            endif()
        endif()
    else()
        set(CUDA_ARCHITECTURES "86" PARENT_SCOPE)
    endif()
endfunction()

function(setup_cuda_language)
    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        if(WIN32)
            set(_CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE INTERNAL "CUDA whole archive flag")
        else()
            set(_CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE INTERNAL "CUDA whole archive flag")
        endif()
    endif()

    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        find_program(NVCC_EXECUTABLE nvcc)
        if(NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "CUDA compiler not found. Please ensure CUDA toolkit is installed and nvcc is in PATH.")
        endif()
    endif()
endfunction()

function(configure_windows_cuda_build)
    if(NOT WIN32)
        return()
    endif()

    set(CMAKE_VERBOSE_MAKEFILE "${SD_VERBOSE_BUILD}" PARENT_SCOPE)
    set(CMAKE_CUDA_VERBOSE_FLAG "${SD_VERBOSE_BUILD}" PARENT_SCOPE)

    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF PARENT_SCOPE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LINK_OBJECTS OFF PARENT_SCOPE)

    set(CMAKE_CUDA_DEPFILE_FORMAT "" PARENT_SCOPE)
    set(CMAKE_CUDA_DEPENDS_USE_COMPILER OFF PARENT_SCOPE)

    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "" PARENT_SCOPE)

    set(CMAKE_CXX_FLAGS "" PARENT_SCOPE)
    set_property(GLOBAL PROPERTY RULE_MESSAGES ON)
    set_property(GLOBAL PROPERTY TARGET_MESSAGES ON)
endfunction()

# Resolve the PTX images used by ZLUDA without changing ordinary CUDA
# architecture selection. ZLUDA translates virtual PTX at runtime; emitting
# NVIDIA SASS or silently forcing an unrelated legacy target makes the runtime
# choose an invalid payload on modern AMD GPUs.
function(resolve_zluda_ptx_architectures COMPUTE_VALUE)
    if(NOT SD_ZLUDA)
        return()
    endif()

    set(_requested "${COMPUTE_VALUE}")
    string(STRIP "${_requested}" _requested)
    string(TOLOWER "${_requested}" _requested_lower)
    set(_explicit TRUE)
    if(_requested STREQUAL "" OR
       _requested_lower STREQUAL "auto" OR
       _requested_lower STREQUAL "all")
        set(_explicit FALSE)
    endif()

    # An explicit user/build value is authoritative. Only the ZLUDA fallback
    # is ROCm-aware; normal CUDA continues through the existing code below.
    if(NOT _explicit)
        set(_rocm_root "")
        set(_rocm_major "")
        set(_rocm_minor "")
        set(_rocm_candidates
            "${ROCM_PATH}"
            "$ENV{ROCM_PATH}"
            "$ENV{ROCM_HOME}"
            "$ENV{HIP_PATH}"
            "/opt/rocm")
        foreach(_rocm_candidate IN LISTS _rocm_candidates)
            if(NOT _rocm_candidate STREQUAL "" AND
               EXISTS "${_rocm_candidate}/include/hip/hip_runtime.h")
                get_filename_component(_rocm_root "${_rocm_candidate}" REALPATH)
                break()
            endif()
        endforeach()

        # Release workers normally use a versioned ROCm root (for example
        # /opt/rocm-6.2.4). Also accept the canonical ROCm version header when
        # /opt/rocm is an unversioned installation.
        if(_rocm_root)
            get_filename_component(_rocm_root_name "${_rocm_root}" NAME)
            if(_rocm_root_name MATCHES "^rocm[-_]([0-9]+)[.]([0-9]+)")
                set(_rocm_major "${CMAKE_MATCH_1}")
                set(_rocm_minor "${CMAKE_MATCH_2}")
            endif()

            if(_rocm_major STREQUAL "")
                foreach(_rocm_version_header IN ITEMS
                        "${_rocm_root}/include/rocm-core/rocm_version.h"
                        "${_rocm_root}/include/hip/hip_version.h")
                    if(EXISTS "${_rocm_version_header}")
                        file(STRINGS "${_rocm_version_header}" _rocm_version_lines
                             REGEX "ROCM_VERSION_(MAJOR|MINOR)")
                        foreach(_rocm_version_line IN LISTS _rocm_version_lines)
                            if(_rocm_version_line MATCHES
                               "ROCM_VERSION_MAJOR[ \\t]+([0-9]+)")
                                set(_rocm_major "${CMAKE_MATCH_1}")
                            elseif(_rocm_version_line MATCHES
                                   "ROCM_VERSION_MINOR[ \\t]+([0-9]+)")
                                set(_rocm_minor "${CMAKE_MATCH_1}")
                            endif()
                        endforeach()
                    endif()
                endforeach()
            endif()
        endif()

        # ZLUDA v6 is built for ROCm 6/7. Keep ROCm 6 on the conservative
        # Ampere-era PTX baseline, while ROCm 7 gets the project's CUDA 12.x
        # pair. Unknown roots remain usable but are explicit in the configure
        # output and use the conservative baseline.
        if(_rocm_major STREQUAL "6")
            set(_requested "8.6")
        elseif(_rocm_major STREQUAL "7")
            set(_requested "8.6 9.0")
        else()
            set(_requested "8.6")
            if(_rocm_major STREQUAL "")
                message(WARNING
                    "ZLUDA ROCm version could not be detected; defaulting PTX target to compute_86")
            else()
                message(WARNING
                    "ZLUDA ROCm ${_rocm_major}.${_rocm_minor} is outside the validated 6.x/7.x policy; "
                    "defaulting PTX target to compute_86")
            endif()
        endif()
        message(STATUS
            "ZLUDA PTX defaults: ROCm=${_rocm_major}.${_rocm_minor}, targets='${_requested}'")
    else()
        message(STATUS "ZLUDA PTX targets requested by build: '${_requested}'")
    endif()

    # Normalize both comma- and whitespace-separated libnd4j.compute values.
    string(REGEX REPLACE "[ \\t,]+" ";" _arch_list "${_requested}")
    set(_arch_flags "")
    set(_normalized_targets "")
    foreach(_arch IN LISTS _arch_list)
        string(STRIP "${_arch}" _arch)
        if(_arch STREQUAL "")
            continue()
        endif()
        string(REPLACE "." "" _arch_clean "${_arch}")
        if(NOT _arch_clean MATCHES "^[0-9][0-9]$")
            message(FATAL_ERROR
                "Invalid ZLUDA PTX compute target '${_arch}'; use values such as 8.6 or 9.0")
        endif()
        list(APPEND _normalized_targets "${_arch_clean}")
        list(APPEND _arch_flags
            "-gencode arch=compute_${_arch_clean},code=compute_${_arch_clean}")
    endforeach()

    if(NOT _arch_flags)
        message(FATAL_ERROR "ZLUDA requires at least one PTX compute target")
    endif()

    list(REMOVE_DUPLICATES _normalized_targets)
    list(REMOVE_DUPLICATES _arch_flags)
    string(REPLACE ";" " " _arch_flags_string "${_arch_flags}")
    string(REPLACE ";" "," _normalized_targets_string "${_normalized_targets}")
    set(CUDA_ARCH_FLAGS "${_arch_flags_string}" PARENT_SCOPE)
    set(ZLUDA_PTX_TARGETS "${_normalized_targets_string}" PARENT_SCOPE)
    set(CMAKE_CUDA_ARCHITECTURES "OFF" PARENT_SCOPE)
    message(STATUS
        "ZLUDA PTX flags: ${_arch_flags_string} (targets=${_normalized_targets_string})")
endfunction()

function(configure_cuda_architecture_flags COMPUTE)
    if(SD_ZLUDA)
        resolve_zluda_ptx_architectures("${COMPUTE}")
        set(CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}" PARENT_SCOPE)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
        set(ZLUDA_PTX_TARGETS "${ZLUDA_PTX_TARGETS}" PARENT_SCOPE)
        return()
    endif()

    # SD_GCC_FUNCTRACE: sm_86
    if(SD_GCC_FUNCTRACE)
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86" PARENT_SCOPE)
        set(CMAKE_CUDA_ARCHITECTURES "86" PARENT_SCOPE)
        return()
    endif()

    string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
    if(COMPUTE_CMP STREQUAL "all" OR COMPUTE_CMP STREQUAL "auto")
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86" PARENT_SCOPE)
    else()
        string(REPLACE "," ";" ARCH_LIST "${COMPUTE}")
        set(ARCH_FLAGS "")
        foreach(ARCH ${ARCH_LIST})
            string(REPLACE "." "" ARCH_CLEAN "${ARCH}")
            if(ARCH_CLEAN MATCHES "^[0-9][0-9]$")
                set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_${ARCH_CLEAN},code=sm_${ARCH_CLEAN}")
            endif()
        endforeach()
        string(STRIP "${ARCH_FLAGS}" ARCH_FLAGS)
        if(ARCH_FLAGS)
            set(CUDA_ARCH_FLAGS "${ARCH_FLAGS}" PARENT_SCOPE)
        else()
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(build_cuda_compiler_flags CUDA_ARCH_FLAGS)
    set(LOCAL_CUDA_FLAGS "")

    # ZLUDA compatibility
    set(ZLUDA_MODE OFF)
    if(SD_ZLUDA)
        set(ZLUDA_MODE ON)
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --relocatable-device-code=false")
    endif()

    if(WIN32 AND MSVC)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} PARENT_SCOPE)
        set(LOCAL_CUDA_FLAGS "-maxrregcount=128")

        # Verbose flags only when requested
        if(SD_VERBOSE_CUDA)
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --verbose --ptxas-options=-v --resource-usage")
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-showIncludes -Xcompiler=-verbose")
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xlinker=-VERBOSE -Xlinker=-TIME")
        endif()

        # Use hyphen prefix (-std:c++17) not slash (/std:c++17) for all MSVC flags
        # passed via -Xcompiler. sccache wraps nvcc and misparses slash-prefixed flags
        # as drive-relative paths on Windows (e.g. /bigobj → D:\bigobj).
        # MSVC cl.exe accepts both / and - as option prefixes.
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-std:c++17")

        if(WIN32 AND NOT CMAKE_CUDA_HOST_COMPILER)
            find_program(CL_IN_PATH cl.exe)
            if(CL_IN_PATH)
                get_filename_component(CL_DIR ${CL_IN_PATH} DIRECTORY)
                if(CL_DIR MATCHES "x64")
                    set(CMAKE_CUDA_HOST_COMPILER ${CL_IN_PATH})
                else()
                    get_filename_component(CL_PARENT ${CL_DIR} DIRECTORY)
                    if(EXISTS "${CL_PARENT}/x64/cl.exe")
                        set(CMAKE_CUDA_HOST_COMPILER "${CL_PARENT}/x64/cl.exe")
                    endif()
                endif()
            endif()
        endif()

        if(WIN32 AND NOT CMAKE_CUDA_HOST_COMPILER)
            set(COMMON_PATHS
                    "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"
                    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC"
                    "C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"
            )

            foreach(BASE_PATH ${COMMON_PATHS})
                if(EXISTS ${BASE_PATH})
                    file(GLOB MSVC_VERSIONS "${BASE_PATH}/*")
                    if(MSVC_VERSIONS)
                        list(SORT MSVC_VERSIONS)
                        list(REVERSE MSVC_VERSIONS)
                        list(GET MSVC_VERSIONS 0 LATEST_MSVC)
                        set(CANDIDATE_COMPILER "${LATEST_MSVC}/bin/Hostx64/x64/cl.exe")
                        if(EXISTS ${CANDIDATE_COMPILER})
                            set(CMAKE_CUDA_HOST_COMPILER ${CANDIDATE_COMPILER})
                            break()
                        endif()
                    endif()
                endif()
            endforeach()
        endif()

        if(WIN32 AND CMAKE_CUDA_HOST_COMPILER AND NOT EXISTS ${CMAKE_CUDA_HOST_COMPILER})
            message(WARNING "CUDA host compiler path does not exist: ${CMAKE_CUDA_HOST_COMPILER}")
            unset(CMAKE_CUDA_HOST_COMPILER)
        endif()

        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-std:c++17 -Xcompiler=-bigobj -Xcompiler=-EHsc -Xcompiler=-Zc:preprocessor")
        set(CMAKE_CXX_FLAGS "" PARENT_SCOPE)
    else()
        # Unix/Linux
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -maxrregcount=128")

        # Verbose flags only when requested
        if(SD_VERBOSE_CUDA)
            set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --verbose --ptxas-options=-v --resource-usage")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SD_VERBOSE_CUDA)
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-v -Xcompiler=-H -Xlinker=-v -Xlinker=--verbose")
            endif()

            if(SD_GCC_FUNCTRACE OR SD_ZLUDA)
                # Large binary support flags. ZLUDA's all-ops CUDA library can
                # span more than 2 GiB between code and read-only data, which
                # requires the large model. Lifecycle tracing builds retain the
                # established medium model.
                if(SD_ZLUDA)
                    set(DL4J_LARGE_BINARY_CODE_MODEL "large")
                else()
                    set(DL4J_LARGE_BINARY_CODE_MODEL "medium")
                endif()
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fPIC")
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-mcmodel=${DL4J_LARGE_BINARY_CODE_MODEL}")
                if(SD_ZLUDA)
                    # GCC 11 emits 32-bit jump-table references at the project's
                    # -O level even under the large code model. They overflow
                    # once ZLUDA's all-ops image separates text and rodata by
                    # more than 2 GiB.
                    set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fno-jump-tables")
                endif()
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fno-plt")
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fno-asynchronous-unwind-tables")
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fno-omit-frame-pointer")
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-gsplit-dwarf")
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -lineinfo")

                # Linker selection for large binaries. Gold overflows its
                # APLT and .eh_frame relocations on ZLUDA's >2 GiB all-ops
                # image, so that classifier requires LLD. Keep the established
                # Gold-first fallback for lifecycle tracing builds.
                find_program(GOLD_LINKER ld.gold)
                find_program(LLD_LINKER ld.lld)
                find_program(MOLD_LINKER mold)

                set(LINKER_FLAG "")
                set(LINKER_EXTRA_FLAGS "")
                if(SD_ZLUDA)
                    if(NOT LLD_LINKER)
                        message(FATAL_ERROR
                                "Linux ZLUDA builds require LLVM ld.lld for the large all-ops shared library")
                    endif()
                    set(LINKER_FLAG "-fuse-ld=lld")
                    set(LINKER_EXTRA_FLAGS
                        "-Wl,--icf=all -Wl,--gc-sections -Wl,--as-needed -Wl,-z,notext")
                elseif(GOLD_LINKER)
                    set(LINKER_FLAG "-fuse-ld=gold")
                    set(LINKER_EXTRA_FLAGS "-Wl,--icf=safe -Wl,--no-keep-memory -Wl,-z,notext -Wl,--no-relax -Wl,--sort-section=name")
                elseif(LLD_LINKER)
                    set(LINKER_FLAG "-fuse-ld=lld")
                    set(LINKER_EXTRA_FLAGS "-Wl,--icf=all")
                elseif(MOLD_LINKER)
                    set(LINKER_FLAG "-fuse-ld=mold")
                endif()

                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -mcmodel=${DL4J_LARGE_BINARY_CODE_MODEL} -fno-plt -gsplit-dwarf ${LINKER_FLAG}")
                if(SD_ZLUDA)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-jump-tables")
                endif()
                add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wa,-mrelax-relocations=no>)
                add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wa,-mrelax-relocations=no>)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LINKER_FLAG} ${LINKER_EXTRA_FLAGS}")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAG} ${LINKER_EXTRA_FLAGS}")

                # Cache the values for subsequent configures and explicitly return
                # them to setup_cuda_build(). CACHE writes alone do not replace an
                # inherited normal variable inside a CMake function scope.
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ compiler flags" FORCE)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "Shared library linker flags" FORCE)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "Executable linker flags" FORCE)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" PARENT_SCOPE)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" PARENT_SCOPE)

                if(LINKER_FLAG)
                    set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=${LINKER_FLAG}")
                endif()
            else()
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -Xcompiler=-fPIC -Xcompiler=-fpermissive")
            endif()
        endif()
    endif()

    if("${SD_PTXAS}" STREQUAL "ON")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --ptxas-options=-v")
    endif()

    if(SD_KEEP_NVCC_OUTPUT)
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --keep")
    endif()

    if(DEFINED CUDA_ARCH_FLAGS)
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
    endif()
    if(SD_ZLUDA)
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all")
    else()
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all")
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION)
        string(REGEX MATCH "^([0-9]+)" CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
        string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" CUDA_VERSION_MATCH "${CMAKE_CUDA_COMPILER_VERSION}")
        string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${CMAKE_CUDA_COMPILER_VERSION}")
        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")

        # CUDA 11.2+ parallel compilation (not for ZLUDA or functrace)
        if(NOT ZLUDA_MODE AND NOT SD_GCC_FUNCTRACE)
            if(CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND (CUDA_VERSION_MAJOR GREATER 11 OR CUDA_VERSION_MINOR GREATER_EQUAL 2))
                if(DEFINED SD_CUDA_THREADS AND NOT SD_CUDA_THREADS STREQUAL "0")
                    set(NVCC_THREADS ${SD_CUDA_THREADS})
                else()
                    cmake_host_system_information(RESULT NVCC_THREADS QUERY NUMBER_OF_LOGICAL_CORES)
                    cmake_host_system_information(RESULT TOTAL_MEM_MB QUERY TOTAL_PHYSICAL_MEMORY)
                    math(EXPR MEM_BASED_CAP "${TOTAL_MEM_MB} / 4000")
                    if(MEM_BASED_CAP LESS 2)
                        set(MEM_BASED_CAP 2)
                    endif()
                    if(NVCC_THREADS GREATER MEM_BASED_CAP)
                        set(NVCC_THREADS ${MEM_BASED_CAP})
                    endif()
                    if(NVCC_THREADS GREATER 4)
                        set(NVCC_THREADS 4)
                    endif()
                endif()
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --threads ${NVCC_THREADS}")

                if(DEFINED SD_CUDA_SPLIT_COMPILE AND NOT SD_CUDA_SPLIT_COMPILE STREQUAL "0")
                    set(SPLIT_THREADS ${SD_CUDA_SPLIT_COMPILE})
                else()
                    math(EXPR SPLIT_THREADS "${NVCC_THREADS} / 2")
                    if(SPLIT_THREADS LESS 2)
                        set(SPLIT_THREADS 2)
                    endif()
                endif()
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --split-compile ${SPLIT_THREADS}")

                if(SD_CUDA_DEVICE_LTO)
                    set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} -dlto")
                    if(CUDA_VERSION_MAJOR GREATER_EQUAL 12)
                        set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --split-compile-extended ${SPLIT_THREADS}")
                    endif()
                endif()
            endif()
        endif()

        # CUDA 12.8+ time trace
        if(NOT ZLUDA_MODE AND SD_CUDA_TIME_TRACE)
            if(CUDA_VERSION_MAJOR GREATER_EQUAL 13 OR (CUDA_VERSION_MAJOR EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 8))
                set(LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS} --fdevice-time-trace")
            endif()
        endif()
    endif()

    # Clean up Windows flags
    if(WIN32)
        string(REGEX REPLACE "-MD[^a-zA-Z]" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-MT[^a-zA-Z]" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-MF[^a-zA-Z]" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-x cu" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-fpermissive" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-Wno-error" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "/FS[ ]*" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-Xcompiler=-Fd[^,]*,?" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-Xcompiler=," "-Xcompiler=" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
        string(REGEX REPLACE "-Xcompiler=$" "" LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
    endif()

    string(REGEX REPLACE "  +" " " LOCAL_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}")
    string(STRIP "${LOCAL_CUDA_FLAGS}" LOCAL_CUDA_FLAGS)

    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF PARENT_SCOPE)
    set(CMAKE_CUDA_FLAGS "${LOCAL_CUDA_FLAGS}" CACHE STRING "CUDA compiler flags" FORCE)
endfunction()

# Debug configuration (only called when needed)
function(debug_cuda_configuration)
    message(STATUS "=== CUDA Configuration ===")
    message(STATUS "  Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "  Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  Flags: ${CMAKE_CUDA_FLAGS}")
    if(HAVE_CUDNN)
        message(STATUS "  cuDNN: ${CUDNN_VERSION_STRING}")
    endif()
    if(SD_ZLUDA)
        message(STATUS "  ZLUDA: ${SD_ZLUDA_TARGET}")
    endif()
    message(STATUS "==========================")
endfunction()

function(setup_cuda_include_directories)
    setup_cuda_toolkit_paths()

    if(CUDA_INCLUDE_DIRS)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()

    if(CUDAToolkit_INCLUDE_DIRS)
        set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
    endif()
endfunction()

function(setup_cuda_build)
    # TMPDIR: Use build directory instead of /tmp
    set(CUDA_TMPDIR "${CMAKE_BINARY_DIR}/nvcc_tmp")
    file(MAKE_DIRECTORY "${CUDA_TMPDIR}")
    set(ENV{TMPDIR} "${CUDA_TMPDIR}")
    set(ENV{TMP} "${CUDA_TMPDIR}")
    set(ENV{TEMP} "${CUDA_TMPDIR}")

    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        if(WIN32)
            set(_CMAKE_CUDA_WHOLE_FLAG "/WHOLEARCHIVE:" CACHE INTERNAL "CUDA whole archive flag")
        else()
            set(_CMAKE_CUDA_WHOLE_FLAG "-Wl,--whole-archive" CACHE INTERNAL "CUDA whole archive flag")
        endif()
    endif()

    if(SD_GCC_FUNCTRACE)
        add_compile_definitions(SD_GCC_FUNCTRACE=ON)
    endif()

    setup_cuda_include_directories()

    if(NOT DEFINED COMPUTE)
        set(COMPUTE "auto")
    endif()
    configure_cuda_architecture_flags("${COMPUTE}")
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    setup_cuda_language()

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found after enabling CUDA language")
    endif()

    configure_windows_cuda_build()
    build_cuda_compiler_flags("${CUDA_ARCH_FLAGS}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)

    # Print summary
    message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER_VERSION}, arch=${CMAKE_CUDA_ARCHITECTURES}")
    if(SD_ZLUDA)
        message(STATUS "ZLUDA: ${SD_ZLUDA_TARGET}")
    endif()

    add_compile_definitions(SD_CUDA=true)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA" PARENT_SCOPE)
endfunction()

function(ensure_cuda_paths_available)
    if(NOT SD_CUDA)
        return()
    endif()

    find_package(CUDAToolkit REQUIRED)
    setup_cuda_toolkit_paths()

    set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}" PARENT_SCOPE)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" PARENT_SCOPE)
endfunction()

# Legacy function
function(setup_cudnn)
    setup_modern_cudnn()

    set(HAVE_CUDNN ${HAVE_CUDNN} PARENT_SCOPE)
    if(HAVE_CUDNN)
        set(CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR} PARENT_SCOPE)
        set(CUDNN ${CUDNN_LIBRARIES} PARENT_SCOPE)
    endif()
endfunction()
