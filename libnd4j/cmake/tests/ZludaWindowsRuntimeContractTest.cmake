cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED LIBND4J_SOURCE_DIR)
    message(FATAL_ERROR "LIBND4J_SOURCE_DIR is required")
endif()

include("${LIBND4J_SOURCE_DIR}/cmake/ZludaConfiguration.cmake")

set(_fixture "${CMAKE_CURRENT_BINARY_DIR}/zluda-windows-runtime-contract")
file(REMOVE_RECURSE "${_fixture}")
file(MAKE_DIRECTORY "${_fixture}/bin" "${_fixture}/lib")
file(WRITE "${_fixture}/bin/nvcuda.dll" "runtime")
file(WRITE "${_fixture}/bin/nvcudart_hybrid64.dll" "runtime")
file(WRITE "${_fixture}/bin/zluda.exe" "launcher")
file(WRITE "${_fixture}/bin/zluda_redirect.dll" "runtime")
file(WRITE "${_fixture}/lib/nvcuda.lib" "not distributed by official ZLUDA packages")

resolve_zluda_runtime("${_fixture}" TRUE _windows_link _windows_runtime)
if(NOT _windows_runtime STREQUAL "${_fixture}/bin/nvcuda.dll")
    message(FATAL_ERROR "Windows ZLUDA DLL was not resolved: '${_windows_runtime}'")
endif()
if(NOT _windows_link STREQUAL "")
    message(FATAL_ERROR "Windows must use the CUDA SDK import library, got: '${_windows_link}'")
endif()
resolve_zluda_runtime_bundle("${_fixture}" TRUE
    _windows_bundle _windows_bundle_root)
list(LENGTH _windows_bundle _windows_bundle_count)
if(NOT _windows_bundle_count EQUAL 3)
    message(FATAL_ERROR
        "Windows bundle must contain all three runtime DLLs, got: ${_windows_bundle}")
endif()
if(NOT _windows_bundle_root STREQUAL "${_fixture}/bin")
    message(FATAL_ERROR
        "Windows bundle root was not resolved: '${_windows_bundle_root}'")
endif()

file(REMOVE "${_fixture}/bin/zluda_redirect.dll")
resolve_zluda_runtime("${_fixture}" TRUE _incomplete_link _incomplete_runtime)
if(NOT _incomplete_runtime STREQUAL "")
    message(FATAL_ERROR "Incomplete Windows ZLUDA layout must be rejected")
endif()
file(WRITE "${_fixture}/bin/zluda_redirect.dll" "runtime")

file(WRITE "${_fixture}/lib/libcuda.so" "runtime")
file(WRITE "${_fixture}/lib/libnvcuda.so" "runtime")
file(WRITE "${_fixture}/lib/libcublas.so" "runtime")
file(WRITE "${_fixture}/lib/libcublas.so.12" "compatibility alias")
file(WRITE "${_fixture}/lib/libcublaslt.so" "runtime")
file(WRITE "${_fixture}/lib/libcusparse.so" "runtime")
file(WRITE "${_fixture}/lib/libcudnn.so.8" "runtime")
file(WRITE "${_fixture}/lib/libcudnn.so.9" "runtime")
resolve_zluda_runtime("${_fixture}" FALSE _linux_link _linux_runtime)
if(NOT _linux_runtime STREQUAL "${_fixture}/lib/libcuda.so")
    message(FATAL_ERROR "Linux ZLUDA runtime was not resolved: '${_linux_runtime}'")
endif()
if(NOT _linux_link STREQUAL "${_linux_runtime}")
    message(FATAL_ERROR "Linux must link its ZLUDA runtime directly: '${_linux_link}'")
endif()
resolve_zluda_runtime_bundle("${_fixture}" FALSE
    _linux_bundle _linux_bundle_root)
list(LENGTH _linux_bundle _linux_bundle_count)
if(NOT _linux_bundle_count EQUAL 8)
    message(FATAL_ERROR
        "Linux bundle must contain all shared libraries, got: ${_linux_bundle}")
endif()
resolve_zluda_cuda_abi_libraries("${_linux_bundle}"
    _linux_cuda_abi _linux_cudnn_abi)
list(LENGTH _linux_cuda_abi _linux_cuda_abi_count)
if(NOT _linux_cuda_abi_count EQUAL 3)
    message(FATAL_ERROR
        "Linux ZLUDA link set must contain one implementation per CUDA ABI family: ${_linux_cuda_abi}")
endif()
list(LENGTH _linux_cudnn_abi _linux_cudnn_count)
if(NOT _linux_cudnn_count EQUAL 2)
    message(FATAL_ERROR
        "Linux ZLUDA bundle must expose both supported cuDNN ABIs: ${_linux_cudnn_abi}")
endif()
if(NOT _linux_bundle_root STREQUAL "${_fixture}/lib")
    message(FATAL_ERROR
        "Linux bundle root was not resolved: '${_linux_bundle_root}'")
endif()

file(REMOVE_RECURSE "${_fixture}")
