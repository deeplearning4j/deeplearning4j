# Linux x86_64 -> NVIDIA SBSA/AArch64 CUDA cross toolchain.
# The CUDA 13 cross repository installs target headers and libraries under
# /usr/local/cuda/targets/sbsa-linux while Debian supplies the GNU host tools.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-g++)

set(_DL4J_SBSA_CUDA_ROOT /usr/local/cuda/targets/sbsa-linux)
set(CUDAToolkit_ROOT /usr/local/cuda)
set(CUDAToolkit_TARGET_DIR "${_DL4J_SBSA_CUDA_ROOT}")

set(CMAKE_FIND_ROOT_PATH
    /usr/aarch64-linux-gnu
    "${_DL4J_SBSA_CUDA_ROOT}"
    "$ENV{OPENBLAS_PATH}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
