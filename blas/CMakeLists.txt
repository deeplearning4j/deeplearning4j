if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
endif()

include_directories(../include)
if(APPLE)
    message("Using apple")
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()
if(WIN32)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()
endif()
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(../include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -ftree-vectorize -msse2")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    set(CMAKE_C_COMPILER clang-omp++)
    set(CMAKE_CXX_COMPILER clang-omp++)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -ftree-vectorizer-verbose=2 -fopt-info-vec -fopt-info-vec-missed")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -ffast-math -ftree-vectorize -msse2 -ftree-vectorizer-verbose=2 -fopt-info-vec -fopt-info-vec-missed")

    # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
endif()


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
if(!CUDA_BLAS)
    if(!CPU_BLAS)
        set(CUDA_BLAS FALSE)
        set(CPU_BLAS TRUE)
    endif()
endif()


if(CUDA_BLAS)
    message("Build cublas")
    find_package(CUDA)
    add_definitions(-D__CUDABLAS__=true)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        list(APPEND CUDA_NVCC_FLAGS "-ccbin clang-omp")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set (CMAKE_CXX_FLAGS "")
    endif()

    if (CUDA_FOUND)
        message("CUDA found!")
        set( CUDA_ARCHITECTURE_MINIMUM "3.5" CACHE STRING "Minimum required CUDA compute capability" )
        SET(CUDA_VERBOSE_BUILD ON)
        SET(CUDA_SEPARABLE_COMPILATION OFF)
        set(CUDA_COMPUTE_CAPABILITY "35")
        if (FATBIN)
            list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-G;-g;-lcublas; -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52")
        else()
            list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-G;-g;-lcublas; -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52")
        endif()
        CUDA_ADD_LIBRARY(nd4j SHARED cuda/NativeBlas.cu cuda/NativeOps.cu)
        target_link_libraries(nd4j ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cuda)
    endif(CUDA_FOUND)

elseif(CPU_BLAS)
    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)
    FIND_PACKAGE(BLAS)
    FIND_PACKAGE(LAPACK)
    add_library(nd4j SHARED cpu/NativeBlas.cpp cpu/NativeOps.cpp ../include/nd4jmemset.h ../include/nd4jmalloc.h)
    if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
          find_package(OpenBLAS)
            target_link_libraries(nd4j ${OpenBLAS_LIB})
        else()
        target_link_libraries(nd4j ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        #install(TARGETS mySharedLib DESTINATION /some/full/path)
        install(TARGETS nd4j DESTINATION  $ENV{ND4J_HOME}/nd4j-native-parent/nd4j-native/src/main/resources)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cpu)
    endif()

endif()




