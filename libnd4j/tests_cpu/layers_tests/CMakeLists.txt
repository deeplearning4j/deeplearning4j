include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

if (SD_CUDA)
	find_package(CUDA)
	message("Tests CUDA include directory: ${CUDA_INCLUDE_DIRS}")
	include_directories(${CUDA_INCLUDE_DIRS})
	add_definitions(-D__CUDABLAS__=true)

    if(WIN32)
        message("CUDA on Windows: enabling /EHsc")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /FS")
    endif()

    string( TOLOWER "${COMPUTE}" COMPUTE_CMP ) 
    if ("${COMPUTE_CMP}" STREQUAL "all")
        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Common")
    elseif("${COMPUTE_CMP}" STREQUAL "auto")
        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
    elseif(COMPUTE_CMP MATCHES "^[0-9]+$")
        #matches USER COMPUTE old way 
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE} ")
    else()
        #matches numbers NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
        #NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal
        #NUM:   2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 et cetera
        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
    endif()
    # list to spaces
    string (REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")

    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}")

endif()

# -fsanitize=address
# -fsanitize=leak
if (APPLE)
    set(CMAKE_CXX_FLAGS  " -fPIC -D__APPLE_OS__=true")
elseif(WIN32)
    if (SD_CPU)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -msse4 -mavx -mavx2 -O3")
    endif()

	if (SD_CPU AND LINUX)
		set(CMAKE_CXX_FLAGS  " -fPIC")
	endif()
else()
    set(CMAKE_CXX_FLAGS  " -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    IF(${SD_ARCH} MATCHES "arm*")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${SD_ARCH}")
    else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -mcpu=native")
    else()
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif()
    endif()
    if (SD_CPU AND SD_SANITIZE)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    else()
        # CUDA?
    endif()
endif()


# tests are always compiled with all ops included
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DSD_ALL_OPS=true -DBUILD_TESTS=true")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fmax-errors=2")

    if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND NOT(MINGW))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic")
    endif()
endif()

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
    message(FATAL_ERROR "You need at least GCC 4.9")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	find_package(OpenMP)
endif()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("OPENMP NOT FOUND")
endif()

if (SD_CPU)
	file(GLOB_RECURSE TEST_SOURCES false ./*.cpp ./*.h)
elseif (SD_CUDA)
	file(GLOB_RECURSE TEST_SOURCES false ./*.cpp ./*.cu ./*.h)
endif()

# Filter out any source files from */CMakeFiles/* paths. these tend to cause problems such a multiple main definitions.
set (EXCLUDE_DIR "/CMakeFiles/")
foreach (TMP_PATH ${TEST_SOURCES})
    string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
    if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
        list (REMOVE_ITEM TEST_SOURCES ${TMP_PATH})
    endif ()
endforeach(TMP_PATH)

if (SD_CPU)
    if (NOT BLAS_LIBRARIES)
        set(BLAS_LIBRARIES "")
    endif()

	add_executable(runtests ${TEST_SOURCES})
	target_link_libraries(runtests samediff_obj ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES} ${MKLDNN} ${BLAS_LIBRARIES} ${CPU_FEATURES}  ${ARMCOMPUTE_LIBRARIES}  gtest gtest_main)
elseif(SD_CUDA)

	add_executable(runtests ${TEST_SOURCES})

    if (WIN32)
        message("MSVC runtime for tests: ${MSVC_RT_LIB}")
    endif()

    # applies to windows only
    set_property(TARGET runtests PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET gtest PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET gtest_main PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

    if (HAVE_CUDNN)
        message("CUDNN library: ${CUDNN}")
    endif()

	target_link_libraries(runtests samediff_obj ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDNN} ${MKLDNN} gtest gtest_main)
endif()