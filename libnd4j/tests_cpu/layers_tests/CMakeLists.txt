include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()
set(SD_ALL_OPS 1)
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
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        set(CUDA_DEFAULT_ARCHITECTURES 60 61 62 70 72 75 80 CACHE STRING "CUDA architectures for CUDA 11.x based compilers" FORCE)
    else()
        set(CUDA_DEFAULT_ARCHITECTURES 60 61 62 70 72 75 80 86 CACHE STRING "CUDA architectures for nvcc 12 and up" FORCE)
    endif()
    # see: https://stackoverflow.com/questions/2876357/determine-the-line-of-code-that-causes-a-segmentation-fault
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA)
    message("Tests CUDA include directory: ${CUDA_INCLUDE_DIRS}")
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(../../include/array)
    add_definitions(-D__CUDABLAS__=true)
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        add_definitions(-DSD_GCC_FUNCTRACE=1)
    endif()
    if(WIN32)
        message("CUDA on Windows: enabling /EHsc")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /FS")
    endif()

    string( TOLOWER "${COMPUTE}" COMPUTE_CMP )
    if ("${COMPUTE_CMP}" STREQUAL "all")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80")
        else()
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_DEFAULT_ARCHITECTURES}")

    elseif("${COMPUTE_CMP}" STREQUAL "auto")
        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_DEFAULT_ARCHITECTURES}")
    elseif(COMPUTE_CMP MATCHES "^[0-9]+$")
        #matches USER COMPUTE old way
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE} ")
        set(CMAKE_CUDA_ARCHITECTURES "${COMPUTE}")

    else()
        #matches numbers NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
        #NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal
        #NUM:   2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 et cetera
        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
    endif()
    # list to spaces
    string (REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(NOT STREQUAL ${CUDA_ARCH_FLAGS} "")
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS} CACHE STRING "CUDA architectures" FORCE)
        else()
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
                set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 CACHE STRING "CUDA architectures for CUDA 11.x based compilers" FORCE)
                add_compile_definitions(CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
            else()
                set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 CACHE STRING "CUDA architectures for nvcc 12 and up" FORCE)
                add_compile_definitions(CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
            endif()
        endif()
    endif()
    set(CMAKE_CUDA_FLAGS " --ptxas-options=-v  ${CMAKE_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} ${EXPM} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}")

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
elseif(NOT SD_AURORA)
    set(CMAKE_CXX_FLAGS  " -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    IF(${SD_ARCH} MATCHES "arm*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${SD_ARCH}")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
            set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -mcpu=native")
        else()
            set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
        endif()
    endif()

    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        # Set C++ compiler and flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -pthread -MT  -Bsymbolic   -lbfd -rdynamic  -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls  -rdynamic -finstrument-functions -g -O0")
        add_compile_definitions(SD_GCC_FUNCTRACE)
    endif()
    if (SD_SANITIZE)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -ftls-model=local-dynamic  -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow")

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
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${INFORMATIVE_FLAGS}")

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

if (SD_CPU OR SD_AURORA)
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

if (SD_CPU OR SD_AURORA)
    if (NOT BLAS_LIBRARIES)
        set(BLAS_LIBRARIES "")
    endif()

    add_executable(runtests ${TEST_SOURCES})
    target_link_libraries(runtests samediff_obj   ${ONEDNN_LIBRARIES} ${OPENBLAS_LIBRARIES} ${EXTERNAL_DEPENDENCY_LIBS} ${ONEDNN}  ${BLAS_LIBRARIES}  ${ARMCOMPUTE_LIBRARIES}  gtest gtest_main)
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
    # note cuda and cudart are for the driver api access needed for using the driver api for setting things like attributes
    # with the DeviceValidator
    target_link_libraries(runtests samediff_obj  cuda cudart ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDNN} ${EXTERNAL_DEPENDENCY_LIBS} ${ONEDNN} gtest gtest_main)
endif()



list(APPEND GTEST_LIST "ArrayOptionsTests;AttentionTests;BackpropTests;BitwiseUtilsTests;BooleanOpsTests;BroadcastableOpsTests;BroadcastMultiDimTest")
list(APPEND GTEST_LIST "HeaderTest;ConditionalTests;ConstantShapeHelperTests;ConstantTadHelperTests;ConstantHelperTests;ContextTests;TypedConvolutionTests1")
list(APPEND GTEST_LIST "ConvolutionTests1;ConvolutionTests2;TypedConvolutionTests2;CudaLaunchHelperTests;DataBufferTests;DataTypesValidationTests")
list(APPEND GTEST_LIST "DeclarableOpsTests1;DeclarableOpsTests10;DeclarableOpsTests11;DeclarableOpsTests12;DeclarableOpsTests13;TypedDeclarableOpsTests13")
list(APPEND GTEST_LIST "DeclarableOpsTests14;DeclarableOpsTests15;DeclarableOpsTests16;DeclarableOpsTests17;DeclarableOpsTests18;DeclarableOpsTests19")
list(APPEND GTEST_LIST "DeclarableOpsTests2;DeclarableOpsTests3;TypedDeclarableOpsTests4;DeclarableOpsTests4;DeclarableOpsTests5;DeclarableOpsTests6")
list(APPEND GTEST_LIST "DeclarableOpsTests7;TypedDeclarableOpsTests7;DeclarableOpsTests8;TypedDeclarableOpsTests8;DeclarableOpsTests9;EmptyTests")
list(APPEND GTEST_LIST "ExtraArgumentsTests;FlatBuffersTest;FlatUtilsTests;GraphHolderTests;GraphRandomGeneratorTests;GraphStateTests;GraphTests")
list(APPEND GTEST_LIST "HashUtilsTests;HelpersTests1;HelpersTests2;IndexingTests;JavaInteropTests;LegacyOpsTests;ListOperationsTests;LoopCoordsHelper;MemoryUtilsTests")
list(APPEND GTEST_LIST "MmapTests;MultiDataTypeTests;MultiDeviceTests;NDArrayListTests;NDArrayTest;NDArrayTest2;NativeOpsTests;NlpTests;NodeTests;OmpLaunchHelperTests")
list(APPEND GTEST_LIST "OneOffTests;OpTrackerTests;OpTupleTests;EqualsTest;ParityOpsTests;PlaygroundTests;PrimitivesTests;QuantizationTests;RNGTests;ResultSetTests")
list(APPEND GTEST_LIST "SanityTests;ScalarTests;ScopeTests;ShapeTests;LeadingOnes;NormalThreeFourFive;DimensionWarning;FourDTest;SixDTest;TrailingTest;ThreeTest;BeginOneTadTest")
list(APPEND GTEST_LIST "LabelTest;ExpectedValuesTest;OrderTest;ThreeDTest;NumTadTests;TADStall;LengthPerSliceTest;PermuteTest;ElementWiseStrideTest;SliceVectorTest;SliceTensorTest")
list(APPEND GTEST_LIST "SliceMatrixTest;TestConcat;TestReverseCopy;TestRemoveIndex;TensorTwoFromFourDDimTest;TensorTwoDimTest;TensorOneDimTest;MatrixTest;VectorTest;ShapeTest")
list(APPEND GTEST_LIST ";ShapeUtilsTests;SingleDimTests;SortCpuTests;SparseUtilsTest;StashTests;StringTests;SwitchTests;TadTests;ThreadsTests;TypeCastTests;VariableProxyTests")
list(APPEND GTEST_LIST "VariableSpaceTest;VariableTests;WorkspaceTests")

#Generate ctest from gtests entries
foreach(GTEST_ENTRY ${GTEST_LIST})
    add_test(NAME "${GTEST_ENTRY}" COMMAND runtests "--gtest_filter=${GTEST_ENTRY}.*" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests_cpu)
endforeach()
