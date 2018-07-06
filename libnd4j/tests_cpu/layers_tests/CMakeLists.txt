include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../../include ../../layers ../../include/helpers ../../include/array ../../include/memory ../../include/loops ../../include/graph ../../include/ops ../../include/types ../../include/cnpy ../../blas)
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

# -fsanitize=address
# -fsanitize=leak
if (APPLE)
    set(CMAKE_CXX_FLAGS  " -O0 -g -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -fmax-errors=2 -D__APPLE_OS__=true")
elseif(WIN32)    
    set(CMAKE_CXX_FLAGS  " -g -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -fmax-errors=2")
else()
    if ("${_RELEASE}")
        set(CMAKE_CXX_FLAGS  "-O3 -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -fmax-errors=2")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
            set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -mcpu=native")
        else()
            set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
        endif()
    else()
        set(CMAKE_CXX_FLAGS  " -g -O0 -fPIC -std=c++11 -fassociative-math -funsafe-math-optimizations -fmax-errors=2 -fsanitize=address")
    endif()
endif()

# TODO: get rid of this once problem confirmed solved
#if (APPLE)
#    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#        if ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER 6.0 OR "${CMAKE_C_COMPILER_VERSION}" VERSION_EQUAL 6.0)
#            SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wa,-mavx512f -fmax-errors=1")
#        endif()
#    endif()
#endif()

# tests are always compiled with all ops included
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DLIBND4J_ALL_OPS=true")

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
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
endif()


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
    message(FATAL_ERROR "You need at least GCC 4.9")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("OPENMP NOT FOUND")
endif()


if(APPLE)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    #SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static")
elseif(MSYS)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static")
endif()


if (NOT DEFINED ENV{CLION_IDE})
    message("NOT CLION")
    add_executable(runtests ArrayOptionsTests.cpp GraphStateTests.cpp SingleDimTests.cpp ScalarTests.cpp BackpropTests.cpp RNGTests.cpp ShapeTests.cpp StashTests.cpp 
                   VariableProxyTests.cpp SessionLocalTests.cpp FlatBuffersTests.cpp ConvolutionTests.cpp DeclarableOpsTests1.cpp DeclarableOpsTests2.cpp DeclarableOpsTests3.cpp 
                   DeclarableOpsTests4.cpp GraphTests.cpp HashUtilsTests.cpp NDArrayTests.cpp NDArrayTests2.cpp TadTests.cpp VariableSpaceTests.cpp VariableTests.cpp WorkspaceTests.cpp 
                   JavaInteropTests.cpp MemoryUtilsTests.cpp OpsArena.cpp OpTupleTests.cpp ParityOpsTests.cpp BooleanOpsTests.cpp SwitchTests.cpp ScopeTests.cpp ConditionalTests.cpp 
                   LegacyOpsTests.cpp ContextTests.cpp IndexingTests.cpp ShapeUtilsTests.cpp NDArrayListTests.cpp ListOperationsTests.cpp BitwiseUtilsTests.cpp 
                   SanityTests.cpp PlaygroundTests.cpp BroadcastableOpsTests.cpp GraphExecutionerTests.cpp GraphHolderTests.cpp OpTrackerTests.cpp HelpersTests1.cpp 
                   DeclarableOpsTests5.cpp DeclarableOpsTests6.cpp CnpyTests.cpp BrodcastTests.cpp Reduce3Tests.cpp ReduceTests.cpp ShapeTests2.cpp PairwiseTests.cpp 
                   DeclarableOpsTests7.cpp MmapTests.cpp DeclarableOpsTests8.cpp DeclarableOpsTests9.cpp EmptyTests.cpp TypeCastTests.cpp)
endif ()

if ($ENV{CLION_IDE})
    file(GLOB_RECURSE TEST_SOURCES false ./*.cpp ./*.h)

    # Filter out any source files from */CMakeFiles/* paths. these tend to cause problems such a multiple main definitions.
    set (EXCLUDE_DIR "/CMakeFiles/")
    foreach (TMP_PATH ${TEST_SOURCES})
        string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
        if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
            list (REMOVE_ITEM TEST_SOURCES ${TMP_PATH})
        endif ()
    endforeach(TMP_PATH)

    add_executable(runtests ${TEST_SOURCES})
endif ()

target_link_libraries(runtests ${LIBND4J_NAME}static gtest gtest_main)
