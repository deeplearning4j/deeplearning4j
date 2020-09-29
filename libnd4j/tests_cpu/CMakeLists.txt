cmake_minimum_required(VERSION 3.15)
project(tests_cpu)

if (SD_AURORA_BUILD)
  set(GTEST_CFG "CMakeLists.txt.gtest_aurora.in")
else()
  set(GTEST_CFG "CMakeLists.txt.gtest.in")
endif()


# Download and unpack googletest at configure time
configure_file(${GTEST_CFG} googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

set(gtest_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest-src)
#add_subdirectory(libnd4j_tests)
add_subdirectory(layers_tests)
