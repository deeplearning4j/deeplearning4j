cmake_minimum_required(VERSION 3.15)

if(NOT DEFINED LIBND4J_SOURCE_DIR)
    message(FATAL_ERROR "LIBND4J_SOURCE_DIR is required")
endif()

include("${LIBND4J_SOURCE_DIR}/cmake/ExternalProjectCompatibility.cmake")

list(LENGTH SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS _timestamp_arg_count)
if(CMAKE_VERSION VERSION_LESS "3.24")
    if(NOT _timestamp_arg_count EQUAL 0)
        message(FATAL_ERROR
            "CMake ${CMAKE_VERSION} must not receive DOWNLOAD_EXTRACT_TIMESTAMP")
    endif()
else()
    if(NOT _timestamp_arg_count EQUAL 2)
        message(FATAL_ERROR
            "CMake ${CMAKE_VERSION} must receive the timestamp keyword and value")
    endif()
    list(GET SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS 0 _timestamp_keyword)
    list(GET SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS 1 _timestamp_value)
    if(NOT _timestamp_keyword STREQUAL "DOWNLOAD_EXTRACT_TIMESTAMP" OR
       NOT _timestamp_value STREQUAL "TRUE")
        message(FATAL_ERROR
            "Unexpected timestamp arguments: '${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}'")
    endif()
endif()

file(READ "${LIBND4J_SOURCE_DIR}/cmake/Dependencies.cmake" _dependencies)
if(_dependencies MATCHES "DOWNLOAD_EXTRACT_TIMESTAMP[ \t]+TRUE")
    message(FATAL_ERROR
        "Dependencies.cmake contains an unconditional CMake 3.24-only option")
endif()
if(NOT _dependencies MATCHES
   "include\\(\"\\\${CMAKE_CURRENT_LIST_DIR}/ExternalProjectCompatibility.cmake\"\\)")
    message(FATAL_ERROR
        "Dependencies.cmake does not include ExternalProjectCompatibility.cmake")
endif()

if(NOT DEFINED TEST_BINARY_DIR)
    set(TEST_BINARY_DIR
        "${LIBND4J_SOURCE_DIR}/blasbuild/external-project-compatibility-contract")
endif()
set(_fixture "${TEST_BINARY_DIR}")
file(REMOVE_RECURSE "${_fixture}")
file(MAKE_DIRECTORY "${_fixture}")

set(MODULE_PATH "${LIBND4J_SOURCE_DIR}/cmake/ExternalProjectCompatibility.cmake")
set(_fixture_cmakelists [=[
cmake_minimum_required(VERSION 3.15)
project(external_project_compatibility_contract NONE)
include(ExternalProject)
include("@MODULE_PATH@")

ExternalProject_Add(contract_dependency
    URL "https://example.invalid/payload.tar.gz"
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads"
    DOWNLOAD_NAME "payload.tar.gz"
    ${SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")
]=])
string(CONFIGURE "${_fixture_cmakelists}" _configured_cmakelists @ONLY)
file(WRITE "${_fixture}/CMakeLists.txt" "${_configured_cmakelists}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${_fixture}" -B "${_fixture}/build"
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_stdout
    ERROR_VARIABLE _configure_stderr)
if(NOT _configure_result EQUAL 0)
    message(FATAL_ERROR
        "ExternalProject fixture configuration failed:\n${_configure_stdout}\n${_configure_stderr}")
endif()

file(GLOB_RECURSE _generated_scripts LIST_DIRECTORIES FALSE
    "${_fixture}/build/*contract_dependency*.cmake")
if(NOT _generated_scripts)
    message(FATAL_ERROR "No ExternalProject scripts were generated")
endif()

set(_saw_download_path FALSE)
foreach(_script IN LISTS _generated_scripts)
    file(READ "${_script}" _script_contents)
    if(_script_contents MATCHES
       "downloads;DOWNLOAD_EXTRACT_TIMESTAMP;TRUE[/\\\\]payload\\.tar\\.gz")
        message(FATAL_ERROR
            "CMake ${CMAKE_VERSION} generated a list-valued download path in ${_script}")
    endif()
    if(_script_contents MATCHES "downloads[/\\\\]payload\\.tar\\.gz")
        set(_saw_download_path TRUE)
    endif()
endforeach()
if(NOT _saw_download_path)
    message(FATAL_ERROR
        "Generated scripts did not contain the expected scalar download path")
endif()

file(REMOVE_RECURSE "${_fixture}")
message(STATUS
    "ExternalProject compatibility contract passed with CMake ${CMAKE_VERSION}")
