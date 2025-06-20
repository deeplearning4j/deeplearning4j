# BasicSetup.cmake - Basic CMake configuration and initial setup

# Basic CMake Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
include(CheckCXXCompilerFlag)

function(setup_build_configuration)
    # This command takes the template file 'config.h.in' and creates 'config.h'
    # in the binary directory, substituting any @VAR@ or #cmakedefine variables.
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
            "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"
    )

    # Add the directory containing the generated config.h to the include path.
    # This ensures that the compiler can find it with #include <config.h>.
    include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

    # Also include the main source include directory.
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
endfunction()


# Set Windows specific flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

# Helper function for colored status messages
function(print_status_colored type message)
    if(type STREQUAL "ERROR")
        message(FATAL_ERROR "❌ ${message}")
    elseif(type STREQUAL "WARNING")
        message(WARNING "⚠️  ${message}")
    elseif(type STREQUAL "SUCCESS")
        message(STATUS "✅ ${message}")
    elseif(type STREQUAL "INFO")
        message(STATUS "ℹ️  ${message}")
    elseif(type STREQUAL "DEBUG")
        message(STATUS "🔍 ${message}")
    elseif(type STREQUAL "NOTICE")
        message(NOTICE "📢 ${message}")
    else()
        message(STATUS "${message}")
    endif()
endfunction()

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
