# JNI configuration for native Java integration.

option(SD_BUILD_WITH_JAVA "Enable Java integration via JNI" ON)

if(NOT SD_BUILD_WITH_JAVA)
    message(STATUS "📴 Java integration disabled (SD_BUILD_WITH_JAVA=OFF)")
    set(SD_JNI_ENABLED FALSE CACHE INTERNAL "JNI support is enabled" FORCE)
    return()
endif()

message(STATUS "🔍 Detecting the JDK for JNI integration...")

# JAVA_HOME_PATH is an explicit CMake hint. Otherwise use JAVA_HOME when supplied,
# then derive the JDK root from the Java development tools selected on PATH.
if(NOT JAVA_HOME_PATH)
    if(DEFINED ENV{JAVA_HOME} AND NOT "$ENV{JAVA_HOME}" STREQUAL "")
        set(JAVA_HOME_PATH "$ENV{JAVA_HOME}")
    else()
        find_package(Java REQUIRED COMPONENTS Development)
        get_filename_component(_JAVA_BIN_DIR "${Java_JAVA_EXECUTABLE}" DIRECTORY)
        get_filename_component(JAVA_HOME_PATH "${_JAVA_BIN_DIR}/.." REALPATH)
    endif()
endif()
message(STATUS "   JDK home: ${JAVA_HOME_PATH}")

find_path(JNI_INCLUDE_DIR
    NAMES jni.h
    PATHS
        "${JAVA_HOME_PATH}/include"
        "${JAVA_HOME_PATH}/Headers"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(APPLE)
    find_path(JNI_INCLUDE_DIR_PLATFORM
        NAMES jni_md.h
        PATHS "${JAVA_HOME_PATH}/include/darwin" "${JAVA_HOME_PATH}/Headers"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
    )
elseif(UNIX)
    find_path(JNI_INCLUDE_DIR_PLATFORM
        NAMES jni_md.h
        PATHS "${JAVA_HOME_PATH}/include/linux" "${JAVA_HOME_PATH}/include/freebsd"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
    )
elseif(WIN32)
    find_path(JNI_INCLUDE_DIR_PLATFORM
        NAMES jni_md.h
        PATHS "${JAVA_HOME_PATH}/include/win32"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
    )
endif()

find_library(JVM_LIBRARY
    NAMES jvm
    PATHS
        "${JAVA_HOME_PATH}/lib/server"
        "${JAVA_HOME_PATH}/jre/lib/server"
        "${JAVA_HOME_PATH}/lib"
        "${JAVA_HOME_PATH}/lib/amd64/server"
        "${JAVA_HOME_PATH}/lib/i386/server"
        "${JAVA_HOME_PATH}/jre/lib/amd64/server"
        "${JAVA_HOME_PATH}/jre/lib/i386/server"
        "${JAVA_HOME_PATH}/bin/server"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(NOT JNI_INCLUDE_DIR OR NOT JNI_INCLUDE_DIR_PLATFORM)
    message(FATAL_ERROR
        "SD_BUILD_WITH_JAVA=ON requires jni.h and jni_md.h from a development JDK; "
        "derived JDK home was '${JAVA_HOME_PATH}'")
endif()
if(NOT JVM_LIBRARY)
    message(FATAL_ERROR
        "SD_BUILD_WITH_JAVA=ON requires the JVM shared library under '${JAVA_HOME_PATH}'")
endif()

set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIR} ${JNI_INCLUDE_DIR_PLATFORM})
include_directories(SYSTEM ${JNI_INCLUDE_DIRS})
add_compile_definitions(SD_JNI_AVAILABLE=1)

set(JAVA_HOME_PATH "${JAVA_HOME_PATH}" CACHE PATH "JDK home used for JNI integration" FORCE)
set(JVM_LIBRARY "${JVM_LIBRARY}" CACHE FILEPATH "JVM library for JNI support" FORCE)
set(SD_JNI_ENABLED TRUE CACHE INTERNAL "JNI support is enabled" FORCE)

message(STATUS "   ✅ Found jni.h: ${JNI_INCLUDE_DIR}")
message(STATUS "   ✅ Found jni_md.h: ${JNI_INCLUDE_DIR_PLATFORM}")
message(STATUS "   ✅ Found JVM library: ${JVM_LIBRARY}")
message(STATUS "   ✅ JNI integration enabled")
