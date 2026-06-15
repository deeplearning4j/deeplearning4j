# JNI Configuration for Lifecycle Tracking
# This file detects and configures JNI support for capturing Java stack traces

option(SD_BUILD_WITH_JAVA "Enable Java stack trace capture via JNI" ON)

if(SD_BUILD_WITH_JAVA)
    message(STATUS "🔍 Detecting JNI for Java stack trace capture...")

    # First try to use JAVA_HOME environment variable
    if(DEFINED ENV{JAVA_HOME})
        set(JAVA_HOME_PATH "$ENV{JAVA_HOME}")
        message(STATUS "   Using JAVA_HOME: ${JAVA_HOME_PATH}")

        # Look for jni.h in JAVA_HOME
        find_path(JNI_INCLUDE_DIR
            NAMES jni.h
            PATHS
                "${JAVA_HOME_PATH}/include"
                "${JAVA_HOME_PATH}/Headers"  # macOS location
            NO_DEFAULT_PATH
        )

        # Look for platform-specific jni_md.h
        if(APPLE)
            find_path(JNI_INCLUDE_DIR_PLATFORM
                NAMES jni_md.h
                PATHS
                    "${JAVA_HOME_PATH}/include/darwin"
                    "${JAVA_HOME_PATH}/Headers"
                NO_DEFAULT_PATH
            )
        elseif(UNIX)
            find_path(JNI_INCLUDE_DIR_PLATFORM
                NAMES jni_md.h
                PATHS
                    "${JAVA_HOME_PATH}/include/linux"
                    "${JAVA_HOME_PATH}/include/freebsd"
                NO_DEFAULT_PATH
            )
        elseif(WIN32)
            find_path(JNI_INCLUDE_DIR_PLATFORM
                NAMES jni_md.h
                PATHS
                    "${JAVA_HOME_PATH}/include/win32"
                NO_DEFAULT_PATH
            )
        endif()
    else()
        message(STATUS "   JAVA_HOME not set, searching system paths...")
        # Fall back to FindJNI module
        find_package(JNI)
        if(JNI_FOUND)
            set(JNI_INCLUDE_DIR ${JNI_INCLUDE_DIRS})
        endif()
    endif()

    # Verify we found JNI headers
    if(JNI_INCLUDE_DIR)
        message(STATUS "   ✅ Found jni.h: ${JNI_INCLUDE_DIR}")

        if(JNI_INCLUDE_DIR_PLATFORM)
            message(STATUS "   ✅ Found jni_md.h: ${JNI_INCLUDE_DIR_PLATFORM}")
            set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIR} ${JNI_INCLUDE_DIR_PLATFORM})
        else()
            set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIR})
        endif()

        # Find the JVM library (required for linking)
        if(DEFINED ENV{JAVA_HOME})
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
                NO_DEFAULT_PATH
            )
        else()
            # Fall back to system search
            find_library(JVM_LIBRARY NAMES jvm)
        endif()

        if(JVM_LIBRARY)
            message(STATUS "   ✅ Found libjvm: ${JVM_LIBRARY}")

            # Add JNI include directories
            include_directories(SYSTEM ${JNI_INCLUDE_DIRS})

            # Define preprocessor flag to enable JNI stack capture
            add_compile_definitions(SD_JNI_AVAILABLE=1)

            # Export JVM_LIBRARY for linking in BuildCPU.cmake
            # Set as both regular variable (for immediate use) and cache variable (for persistence)
            set(JVM_LIBRARY ${JVM_LIBRARY})
            set(JVM_LIBRARY ${JVM_LIBRARY} CACHE INTERNAL "JVM library for JNI support")
            set(SD_JNI_ENABLED TRUE)
            set(SD_JNI_ENABLED TRUE CACHE INTERNAL "JNI support is enabled")

            message(STATUS "   ✅ JNI support enabled for lifecycle tracking")
        else()
            message(WARNING "⚠️  JVM library (libjvm.so) not found - Java stack traces will be disabled")
            message(WARNING "   Found headers but cannot link without library")
            set(SD_BUILD_WITH_JAVA OFF)
        endif()
    else()
        message(WARNING "⚠️  JNI headers not found - Java stack traces will be disabled")
        message(WARNING "   Set JAVA_HOME or install JDK development headers")
        set(SD_BUILD_WITH_JAVA OFF)
    endif()
else()
    message(STATUS "📴 Java stack trace capture disabled (SD_BUILD_WITH_JAVA=OFF)")
endif()
