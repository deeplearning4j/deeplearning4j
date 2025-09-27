# PlatformDetection.cmake - Platform and architecture detection

# Ensure SD_CPU is TRUE if neither SD_CUDA nor SD_CPU is set
if(NOT SD_CUDA)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

# Set SD_LIBRARY_NAME Based on Build Type
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()

# Set default engine
if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
endif()

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

# Determine platform type more accurately
set(SD_X86_BUILD false)
set(SD_ARM_BUILD false)

if(SD_ANDROID_BUILD)
    if(ANDROID_ABI MATCHES "x86_64")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86-64")
    elseif(ANDROID_ABI MATCHES "x86")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86")
    elseif(ANDROID_ABI MATCHES "arm64-v8a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "arm64-v8a")
    elseif(ANDROID_ABI MATCHES "armeabi-v7a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "armv7-a")
    endif()
elseif(NOT SD_IOS_BUILD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_X86_BUILD true)
        if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
            set(SD_ARCH "x86-64")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
        set(SD_ARM_BUILD true)
        if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                set(SD_ARCH "armv8-a")
            else()
                set(SD_ARCH "armv7-a")
            endif()
        endif()
    endif()
endif()

if(SD_ARM_BUILD)
    if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
        message(STATUS "Warning: SD_ARCH was not set for this ARM build. Defaulting to 'armv8-a'.")
        set(SD_ARCH "armv8-a")
    endif()

    if(SD_ANDROID)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    endif()
endif()

# Define Compiler Flags for Specific Builds
if(SD_APPLE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
endif()

if(SD_ARM_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ARM_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ARM_BUILD=true")
endif()

if(SD_ANDROID_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ANDROID_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ANDROID_BUILD=true")
endif()

if(SD_IOS_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_IOS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_IOS_BUILD=true")
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")

# Include Directories Based on OS
if(UNIX)
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(APPLE)
    message("Using Apple")
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

# External Include Directories
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include")
endif()

# Initialize job pools for parallel builds
set_property(GLOBAL PROPERTY JOB_POOLS one_jobs=1 two_jobs=2)
