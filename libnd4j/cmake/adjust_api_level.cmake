include(${CMAKE_ANDROID_NDK}/build/cmake/abis.cmake)
include(${CMAKE_ANDROID_NDK}/build/cmake/platforms.cmake)

function(adjust_api_level api_level result_name)
  # If no platform version was chosen by the user, default to the minimum
  # version supported by this NDK.
  if(NOT api_level)
    message(STATUS
      "ANDROID_PLATFORM not set. Defaulting to minimum supported version "
      "${NDK_MIN_PLATFORM_LEVEL}.")

    set(api_level "android-${NDK_MIN_PLATFORM_LEVEL}")
  endif()

  if(api_level STREQUAL "latest")
    message(STATUS
      "Using latest available ANDROID_PLATFORM: ${NDK_MAX_PLATFORM_LEVEL}.")
    set(api_level "android-${NDK_MAX_PLATFORM_LEVEL}")
  endif()

  string(REPLACE "android-" "" result ${api_level})

  # Aliases defined by meta/platforms.json include codename aliases for platform
  # API levels as well as cover any gaps in platforms that may not have had NDK
  # APIs.
  if(NOT "${NDK_PLATFORM_ALIAS_${result}}" STREQUAL "")
    message(STATUS
      "${api_level} is an alias for ${NDK_PLATFORM_ALIAS_${result}}. Adjusting "
      "ANDROID_PLATFORM to match.")
    set(api_level "${NDK_PLATFORM_ALIAS_${result}}")
    string(REPLACE "android-" "" result ${api_level})
  endif()

  # Pull up to the minimum supported version if an old API level was requested.
  if(result LESS NDK_MIN_PLATFORM_LEVEL)
    message(STATUS
      "${api_level} is unsupported. Using minimum supported version "
      "${NDK_MIN_PLATFORM_LEVEL}.")
    set(api_level "android-${NDK_MIN_PLATFORM_LEVEL}")
    string(REPLACE "android-" "" result ${api_level})
  endif()

  # Pull up any ABI-specific minimum API levels.
  set(min_for_abi ${NDK_ABI_${ANDROID_ABI}_MIN_OS_VERSION})

  if(result LESS min_for_abi)
    message(STATUS
      "android-${result} is not supported for ${ANDROID_ABI}. Using minimum "
      "supported ${ANDROID_ABI} version ${min_for_abi}.")
    set(api_level android-${min_for_abi})
    set(result ${min_for_abi})
  endif()

  # ANDROID_PLATFORM beyond the maximum is an error. The correct way to specify
  # the latest version is ANDROID_PLATFORM=latest.
  if(result GREATER NDK_MAX_PLATFORM_LEVEL)
    message(SEND_ERROR
      "${api_level} is above the maximum supported version "
      "${NDK_MAX_PLATFORM_LEVEL}. Choose a supported API level or set "
      "ANDROID_PLATFORM to \"latest\".")
  endif()

  set(${result_name} ${result} PARENT_SCOPE)
endfunction()
