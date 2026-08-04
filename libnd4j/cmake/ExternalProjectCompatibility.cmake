# Compatibility arguments shared by every libnd4j ExternalProject.
#
# DOWNLOAD_EXTRACT_TIMESTAMP was added in CMake 3.24. Passing it to an older
# ExternalProject parser appends the unknown tokens to the preceding option
# (for example DOWNLOAD_DIR), producing invalid semicolon-delimited paths.
set(SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    list(APPEND SD_EXTERNAL_PROJECT_DOWNLOAD_TIMESTAMP_ARGS
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()
