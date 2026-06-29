# cmake/GenerateBuildStamp.cmake
# Invoked as a CMake -P script by the generate_build_stamp custom target every
# build.  Writes ${OUT_FILE} with a timestamp-based LIBND4J_BUILD_STAMP string
# that changes on each rebuild so the Java DSP plan disk cache (DspPlanDiskCache)
# can detect stale cached plans compiled by an older .so.
#
# Called with:
#   cmake -DOUT_FILE=<path/to/build_stamp.h> -P GenerateBuildStamp.cmake

cmake_minimum_required(VERSION 3.15)

if(NOT DEFINED OUT_FILE OR OUT_FILE STREQUAL "")
    message(FATAL_ERROR "GenerateBuildStamp.cmake: OUT_FILE must be set via -DOUT_FILE=...")
endif()

# Timestamp: ISO-8601 UTC, e.g. "2026-06-26T14:32:05Z"
string(TIMESTAMP _ts UTC)

# Always write — this file is intentionally regenerated on every build so that
# build_info.cpp's translation unit is recompiled with a fresh timestamp.
file(WRITE "${OUT_FILE}"
    "/* Auto-generated every build by cmake/GenerateBuildStamp.cmake -- do NOT edit. */\n"
    "#ifndef LIBND4J_BUILD_STAMP_H\n"
    "#define LIBND4J_BUILD_STAMP_H\n"
    "#define LIBND4J_BUILD_STAMP \"${_ts}\"\n"
    "#endif /* LIBND4J_BUILD_STAMP_H */\n"
)
