# install_openvino.cmake — Post-build: describe OpenVINO static libs for linking
#
# After OpenVINO's `cmake --build --target install`, the install dir contains:
#   runtime/include/openvino/  — C++ headers
#   runtime/lib/intel64/       — Static archives (.a or .lib)
#   runtime/lib/cmake/OpenVINO/ — CMake config (OpenVINOConfig.cmake)
#   runtime/3rdparty/          — Bundled dependency libs (TBB, etc.)
#
# This script creates a GNU ld response file containing every installed static
# archive. OpenVINO's static CPU plugin has private dependencies (oneDNN,
# pugixml, XML utilities, snippets, and reference implementations) that are not
# represented by a stable, hand-maintainable archive list. Keeping the archive
# discovery here makes fresh ExternalProject builds and dependency-cache restores
# use the same complete link contract without copying or recombining archives.
#
# Usage:
#   cmake -DINSTALL_DIR=<openvino_install> \
#         [-DRESPONSE_FILE=<path>] -P install_openvino.cmake

if(NOT INSTALL_DIR)
    message(FATAL_ERROR "install_openvino: INSTALL_DIR not set")
endif()

if(NOT RESPONSE_FILE)
    set(RESPONSE_FILE "${INSTALL_DIR}/runtime/lib/intel64/openvino-static-link.rsp")
endif()

file(GLOB_RECURSE _OV_STATIC_LIBS LIST_DIRECTORIES FALSE
    "${INSTALL_DIR}/runtime/*.a")
list(FILTER _OV_STATIC_LIBS EXCLUDE REGEX "/libopenvino_all\\.a$")
list(SORT _OV_STATIC_LIBS)
list(LENGTH _OV_STATIC_LIBS _archive_count)

if(_archive_count EQUAL 0)
    message(FATAL_ERROR
        "install_openvino: no static archives found below ${INSTALL_DIR}/runtime")
endif()

get_filename_component(_response_dir "${RESPONSE_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_response_dir}")
file(WRITE "${RESPONSE_FILE}" "--start-group\n")
foreach(_lib IN LISTS _OV_STATIC_LIBS)
    file(APPEND "${RESPONSE_FILE}" "\"${_lib}\"\n")
endforeach()
file(APPEND "${RESPONSE_FILE}" "--end-group\n")

# OpenVINO installs its bundled oneTBB as shared libraries. Add both supported
# install layouts; GNU ld ignores a missing -L directory and resolves the first
# directory containing the unversioned development symlink.
foreach(_tbb_dir
        "${INSTALL_DIR}/runtime/3rdparty/tbb/lib64"
        "${INSTALL_DIR}/runtime/3rdparty/tbb/lib")
    file(APPEND "${RESPONSE_FILE}" "-L\"${_tbb_dir}\"\n")
endforeach()
file(APPEND "${RESPONSE_FILE}" "-ltbb\n-ltbbmalloc\n")

message(STATUS
    "install_openvino: wrote ${_archive_count} static archives to ${RESPONSE_FILE}")

message(STATUS "install_openvino: post-processing complete")
