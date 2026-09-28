# copy_tbb_libs.cmake — copy OpenVINO's oneTBB shared runtime into the CMake
# binary directory. OpenVINO's static archives link the shared oneTBB libraries
# (libtbb.so.12, libtbbmalloc.so.2; oneTBB has no static variant), so
# libnd4jcpu.so cannot load without them. MainBuildFlow.cmake passes the copies
# to StageSharedRuntime.cmake, which records them in the CPU shared-runtime
# manifest; JavaCPP then extracts and preloads them beside the backend instead
# of relying on the build tree's absolute RUNPATH.
#
# Called via: cmake -D_OV_TBB_ROOT=<openvino_install>/runtime/3rdparty/tbb
#                   -D_DST_DIR=... [-D_REQUIRED=ON] -P copy_tbb_libs.cmake
#
# oneTBB installs through GNUInstallDirs, so its library directory is lib64 on
# Fedora/RHEL and lib on Debian/Ubuntu. Search both, lib64 first, matching the
# -L order of install_openvino.cmake's link response file. _REQUIRED=ON is set
# after libnd4jcpu has linked against oneTBB: a missing runtime is then a broken
# build, not an OpenVINO build that has not run yet.

if(NOT _OV_TBB_ROOT OR NOT _DST_DIR)
    message(FATAL_ERROR "copy_tbb_libs: _OV_TBB_ROOT and _DST_DIR are required")
endif()

set(_tbb_lib_dir "")
foreach(_tbb_lib_subdir IN ITEMS lib64 lib)
    if(EXISTS "${_OV_TBB_ROOT}/${_tbb_lib_subdir}/libtbb.so")
        set(_tbb_lib_dir "${_OV_TBB_ROOT}/${_tbb_lib_subdir}")
        break()
    endif()
endforeach()

if(_tbb_lib_dir STREQUAL "")
    if(_REQUIRED)
        message(FATAL_ERROR
            "copy_tbb_libs: libtbb.so not found in ${_OV_TBB_ROOT}/lib64 or "
            "${_OV_TBB_ROOT}/lib, but the backend links OpenVINO's oneTBB runtime")
    endif()
    message(STATUS "copy_tbb_libs: oneTBB not found under ${_OV_TBB_ROOT} (OpenVINO may not be built yet)")
    return()
endif()

file(GLOB _tbb_files
    "${_tbb_lib_dir}/libtbb.so*"
    "${_tbb_lib_dir}/libtbbmalloc.so*"
    "${_tbb_lib_dir}/libtbbmalloc_proxy.so*"
)

foreach(_f ${_tbb_files})
    get_filename_component(_fname "${_f}" NAME)
    set(_dst "${_DST_DIR}/${_fname}")
    # Use copy_if_different to avoid unnecessary writes (preserves mtime)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "${_dst}"
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "copy_tbb_libs: failed to copy ${_f} -> ${_dst}")
    endif()
    message(STATUS "  Bundled TBB: ${_fname} -> ${_DST_DIR}/")
endforeach()
