# copy_tbb_libs.cmake — POST_BUILD script to copy OpenVINO TBB shared libraries
# into the CMake binary directory so the JavaCPP pom.xml bundling step can
# include libtbb.so.12 in the native jar.
#
# Called via: cmake -D_OV_TBB_SRC_DIR=... -D_DST_DIR=... -P copy_tbb_libs.cmake
#
# Without this step, libnd4jcpu.so has an absolute RUNPATH entry pointing to
# the build directory's openvino_install/. After 'mvn clean', that directory is
# deleted and libtbb.so.12 is not found at runtime, causing UnsatisfiedLinkError.

if(NOT _OV_TBB_SRC_DIR OR NOT _DST_DIR)
    message(STATUS "copy_tbb_libs: nothing to do (_OV_TBB_SRC_DIR or _DST_DIR not set)")
    return()
endif()

if(NOT EXISTS "${_OV_TBB_SRC_DIR}")
    message(STATUS "copy_tbb_libs: TBB source directory not found: ${_OV_TBB_SRC_DIR} (OpenVINO may not be built yet)")
    return()
endif()

file(GLOB _tbb_files
    "${_OV_TBB_SRC_DIR}/libtbb.so*"
    "${_OV_TBB_SRC_DIR}/libtbbmalloc.so*"
    "${_OV_TBB_SRC_DIR}/libtbbmalloc_proxy.so*"
)

foreach(_f ${_tbb_files})
    get_filename_component(_fname "${_f}" NAME)
    set(_dst "${_DST_DIR}/${_fname}")
    # Use copy_if_different to avoid unnecessary writes (preserves mtime)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "${_dst}"
        RESULT_VARIABLE _result
    )
    if(_result EQUAL 0)
        message(STATUS "  Bundled TBB: ${_fname} -> ${_DST_DIR}/")
    else()
        message(WARNING "  Failed to copy TBB library: ${_f} -> ${_dst}")
    endif()
endforeach()
