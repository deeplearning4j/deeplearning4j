# install_openvino.cmake — Post-build: archive OpenVINO static libs for linking
#
# After OpenVINO's `cmake --build --target install`, the install dir contains:
#   runtime/include/openvino/  — C++ headers
#   runtime/lib/intel64/       — Static archives (.a or .lib)
#   runtime/lib/cmake/OpenVINO/ — CMake config (OpenVINOConfig.cmake)
#   runtime/3rdparty/          — Bundled dependency libs (TBB, etc.)
#
# This script creates a combined static library (libopenvino_all.a) that bundles
# ALL static archives together, simplifying the link command in libnd4j.
# This is optional — the main build can also link individual .a files.
#
# Usage:
#   cmake -DINSTALL_DIR=<openvino_install> -P install_openvino.cmake

if(NOT INSTALL_DIR)
    message(FATAL_ERROR "install_openvino: INSTALL_DIR not set")
endif()

set(_LIB_DIR "${INSTALL_DIR}/runtime/lib/intel64")
set(_3P_DIR "${INSTALL_DIR}/runtime/3rdparty/lib")

# Glob all static libs
file(GLOB _OV_CORE_LIBS "${_LIB_DIR}/*.a")
file(GLOB _OV_3P_LIBS "${_3P_DIR}/*.a")

list(LENGTH _OV_CORE_LIBS _core_count)
list(LENGTH _OV_3P_LIBS _3p_count)

message(STATUS "install_openvino: found ${_core_count} core + ${_3p_count} 3rdparty static libs")

foreach(_lib ${_OV_CORE_LIBS})
    get_filename_component(_name "${_lib}" NAME)
    message(STATUS "  core: ${_name}")
endforeach()
foreach(_lib ${_OV_3P_LIBS})
    get_filename_component(_name "${_lib}" NAME)
    message(STATUS "  3rdparty: ${_name}")
endforeach()

# Create a combined thin archive using ar
# This avoids --whole-archive issues and simplifies linking
if(UNIX)
    set(_ALL_LIBS ${_OV_CORE_LIBS} ${_OV_3P_LIBS})
    set(_COMBINED "${_LIB_DIR}/libopenvino_all.a")

    if(_ALL_LIBS AND NOT EXISTS "${_COMBINED}")
        # Extract all .o files and re-archive
        set(_TMP_DIR "${INSTALL_DIR}/_combine_tmp")
        file(MAKE_DIRECTORY "${_TMP_DIR}")

        foreach(_lib ${_ALL_LIBS})
            execute_process(
                COMMAND ar x "${_lib}"
                WORKING_DIRECTORY "${_TMP_DIR}"
                RESULT_VARIABLE _ar_result
            )
        endforeach()

        file(GLOB _ALL_OBJS "${_TMP_DIR}/*.o" "${_TMP_DIR}/*.obj")
        list(LENGTH _ALL_OBJS _obj_count)
        message(STATUS "  combining ${_obj_count} object files into libopenvino_all.a")

        # ar in batches (avoid command line too long)
        set(_batch_size 500)
        set(_first TRUE)
        while(_ALL_OBJS)
            list(LENGTH _ALL_OBJS _remaining)
            if(_remaining GREATER _batch_size)
                list(SUBLIST _ALL_OBJS 0 ${_batch_size} _batch)
                list(SUBLIST _ALL_OBJS ${_batch_size} -1 _ALL_OBJS)
            else()
                set(_batch ${_ALL_OBJS})
                set(_ALL_OBJS "")
            endif()

            if(_first)
                execute_process(COMMAND ar rcs "${_COMBINED}" ${_batch})
                set(_first FALSE)
            else()
                execute_process(COMMAND ar rs "${_COMBINED}" ${_batch})
            endif()
        endwhile()

        file(REMOVE_RECURSE "${_TMP_DIR}")
        message(STATUS "  created ${_COMBINED}")
    endif()
endif()

message(STATUS "install_openvino: post-processing complete")
