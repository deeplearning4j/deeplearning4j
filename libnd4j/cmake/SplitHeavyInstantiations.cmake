# SplitHeavyInstantiations.cmake
# ----------------------------------------------------------------------------
# Splits a monolithic op whose BUILD_SINGLE/DOUBLE_TEMPLATE instantiates the full
# SD_COMMON_TYPES matrix in ONE translation unit (-> 7-15 MB objects, multi-GB
# cicc peaks, 100-200 s serial long-poles) into one generated TU per type. Each
# TU instantiates a single type slice, so per-TU compile memory/time drop ~Nx and
# the work parallelizes across make -j. The type set is UNCHANGED — only the
# instantiation is distributed across TUs.
#
# Measured (transform_strict, single arch): monolith 196 s / 2.68 GB / 14.8 MB
# -> per split TU 20.6 s / 456 MB / 1.14 MB.
#
# Mechanism (uniform across single/double/function templates):
#   - each op's BUILD_*_TEMPLATE line is wrapped:
#       #ifdef SD_SPLIT_TYPE_INDEX
#       #if COUNT_NARG(SD_COMMON_TYPES) > SD_SPLIT_TYPE_INDEX
#         BUILD_..._TEMPLATE(..., M_CONCAT1(SD_COMMON_TYPES_, SD_SPLIT_TYPE_INDEX) [, second]);
#       #endif
#       #else
#         BUILD_..._TEMPLATE(..., SD_COMMON_TYPES [, second]);   <- original, for non-split builds
#       #endif
#   - this module excludes the monolith from the source list and generates
#     <op>_<idx>.cu from split_instantiation.cu.in, each #define-ing the index
#     and #include-ing the op (so the guarded line emits one type slice).
#
# split_heavy_op(<src_list_var> <op_basename.cu> <op_include_path> <max_idx>)
#   src_list_var : CMake list of .cu sources (edited in place)
#   op_basename  : file name to drop from the list, e.g. transform_strict.cu
#   op_include   : path used in `#include <...>` (resolved via -I include),
#                  e.g. loops/cuda/transform/transform_strict.cu
#   max_idx      : generate indices 0..max_idx; keep >= (type_count - 1). For
#                  ops split on SD_COMMON_TYPES (13 types) use 12.

set(_SPLIT_GENERIC_TEMPLATE "${CMAKE_CURRENT_LIST_DIR}/split_instantiation.cu.in"
    CACHE INTERNAL "Generic per-type split TU template")

function(split_heavy_op src_list_var op_basename op_include max_idx)
    if(NOT EXISTS "${_SPLIT_GENERIC_TEMPLATE}")
        message(WARNING "split_heavy_op: template missing: ${_SPLIT_GENERIC_TEMPLATE} — skipping ${op_basename}")
        return()
    endif()

    # Drop the monolith from the source list (anchored on the exact file name so
    # generated <stem>_<idx>.cu are not matched).
    string(REPLACE "." "\\." _mono_re "${op_basename}")
    list(FILTER ${src_list_var} EXCLUDE REGEX "/${_mono_re}$")

    get_filename_component(_stem "${op_basename}" NAME_WE)
    get_filename_component(_ext "${op_basename}" EXT)   # .cu (CUDA) or .cpp (CPU); generated TUs match
    set(_gen_dir "${CMAKE_BINARY_DIR}/split_instantiations")
    file(MAKE_DIRECTORY "${_gen_dir}")

    set(_generated "")
    foreach(_idx RANGE 0 ${max_idx})
        set(SD_SPLIT_IDX "${_idx}")
        set(OP_INCLUDE "${op_include}")
        set(_out "${_gen_dir}/${_stem}_${_idx}${_ext}")
        configure_file("${_SPLIT_GENERIC_TEMPLATE}" "${_out}" @ONLY)
        list(APPEND _generated "${_out}")
    endforeach()

    list(APPEND ${src_list_var} ${_generated})
    set(${src_list_var} "${${src_list_var}}" PARENT_SCOPE)

    math(EXPR _n "${max_idx} + 1")
    message(STATUS "Split heavy op: ${op_basename} -> ${_n} per-type TUs")
endfunction()
