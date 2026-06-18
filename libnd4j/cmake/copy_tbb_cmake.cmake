# copy_tbb_cmake.cmake — copy TBB cmake config from whichever path exists
# TBB installs under lib64/ on some distros and lib/ on others.
# Usage: cmake -DTBB_PREFIX=<tbb_install_dir> -DTBB_DST=<destination> -P copy_tbb_cmake.cmake

set(_candidates
    "${TBB_PREFIX}/lib64/cmake/TBB"
    "${TBB_PREFIX}/lib/cmake/TBB"
    "${TBB_PREFIX}/lib/cmake/tbb"
)

foreach(_candidate IN LISTS _candidates)
    if(IS_DIRECTORY "${_candidate}")
        message(STATUS "Copying TBB cmake config from ${_candidate}")
        file(COPY "${_candidate}/" DESTINATION "${TBB_DST}")
        return()
    endif()
endforeach()

message(WARNING "TBB cmake config not found in any candidate path (tried: ${_candidates}). Skipping copy.")
