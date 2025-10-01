# cmake/Ccache.cmake
# Automatic ccache detection and configuration

function(setup_ccache)
    # Check if ccache is available
    find_program(CCACHE_PROGRAM ccache)
    
    if(CCACHE_PROGRAM)
        message(STATUS "✅ Found ccache: ${CCACHE_PROGRAM}")
        
        # Set ccache as compiler launcher
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" PARENT_SCOPE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" PARENT_SCOPE)
        
        # For CUDA builds
        if(SD_CUDA AND CMAKE_CUDA_COMPILER)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" PARENT_SCOPE)
            message(STATUS "✅ Enabled ccache for CUDA compilation")
        endif()
        
        # Configure ccache settings for optimal performance
        execute_process(COMMAND ${CCACHE_PROGRAM} --max-size=10G ERROR_QUIET)
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression=true ERROR_QUIET)
        execute_process(COMMAND ${CCACHE_PROGRAM} --set-config=compression_level=6 ERROR_QUIET)
        
        # Show ccache statistics
        execute_process(
            COMMAND ${CCACHE_PROGRAM} --show-stats
            OUTPUT_VARIABLE CCACHE_STATS
            ERROR_QUIET
        )
        message(STATUS "📊 Ccache statistics:\n${CCACHE_STATS}")
        
        message(STATUS "✅ Ccache enabled - builds will be significantly faster after first compilation")
    else()
        message(STATUS "ℹ️  Ccache not found - install with: sudo apt-get install ccache (or brew install ccache)")
        message(STATUS "   Continuing without ccache (builds will be slower)")
    endif()
endfunction()
