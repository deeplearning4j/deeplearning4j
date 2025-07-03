################################################################################
# CUDA Configuration Functions
# Functions for CUDA-specific build configuration and optimization
# UPDATED VERSION - Modern cuDNN detection and integration
################################################################################

# Modern cuDNN detection using updated FindCUDNN.cmake practices
function(setup_modern_cudnn)
    set(HAVE_CUDNN FALSE PARENT_SCOPE)
    
    if(NOT (HELPERS_cudnn AND SD_CUDA))
        message(STATUS "🔍 cuDNN: Skipped (HELPERS_cudnn=${HELPERS_cudnn}, SD_CUDA=${SD_CUDA})")
        return()
    endif()

    message(STATUS "