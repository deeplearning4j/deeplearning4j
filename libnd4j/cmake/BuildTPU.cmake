# Configures the existing libnd4jtpu targets.
#
# MainBuildFlow owns source collection and already includes graph/tpu/*.cpp once.
# Eager operations use the mature host-native execution stack; TPU graph
# acceleration is exclusively owned by TpuGraphBackend/PjrtClientManager. The
# TPU graph lowering consumes the shared KernelSpec/KernelExpr semantic lane;
# there is no second eager PJRT client/cache or backend-local operation catalog.

function(setup_tpu_build)
    if(NOT SD_TPU)
        return()
    endif()

    if(NOT HAVE_PJRT OR NOT PJRT_INCLUDE_DIR)
        message(FATAL_ERROR
            "TPU target creation requires setup_pjrt_paths() before MainBuildFlow")
    endif()

    set(_OBJECT_TARGET "${SD_LIBRARY_NAME}_object")
    set(_SHARED_TARGET "${SD_LIBRARY_NAME}")
    if(NOT TARGET ${_OBJECT_TARGET} OR NOT TARGET ${_SHARED_TARGET})
        message(FATAL_ERROR
            "Expected TPU targets '${_OBJECT_TARGET}' and '${_SHARED_TARGET}'")
    endif()

    target_compile_definitions(${_OBJECT_TARGET} PUBLIC
        SD_TPU=1
        HAVE_PJRT=1
        __TPUBLAS__=true
        DEFAULT_ENGINE=samediff::ENGINE_TPU)
    target_compile_definitions(${_SHARED_TARGET} PUBLIC
        SD_TPU=1
        HAVE_PJRT=1
        DEFAULT_ENGINE=samediff::ENGINE_TPU)

    target_include_directories(${_OBJECT_TARGET} PUBLIC ${PJRT_INCLUDE_DIR})
    target_link_libraries(${_SHARED_TARGET} PUBLIC ${CMAKE_DL_LIBS})

    message(STATUS "TPU target configured: ${_SHARED_TARGET}")
    message(STATUS "  PJRT header: ${PJRT_INCLUDE_DIR}/pjrt_c_api.h")
    message(STATUS "  PJRT plugin: runtime-loaded")
    message(STATUS "  TPU graph runtime: TpuGraphBackend + PjrtClientManager")
endfunction()
