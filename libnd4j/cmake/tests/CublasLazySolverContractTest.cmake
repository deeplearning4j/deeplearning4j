if(NOT DEFINED LIBND4J_SOURCE_DIR OR LIBND4J_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "LIBND4J_SOURCE_DIR is required")
endif()

set(_cublas_helper
    "${LIBND4J_SOURCE_DIR}/include/helpers/cuda/cublasHelper.cu")
set(_zluda_runtime
    "${LIBND4J_SOURCE_DIR}/include/execution/ZludaRuntime.h")
get_filename_component(_repository_root "${LIBND4J_SOURCE_DIR}" DIRECTORY)
set(_cuda_zero_handler
    "${_repository_root}/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-backend-common/src/main/java/org/nd4j/jita/handler/impl/CudaZeroHandler.java")
foreach(_contract_file IN ITEMS
        "${_cublas_helper}" "${_zluda_runtime}" "${_cuda_zero_handler}")
    if(NOT EXISTS "${_contract_file}")
        message(FATAL_ERROR "Missing cuSolver contract input: ${_contract_file}")
    endif()
endforeach()

file(READ "${_cublas_helper}" _source)
file(READ "${_zluda_runtime}" _zluda_source)
file(READ "${_cuda_zero_handler}" _java_source)

string(FIND "${_source}" "_solvers[e] = solver_();" _eager_create)
if(NOT _eager_create EQUAL -1)
    message(FATAL_ERROR
        "CublasHelper constructor eagerly creates cuSolver handles")
endif()

foreach(_required_contract IN ITEMS
        "_solvers.resize(numDevices, nullptr);"
        "void* CublasHelper::solver()"
        "_solvers[deviceId] = solver_();"
        "cusolverDnDestroy(*solverHandle);")
    string(FIND "${_source}" "${_required_contract}" _contract_index)
    if(_contract_index EQUAL -1)
        message(FATAL_ERROR
            "CublasHelper lazy cuSolver contract omitted: ${_required_contract}")
    endif()
endforeach()

foreach(_required_zluda_contract IN ITEMS
        "inline bool supportsCusolver()"
        "if (!zluda::supportsCusolver())"
        "return nullptr;")
    string(FIND "${_zluda_source}
${_source}"
        "${_required_zluda_contract}" _zluda_contract_index)
    if(_zluda_contract_index EQUAL -1)
        message(FATAL_ERROR
            "ZLUDA cuSolver capability contract omitted: ${_required_zluda_contract}")
    endif()
endforeach()

foreach(_required_java_contract IN ITEMS
        "private cusolverDnHandle_t getCudaSolverHandle(OpaqueLaunchContext lc)"
        "if (nativeHandle == null || nativeHandle.isNull())"
        ".solverHandle(getCudaSolverHandle(lc))"
        "cusolverDnHandle_t solverHandle = getCudaSolverHandle(lc);"
        ".solverHandle(solverHandle)")
    string(FIND "${_java_source}" "${_required_java_contract}"
        _java_contract_index)
    if(_java_contract_index EQUAL -1)
        message(FATAL_ERROR
            "Java nullable cuSolver contract omitted: ${_required_java_contract}")
    endif()
endforeach()

string(FIND "${_java_source}"
    "nativeOps.lcSolverHandle(lc).retainReference()"
    _unsafe_solver_dereference)
if(NOT _unsafe_solver_dereference EQUAL -1)
    message(FATAL_ERROR
        "Java launch-context initialization still dereferences an optional cuSolver handle")
endif()
