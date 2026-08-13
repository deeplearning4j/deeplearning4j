if(NOT DEFINED LIBND4J_SOURCE_DIR OR LIBND4J_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "LIBND4J_SOURCE_DIR is required")
endif()

set(_cublas_helper
    "${LIBND4J_SOURCE_DIR}/include/helpers/cuda/cublasHelper.cu")
if(NOT EXISTS "${_cublas_helper}")
    message(FATAL_ERROR "Missing CUDA BLAS helper: ${_cublas_helper}")
endif()

file(READ "${_cublas_helper}" _source)

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
