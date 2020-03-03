/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <helpers/svd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

namespace sd    {
namespace ops     {
namespace helpers {


// FIXME -> we should optimize these helpers for the case when input matrices have c order (perform transpositions appropriately)

template <typename T>
__global__ static void inverseColumnSignCuda(void* vu, const Nd4jLong* uShapeInfo, void* vv, const Nd4jLong* vShapeInfo) {

    T* u = reinterpret_cast<T*>(vu);
    T* v = reinterpret_cast<T*>(vv);

    __shared__ int rank, uLastButOneColumn, vLastButOneColumn;    // uRank = vRank
    __shared__ Nd4jLong uLen, vLen;
    __shared__ Nd4jLong *sharedMem;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        rank = shape::rank(uShapeInfo);
        uLen = shape::length(uShapeInfo);
        vLen = shape::length(vShapeInfo);

        uLastButOneColumn = uShapeInfo[rank]     - 2;
        vLastButOneColumn = vShapeInfo[rank - 1] - 2;
    }

    __syncthreads();

    const auto ind = threadIdx.x + blockIdx.x * blockDim.x;

    auto coords = sharedMem + threadIdx.x * rank;

    // u
    for (Nd4jLong i = ind; i < uLen; i += gridDim.x * blockDim.x) {

        shape::index2coords(i, uShapeInfo, coords);

        if(coords[rank - 1] == 0 || coords[rank - 1] == uLastButOneColumn)   // do not change sign in first and last but one columns
            continue;

        const auto uOffset = shape::getOffset(uShapeInfo, coords);

        u[uOffset] = -u[uOffset];
    }

    // v
    for (Nd4jLong i = ind; i < vLen; i += gridDim.x * blockDim.x) {

        shape::index2coords(i, vShapeInfo, coords);

        if(coords[rank - 2] == 0 || coords[rank - 2] == vLastButOneColumn)   // do not change sign in first and last but one columns
            continue;

        const auto vOffset = shape::getOffset(vShapeInfo, coords);

        v[vOffset] = -v[vOffset];
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void inverseColumnSignCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                         void* vu, const Nd4jLong* uShapeInfo,
                                         void* vv, const Nd4jLong* vShapeInfo) {

    inverseColumnSignCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vu, uShapeInfo, vv, vShapeInfo);
}
BUILD_SINGLE_TEMPLATE(template void inverseColumnSignCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t* stream, void* vu, const Nd4jLong* uShapeInfo, void* vv, const Nd4jLong* vShapeInfo), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
static void svdQR(sd::LaunchContext* context, const NDArray* A, NDArray* S, NDArray* U, NDArray* VT, const bool fullUV, const bool calcUV) {

    // since cusa api cusolverDnDgesvd/cusolverDnSgesvd have following constrain on input matrix A: A_rows >= A_columns && A_order = 'f'
    // we make this function to have deal with 2 valid cases only:
    // 1) A_rows >= A_columns and A_corder = 'f'
    // 2) A_rows <= A_columns and A_corder = 'c'    - int this case perform transposition to get f order
    // if 1) or 2) are not met then throw exception

    // A  [m, n]
    // S  [n]
    // U  [m, m] or [m, n] if fullUV = false and m > n
    // VT [n, n] or [m, n] if fullUV = false and m < n

    if(A->rankOf() != 2)
        throw std::runtime_error("svdQR: rank of A array is not equal 2 !");

    auto m = A->sizeAt(0);
    auto n = A->sizeAt(1);
    const int minDim = m < n ? m : n;
    const char orderA = A->ordering();

    if(m < n)
        throw std::runtime_error("svdQR: due to cuda api input constrains given shape of A array are not valid !");

    if(std::vector<Nd4jLong>({minDim}) != S->getShapeAsVector())
        throw std::runtime_error("svdQR: wrong shape of S array !");

    if(calcUV) {

        if(fullUV && std::vector<Nd4jLong>({m,m}) != U->getShapeAsVector())
            throw std::runtime_error("svdQR: wrong shape of U array !");
        else if(!fullUV && std::vector<Nd4jLong>({m,minDim}) != U->getShapeAsVector())
            throw std::runtime_error("svdQR: wrong shape of U array !");

        if(fullUV && std::vector<Nd4jLong>({n,n}) != VT->getShapeAsVector())
            throw std::runtime_error("svdQR: wrong shape of VT array !");
        else if(!fullUV && std::vector<Nd4jLong>({minDim,n}) != VT->getShapeAsVector())
            throw std::runtime_error("svdQR: wrong shape of VT array !");
    }

    NDArray* pA  = const_cast<NDArray*>(A);
    NDArray* pS  = S;
    NDArray* pU  = U;
    NDArray* pVT = VT;

    std::vector<NDArray*> toDelete;

    if(pA->ews() != 1 || pA->ordering() == 'c') {
        pA = new NDArray(A->dup('f'));
        toDelete.push_back(pA);
    }

    if(S->ews() != 1) {
        pS = new NDArray(S->dup('f'));
        toDelete.push_back(pS);
    }

    if(calcUV) {

        if(pU->ews() != 1 || pU->ordering() == 'c') {
            pU = new NDArray(U->dup('f'));
            toDelete.push_back(pU);
        }

        if(pVT->ews() != 1 || pVT->ordering() == 'c') {
            pVT = new NDArray(VT->dup('f'));
            toDelete.push_back(pVT);
        }
    }

    // create cusolverDn handle
    cusolverDnHandle_t handle = nullptr;
    cusolverStatus_t status = cusolverDnCreate(&handle);
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdQR: cuda failed !", status);

    // stream
    status = cusolverDnSetStream(handle, *context->getCudaStream());
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdQR: cuda failed !", status);

    // query working space of SVD
    int lwork = 0;
    if(A->dataType() == DataType::DOUBLE)
        status = cusolverDnDgesvd_bufferSize(handle, m, n, &lwork);
    else if(A->dataType() == DataType::FLOAT32)
        status = cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);
    else
        throw std::invalid_argument("svdQR: given data type is unsupported !");

    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdQR: cuda failed !", status);

    // allocate memory for dWork
    void* dWork = nullptr;
    cudaError_t status2 = cudaMalloc((void**)&dWork , A->sizeOfT() * lwork);
    if(status2 != cudaSuccess)
        throw cuda_exception::build("svdQR: cuda failed !", status2);

    signed char jobu, jobvt;

    if(calcUV) {
        if(fullUV)
            jobu = jobvt = 'A';
        else
            jobu = jobvt = 'S';
    }
    else {
        jobu = jobvt = 'N';
    }

    int *devInfo = nullptr;
    void* rWork = nullptr;

    int lda(m), ldu, ldvt;

    if(calcUV) {
        ldu  = pU->sizeAt(0);
        ldvt = pVT->sizeAt(0);
    }

    PointersManager manager(context, "svdQR");

    NDArray::prepareSpecialUse({pS, pU, pVT}, {pA});

    // choose appropriate cuda gemm api depending on data types
    if(A->dataType() == DataType::DOUBLE) {
        status = cusolverDnDgesvd(handle, jobu, jobvt, m, n, reinterpret_cast<double*>(pA->getSpecialBuffer()), lda, reinterpret_cast<double*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<double*>(pU->getSpecialBuffer()) : nullptr, ldu, calcUV ? reinterpret_cast<double*>(pVT->getSpecialBuffer()) : nullptr, ldvt, reinterpret_cast<double*>(dWork), lwork, reinterpret_cast<double*>(rWork), devInfo);
    }
    else if(A->dataType() == DataType::FLOAT32) {
        status = cusolverDnSgesvd(handle, jobu, jobvt, m, n, reinterpret_cast<float*>(pA->getSpecialBuffer()), lda, reinterpret_cast<float*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<float*>(pU->getSpecialBuffer()) : nullptr, ldu, calcUV ? reinterpret_cast<float*>(pVT->getSpecialBuffer()) : nullptr, ldvt, reinterpret_cast<float*>(dWork), lwork, reinterpret_cast<float*>(rWork), devInfo);
    }
    else
        throw std::invalid_argument("svdQR: given data type is unsupported !");

    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdQR: cuda failed !", status);

    manager.synchronize();

    NDArray::registerSpecialUse({pS, pU, pVT}, {pA});

    S->assign(pS);

    if(calcUV) {
        U->assign(pU);
        VT->assign(pVT);
    }

    for (int i = toDelete.size() - 1; i >= 0; --i)
        delete toDelete[i];

    if (devInfo)
        cudaFree(devInfo);
    if (dWork )
        cudaFree(dWork);
    if (rWork)
        cudaFree(rWork);

    if(handle)
        cusolverDnDestroy(handle);

    // cudaDeviceReset();
}

//////////////////////////////////////////////////////////////////////////
static void svdJcb(sd::LaunchContext* context, const NDArray* A, NDArray* S, NDArray* U, NDArray* V, const bool fullUV, const bool calcUV) {

    // A [m, n]
    // S [n]
    // U [m, m] or [m, n] if fullUV = false and m > n
    // V [n, n] or [n, m] if fullUV = false and m < n

    if(A->rankOf() != 2)
        throw std::runtime_error("svdJcb: rank of A array is not equal 2 !");

    int m = A->sizeAt(0);
    int n = A->sizeAt(1);
    const int minDim = m < n ? m : n;

    if(std::vector<Nd4jLong>({minDim}) != S->getShapeAsVector())
        throw std::runtime_error("svdJcb: wrong shape of S array !");

    if(calcUV) {

        if(fullUV && std::vector<Nd4jLong>({m,m}) != U->getShapeAsVector())
            throw std::runtime_error("svdJcb: wrong shape of U array !");
        else if(!fullUV && std::vector<Nd4jLong>({m,minDim}) != U->getShapeAsVector())
            throw std::runtime_error("svdJcb: wrong shape of U array !");

        if(fullUV && std::vector<Nd4jLong>({n,n}) != V->getShapeAsVector())
            throw std::runtime_error("svdJcb: wrong shape of V array !");
        else if(!fullUV && std::vector<Nd4jLong>({n,minDim}) != V->getShapeAsVector())
            throw std::runtime_error("svdJcb: wrong shape of V array !");
    }

    NDArray* pA = const_cast<NDArray*>(A);

    const bool aForder = m == 1 || A->strideAt(0) == 1;
    const bool aCorder = n == 1 || A->strideAt(1) == 1;

    const bool transA = !aForder && aCorder;
    const bool dupA   = !aForder && !aCorder;

    std::vector<NDArray*> toDelete;

    if(dupA) {
        pA = new NDArray(A->dup('f'));
        toDelete.push_back(pA);
    }

    NDArray* pS = S;

    if(S->ews() != 1) {
        pS = new NDArray(S->dup('f'));
        toDelete.push_back(pS);
    }

    NDArray *pU(nullptr), *pV(nullptr);

    int lda = transA ? pA->strideAt(0) : pA->strideAt(1);
    int ldu(transA ? n : m), ldv(transA ? m : n);
    bool uForder(true), vForder(true);

    if(calcUV) {

        pU = transA ? V : U;
        pV = transA ? U : V;

        uForder = pU->sizeAt(0) == 1 || pU->strideAt(0) == 1;
        vForder = pV->sizeAt(0) == 1 || pV->strideAt(0) == 1;

        if(!uForder) {
            pU = new NDArray(pU->dup('f'));
            toDelete.push_back(pU);
        }

        if(!vForder) {
            pV = new NDArray(pV->dup('f'));
            toDelete.push_back(pV);
        }

        ldu = pU->strideAt(1);
        ldv = pV->strideAt(1);
    }

    // create cusolverDn handle
    cusolverDnHandle_t handle = nullptr;
    cusolverStatus_t status = cusolverDnCreate(&handle);
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);

    // stream
    status = cusolverDnSetStream(handle, *context->getCudaStream());
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);

    // set parameters
    gesvdjInfo_t gesvdjParams = nullptr;
    status = cusolverDnCreateGesvdjInfo(&gesvdjParams);
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);
     status = cusolverDnXgesvdjSetTolerance(gesvdjParams, 1.e-7);   // tolerance
     if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);
    status = cusolverDnXgesvdjSetMaxSweeps(gesvdjParams, 15);      // max_sweeps
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);

    int *devInfo = nullptr;
    const cusolverEigMode_t jobz = calcUV ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    const int econ = !fullUV;

    if(transA)
        math::nd4j_swap<int>(m, n);

    // *** avoid bug in cuda API ***
    void* nullPtr = nullptr;
    NDArray* arrToAvoidBugInAPI = nullptr;
    if(!calcUV && m != n) {
        int maxDim = m > n ? m : n;
        arrToAvoidBugInAPI = new NDArray('c', {maxDim, maxDim}, pA->dataType(), context);
        nullPtr = arrToAvoidBugInAPI->getSpecialBuffer();
    }
    // ******************

    NDArray::prepareSpecialUse({pS, pU, pV}, {pA});

    // query working space of SVD
    int lwork = 0;
    if(A->dataType() == DataType::DOUBLE)
        status = cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, reinterpret_cast<double*>(pA->getSpecialBuffer()), lda, reinterpret_cast<double*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<double*>(pU->getSpecialBuffer()) : reinterpret_cast<double*>(nullPtr), ldu, calcUV ? reinterpret_cast<double*>(pV->getSpecialBuffer()) : reinterpret_cast<double*>(nullPtr), ldv, &lwork, gesvdjParams);
    else if(A->dataType() == DataType::FLOAT32)
        status = cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, reinterpret_cast<float*>(pA->getSpecialBuffer()), lda, reinterpret_cast<float*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<float*>(pU->getSpecialBuffer()) : reinterpret_cast<float*>(nullPtr), ldu, calcUV ? reinterpret_cast<float*>(pV->getSpecialBuffer()) : reinterpret_cast<float*>(nullPtr), ldv, &lwork, gesvdjParams);
    else
        throw std::invalid_argument("svdJcb: given data type is unsupported !");

    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);

    // allocate memory dWork
    void* dWork = nullptr;
    auto status2 = cudaMalloc((void**)&dWork , A->sizeOfT() * lwork);
    if(status2 != cudaSuccess)
        throw cuda_exception::build("svdJcb: cuda failed !", status2);

    PointersManager manager(context, "svdJcb");

    // choose appropriate cuda gemm api depending on data types
    if(A->dataType() == DataType::DOUBLE) {
        status = cusolverDnDgesvdj(handle, jobz, econ, m, n, reinterpret_cast<double*>(pA->getSpecialBuffer()), lda, reinterpret_cast<double*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<double*>(pU->getSpecialBuffer()) : reinterpret_cast<double*>(nullPtr), ldu, calcUV ? reinterpret_cast<double*>(pV->getSpecialBuffer()) : reinterpret_cast<double*>(nullPtr), ldv, reinterpret_cast<double*>(dWork), lwork, devInfo, gesvdjParams);
    }
    else if(A->dataType() == DataType::FLOAT32) {
        status = cusolverDnSgesvdj(handle, jobz, econ, m, n, reinterpret_cast<float*>(pA->getSpecialBuffer()), lda, reinterpret_cast<float*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<float*>(pU->getSpecialBuffer()) : reinterpret_cast<float*>(nullPtr), ldu, calcUV ? reinterpret_cast<float*>(pV->getSpecialBuffer()) : reinterpret_cast<float*>(nullPtr), ldv, reinterpret_cast<float*>(dWork), lwork, devInfo, gesvdjParams);
    }
    else
        throw std::invalid_argument("svdJcb: given data type is unsupported !");

    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdJcb: cuda failed !", status);

    manager.synchronize();

    NDArray::registerSpecialUse({pS, pU, pV}, {pA});

    if(S->ews() != 1)
        S->assign(pS);

    if(calcUV) {

        if(!uForder)
            U->assign(transA ? pV : pU);
        if(!vForder)
            V->assign(transA ? pU : pV);
    }

    if(!calcUV && m != n)
        delete arrToAvoidBugInAPI;

    for (int i = toDelete.size() - 1; i >= 0; --i)
        delete toDelete[i];

    if (devInfo)
        cudaFree(devInfo);
    if (dWork )
        cudaFree(dWork);
    if(handle)
        cusolverDnDestroy(handle);
    if(gesvdjParams)
        cusolverDnDestroyGesvdjInfo(gesvdjParams);

    // cudaDeviceReset();
}

//////////////////////////////////////////////////////////////////////////
static void svdBatched(sd::LaunchContext* context, const NDArray* A, NDArray* S, NDArray* U, NDArray* V, const bool fullUV, const bool calcUV) {

    // A [..., m, n]
    // S [..., n]
    // U [..., m, m] or [..., m, n] if fullUV = false and m > n
    // V [..., n, n] or [..., n, m] if fullUV = false and m < n

    auto m = A->sizeAt(-2);
    auto n = A->sizeAt(-1);
    const int minDim = m < n ? m : n;
    const Nd4jLong bS = A->lengthOf() / (m * n);

    if(m > 32 || n > 32)
        throw std::runtime_error("svdBatched: numbers of rows and columns should be <= 32 !");

    if(minDim != S->sizeAt(-1))
        throw std::runtime_error("svdBatched: wrong shape of S array !");

    if(calcUV) {

        if(U->sizeAt(-2) != m)
            throw std::runtime_error("svdBatched: wrong shape of U array !");
        if(U->sizeAt(-1) != (fullUV ? m : minDim))
            throw std::runtime_error("svdBatched: wrong shape of U array !");
        if(U->lengthOf() / (U->sizeAt(-2) * U->sizeAt(-1)) != bS)
            throw std::runtime_error("svdBatched: wrong shape of U array !");

        if(V->sizeAt(-2) != n)
              throw std::runtime_error("svdBatched: wrong shape of V array !");
        if(V->sizeAt(-1) != (fullUV ? n : minDim))
            throw std::runtime_error("svdBatched: wrong shape of V array !");
        if(V->lengthOf() / (V->sizeAt(-2) * V->sizeAt(-1)) != bS)
            throw std::runtime_error("svdBatched: wrong shape of V array !");
    }

    NDArray* pA = const_cast<NDArray*>(A);
    NDArray* pS = S;
    NDArray* pU = U;
    NDArray* pV = V;

    std::vector<NDArray*> toDelete;

    if(pA->ews() != 1 || pA->ordering() == 'c') {
        pA = new NDArray(A->dup('f'));
        toDelete.push_back(pA);
    }

    if(S->ews() != 1) {
        pS = new NDArray(S->dup('f'));
        toDelete.push_back(pS);
    }

    if(calcUV) {

        if(pU->ews() != 1 || pU->ordering() == 'c') {
            pU = new NDArray(U->dup('f'));
            toDelete.push_back(pU);
        }

        if(pV->ews() != 1 || pV->ordering() == 'c') {
            pV = new NDArray(V->dup('f'));
            toDelete.push_back(pV);
        }
    }

    // create cusolverDn handle
    cusolverDnHandle_t handle = nullptr;
    cusolverStatus_t status = cusolverDnCreate(&handle);
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);

    // stream
    status = cusolverDnSetStream(handle, *context->getCudaStream());
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);

    // set parameters
    gesvdjInfo_t gesvdjParams = nullptr;
    status = cusolverDnCreateGesvdjInfo(&gesvdjParams);
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);
     status = cusolverDnXgesvdjSetTolerance(gesvdjParams, 1.e-7);   // tolerance
     if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);
    status = cusolverDnXgesvdjSetMaxSweeps(gesvdjParams, 15);      // max_sweeps
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);

    // devInfo
    int *devInfo = nullptr;
    auto status2 = cudaMalloc((void**)&devInfo, sizeof(int) * bS);
    if(status2 != cudaSuccess)
        throw cuda_exception::build("svdBatched: cuda failed !", status2);
    status2 = cudaDeviceSynchronize();
    if(status2 != cudaSuccess)
        throw cuda_exception::build("svdJcb: cuda failed !", status2);

    const cusolverEigMode_t jobz = calcUV ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    int lda(m), ldu, ldv;

    if(calcUV) {
        ldu = pU->sizeAt(-2);
        ldv = pV->sizeAt(-2);
    }

    // Ak (i,j) = A[i + 5*j + 25*k]

    // query working space of SVD
    int lwork = 0;
    if(A->dataType() == DataType::DOUBLE)
        status = cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, reinterpret_cast<double*>(pA->getSpecialBuffer()), lda, reinterpret_cast<double*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<double*>(pU->getSpecialBuffer()) : nullptr, ldu, calcUV ? reinterpret_cast<double*>(pV->getSpecialBuffer()) : nullptr, ldv, &lwork, gesvdjParams, bS);
    else if(A->dataType() == DataType::FLOAT32)
        status = cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, reinterpret_cast<float*>(pA->getSpecialBuffer()), lda, reinterpret_cast<float*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<float*>(pU->getSpecialBuffer()) : nullptr, ldu, calcUV ? reinterpret_cast<float*>(pV->getSpecialBuffer()) : nullptr, ldv, &lwork, gesvdjParams, bS);
    else
        throw std::invalid_argument("svdBatched: given data type is unsupported !");

    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);

    // allocate memory dWork
    void* dWork = nullptr;
    status2 = cudaMalloc((void**)&dWork , A->sizeOfT() * lwork);
    if(status2 != cudaSuccess)
        throw cuda_exception::build("svdBatched: cuda failed !", status2);
    status2 = cudaDeviceSynchronize();
    if(status2 != cudaSuccess)
        throw cuda_exception::build("svdBatched: cuda failed !", status2);

    PointersManager manager(context, "svdBatched");

    NDArray::prepareSpecialUse({pS, pU, pV}, {pA});

    // choose appropriate cuda gemm api depending on data types
    if(A->dataType() == DataType::DOUBLE) {
        status = cusolverDnDgesvdjBatched(handle, jobz, m, n, reinterpret_cast<double*>(pA->getSpecialBuffer()), lda, reinterpret_cast<double*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<double*>(pU->getSpecialBuffer()) : nullptr, ldu, calcUV ? reinterpret_cast<double*>(pV->getSpecialBuffer()) : nullptr, ldv, reinterpret_cast<double*>(dWork), lwork, devInfo, gesvdjParams, bS);
    }
    else if(A->dataType() == DataType::FLOAT32) {
        status = cusolverDnSgesvdjBatched(handle, jobz, m, n, reinterpret_cast<float*>(pA->getSpecialBuffer()), lda, reinterpret_cast<float*>(pS->getSpecialBuffer()), calcUV ? reinterpret_cast<float*>(pU->getSpecialBuffer()) : nullptr, ldu, calcUV ? reinterpret_cast<float*>(pV->getSpecialBuffer()) : nullptr, ldv, reinterpret_cast<float*>(dWork), lwork, devInfo, gesvdjParams, bS);
    }
    else
        throw std::invalid_argument("svdBatched: given data type is unsupported !");

    if(status != CUSOLVER_STATUS_SUCCESS)
        throw cuda_exception::build("svdBatched: cuda failed !", status);

    manager.synchronize();

    NDArray::registerSpecialUse({pS, pU, pV}, {pA});

    S->assign(pS);

    if(calcUV) {
        U->assign(pU);
        V->assign(pV);
    }

    for (int i = toDelete.size() - 1; i >= 0; --i)
        delete toDelete[i];

    if (devInfo)
        cudaFree(devInfo);
    if (dWork )
        cudaFree(dWork);
    if(handle)
        cusolverDnDestroy(handle);
    if(gesvdjParams)
        cusolverDnDestroyGesvdjInfo(gesvdjParams);

    // cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////
void svd(sd::LaunchContext* context, const NDArray* x, const std::vector<NDArray*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum) {

    NDArray* S = outArrs[0];
    NDArray* U = outArrs[1];
    // NDArray VT = outArrs[2]->transpose();
    NDArray* V = outArrs[2];

    NDArray::prepareSpecialUse({S, U, V}, {x});

    if(x->rankOf() == 2) {
        // svdQR(context, x, S, U, VT, fullUV, calcUV);
        svdJcb(context, x, S, U, V, fullUV, calcUV);
    }
    else {

        // svdBatched(context, *x, *S, *U, *V, fullUV, calcUV);

        ResultSet *tadsU(nullptr), *tadsV(nullptr);

        auto tadsX = x->allTensorsAlongDimension({x->rankOf() - 2, x->rankOf() - 1});
        auto tadsS = S->allTensorsAlongDimension({S->rankOf() - 1});

        if(calcUV) {
            tadsU = new ResultSet(U->allTensorsAlongDimension({U->rankOf() - 2, U->rankOf() - 1}));
            tadsV = new ResultSet(V->allTensorsAlongDimension({V->rankOf() - 2, V->rankOf() - 1}));
        }

        for (int i = 0; i < tadsX.size(); ++i)
            svdJcb(context, tadsX.at(i), tadsS.at(i), calcUV ? tadsU->at(i) : nullptr, calcUV ? tadsV->at(i) : nullptr, fullUV, calcUV);

        if(calcUV) {
            delete tadsU;
            delete tadsV;
        }
    }

    NDArray::registerSpecialUse({S, U, V}, {x});
}


}
}
}