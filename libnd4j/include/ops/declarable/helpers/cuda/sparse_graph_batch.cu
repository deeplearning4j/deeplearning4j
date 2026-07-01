/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// CUDA implementation of graph_disjoint_union helpers.
//
// Each per-graph copy is launched as a separate kernel on the context stream.
// For typical GNN batch sizes (K = 8..64, N_k = 10..5000 nodes,
// F = 64..512 features), one-thread-per-(node,feature) is efficient.
//
// No raw cudaMalloc. All device buffers accessed via specialBuffer().
// prepareSpecialUse / registerSpecialUse bracket each sequence.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_graph_batch.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Kernel: copy X_k rows into Xout at offset cumN
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void copyXKernel(
    const X*  xkBuf,      // [N_k, F] source
    X*        xoBuf,      // [sumN, F] destination
    int*      bvBuf,      // [sumN]  batch vector
    LongType  Nk,
    LongType  F,
    LongType  xks0, LongType xks1,
    LongType  xos0, LongType xos1,
    LongType  cumN,
    int       graphIdx)
{
    const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= Nk * F) return;
    const LongType i = tid / F;
    const LongType f = tid % F;
    xoBuf[(cumN + i) * xos0 + f * xos1] = xkBuf[i * xks0 + f * xks1];
    if (f == 0) bvBuf[cumN + i] = graphIdx;
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel: copy vals_k into valsOut at offset cumNnz
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void copyValsKernel(
    const X*  vkBuf,
    X*        voBuf,
    LongType  nnzk,
    LongType  vks0,
    LongType  vos0,
    LongType  cumNnz)
{
    const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= nnzk) return;
    voBuf[(cumNnz + tid) * vos0] = vkBuf[tid * vks0];
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel: copy and offset colIdx_k
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static SD_KERNEL void copyColIdxKernel(
    const I*  cikBuf,
    I*        ciBuf,
    LongType  nnzk,
    LongType  ciks0,
    LongType  cios0,
    LongType  cumNnz,
    I         colOffset)
{
    const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= nnzk) return;
    ciBuf[(cumNnz + tid) * cios0] = cikBuf[tid * ciks0] + colOffset;
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel: stitch rowPtr_k (offset by cumNnz, placed at cumN+1..cumN+Nk)
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static SD_KERNEL void stitchRowPtrKernel(
    const I*  rpkBuf,      // rowPtr_k [N_k+1]
    I*        rpBuf,       // rowPtr_combined [sumN+1]
    LongType  Nk,
    LongType  rpks0,
    LongType  rpos0,
    LongType  cumN,
    I         nnzOffset)
{
    const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= Nk) return;
    // write position cumN + 1 + tid from source position 1 + tid
    rpBuf[(cumN + 1 + tid) * rpos0] = rpkBuf[(1 + tid) * rpks0] + nnzOffset;
}

// ────────────────────────────────────────────────────────────────────────────
// Forward CUDA dispatch (typed)
// ────────────────────────────────────────────────────────────────────────────
template <typename X, typename I>
static void graphDisjointUnionFwdCuda_(
    const std::vector<NDArray*>& Xs,
    const std::vector<NDArray*>& vals,
    const std::vector<NDArray*>& colIdxs,
    const std::vector<NDArray*>& rowPtrs,
    NDArray& Xout,
    NDArray& valsOut,
    NDArray& colIdxOut,
    NDArray& rowPtrOut,
    NDArray& batchVec)
{
    const LongType K = static_cast<LongType>(Xs.size());
    const LongType F = Xout.sizeAt(1);

    auto* stream = Xout.getContext()->getCudaStream();

    X*   xoBuf  = reinterpret_cast<X*>(Xout.specialBuffer());
    X*   voBuf  = reinterpret_cast<X*>(valsOut.specialBuffer());
    I*   ciBuf  = reinterpret_cast<I*>(colIdxOut.specialBuffer());
    I*   rpBuf  = reinterpret_cast<I*>(rowPtrOut.specialBuffer());
    int* bvBuf  = reinterpret_cast<int*>(batchVec.specialBuffer());

    // Zero rowPtr_combined[0]
    cudaMemsetAsync(rpBuf, 0, sizeof(I), *stream);

    const LongType xos0 = Xout.strideAt(0);
    const LongType xos1 = Xout.strideAt(1);
    const LongType vos0 = valsOut.strideAt(0);
    const LongType cios0 = colIdxOut.strideAt(0);
    const LongType rpos0 = rowPtrOut.strideAt(0);

    LongType cumN   = 0;
    LongType cumNnz = 0;

    for (LongType k = 0; k < K; ++k) {
        const LongType Nk   = Xs[k]->sizeAt(0);
        const LongType nnzk = vals[k]->lengthOf();

        const X*  xkBuf  = reinterpret_cast<const X*>(Xs[k]->specialBuffer());
        const X*  vkBuf  = reinterpret_cast<const X*>(vals[k]->specialBuffer());
        const I*  cikBuf = reinterpret_cast<const I*>(colIdxs[k]->specialBuffer());
        const I*  rpkBuf = reinterpret_cast<const I*>(rowPtrs[k]->specialBuffer());

        const LongType xks0  = Xs[k]->strideAt(0);
        const LongType xks1  = Xs[k]->strideAt(1);
        const LongType vks0  = vals[k]->strideAt(0);
        const LongType ciks0 = colIdxs[k]->strideAt(0);
        const LongType rpks0 = rowPtrs[k]->strideAt(0);

        int bs = 256;

        // X + batchVec
        if (Nk * F > 0) {
            int gs = static_cast<int>((Nk * F + bs - 1) / bs);
            copyXKernel<X><<<gs, bs, 0, *stream>>>(
                xkBuf, xoBuf, bvBuf, Nk, F, xks0, xks1, xos0, xos1, cumN, static_cast<int>(k));
        }

        // vals
        if (nnzk > 0) {
            int gs = static_cast<int>((nnzk + bs - 1) / bs);
            copyValsKernel<X><<<gs, bs, 0, *stream>>>(
                vkBuf, voBuf, nnzk, vks0, vos0, cumNnz);
        }

        // colIdx with offset
        if (nnzk > 0) {
            int gs = static_cast<int>((nnzk + bs - 1) / bs);
            copyColIdxKernel<I><<<gs, bs, 0, *stream>>>(
                cikBuf, ciBuf, nnzk, ciks0, cios0, cumNnz, static_cast<I>(cumN));
        }

        // rowPtr stitch
        if (Nk > 0) {
            int gs = static_cast<int>((Nk + bs - 1) / bs);
            stitchRowPtrKernel<I><<<gs, bs, 0, *stream>>>(
                rpkBuf, rpBuf, Nk, rpks0, rpos0, cumN, static_cast<I>(cumNnz));
        }

        cumN   += Nk;
        cumNnz += nnzk;
    }

    // Write final rowPtr[sumN] = sumNnz  (scalar H2D)
    I finalVal = static_cast<I>(cumNnz);
    cudaMemcpyAsync(rpBuf + cumN * rpos0, &finalVal, sizeof(I),
                    cudaMemcpyHostToDevice, *stream);
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernels: slice dX and dVals
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static SD_KERNEL void sliceXKernel(
    const X*  dxoBuf,    // [sumN, F]
    X*        dxkBuf,    // [N_k, F]
    LongType  Nk,
    LongType  F,
    LongType  dxos0, LongType dxos1,
    LongType  dxks0, LongType dxks1,
    LongType  cumN)
{
    const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= Nk * F) return;
    const LongType i = tid / F;
    const LongType f = tid % F;
    dxkBuf[i * dxks0 + f * dxks1] = dxoBuf[(cumN + i) * dxos0 + f * dxos1];
}

template <typename X>
static SD_KERNEL void sliceValsKernel(
    const X*  dvoBuf,
    X*        dvkBuf,
    LongType  nnzk,
    LongType  dvos0,
    LongType  dvks0,
    LongType  cumNnz)
{
    const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= nnzk) return;
    dvkBuf[tid * dvks0] = dvoBuf[(cumNnz + tid) * dvos0];
}

template <typename X>
static void graphDisjointUnionBpCuda_(
    const std::vector<sd::LongType>& cumN,
    const std::vector<sd::LongType>& cumNnz,
    NDArray& dXout,
    NDArray& dValsOut,
    std::vector<NDArray*>& dXs,
    std::vector<NDArray*>& dVals)
{
    const LongType K = static_cast<LongType>(dXs.size());
    const LongType F = dXout.sizeAt(1);

    auto* stream = dXout.getContext()->getCudaStream();

    const X*  dxoBuf  = reinterpret_cast<const X*>(dXout.specialBuffer());
    const X*  dvoBuf  = reinterpret_cast<const X*>(dValsOut.specialBuffer());
    const LongType dxos0 = dXout.strideAt(0);
    const LongType dxos1 = dXout.strideAt(1);
    const LongType dvos0 = dValsOut.strideAt(0);

    for (LongType k = 0; k < K; ++k) {
        const LongType Nk   = cumN[k + 1]   - cumN[k];
        const LongType nnzk = cumNnz[k + 1] - cumNnz[k];

        X* dxkBuf  = reinterpret_cast<X*>(dXs[k]->specialBuffer());
        X* dvkBuf  = reinterpret_cast<X*>(dVals[k]->specialBuffer());
        const LongType dxks0 = dXs[k]->strideAt(0);
        const LongType dxks1 = dXs[k]->strideAt(1);
        const LongType dvks0 = dVals[k]->strideAt(0);

        int bs = 256;
        if (Nk * F > 0) {
            int gs = static_cast<int>((Nk * F + bs - 1) / bs);
            sliceXKernel<X><<<gs, bs, 0, *stream>>>(
                dxoBuf, dxkBuf, Nk, F, dxos0, dxos1, dxks0, dxks1, cumN[k]);
        }
        if (nnzk > 0) {
            int gs = static_cast<int>((nnzk + bs - 1) / bs);
            sliceValsKernel<X><<<gs, bs, 0, *stream>>>(
                dvoBuf, dvkBuf, nnzk, dvos0, dvks0, cumNnz[k]);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public interface
// ────────────────────────────────────────────────────────────────────────────

void graph_disjoint_union_fwd(
    const std::vector<NDArray*>& Xs,
    const std::vector<NDArray*>& vals,
    const std::vector<NDArray*>& colIdxs,
    const std::vector<NDArray*>& rowPtrs,
    NDArray& Xout,
    NDArray& valsOut,
    NDArray& colIdxOut,
    NDArray& rowPtrOut,
    NDArray& batchVec)
{
    std::vector<NDArray*> inputs;
    for (auto* p : Xs)      inputs.push_back(p);
    for (auto* p : vals)    inputs.push_back(p);
    for (auto* p : colIdxs) inputs.push_back(p);
    for (auto* p : rowPtrs) inputs.push_back(p);

    NDArray::prepareSpecialUse({&Xout, &valsOut, &colIdxOut, &rowPtrOut, &batchVec}, inputs);

    BUILD_DOUBLE_SELECTOR(Xout.dataType(), colIdxOut.dataType(),
                          graphDisjointUnionFwdCuda_,
                          (Xs, vals, colIdxs, rowPtrs, Xout, valsOut, colIdxOut, rowPtrOut, batchVec),
                          SD_FLOAT_TYPES, SD_INDEXING_TYPES);

    NDArray::registerSpecialUse({&Xout, &valsOut, &colIdxOut, &rowPtrOut, &batchVec}, inputs);
}

void graph_disjoint_union_bp(
    const std::vector<sd::LongType>& cumN,
    const std::vector<sd::LongType>& cumNnz,
    NDArray& dXout,
    NDArray& dValsOut,
    std::vector<NDArray*>& dXs,
    std::vector<NDArray*>& dVals)
{
    std::vector<NDArray*> inputs = {&dXout, &dValsOut};
    std::vector<NDArray*> outputs;
    for (auto* p : dXs)   outputs.push_back(p);
    for (auto* p : dVals) outputs.push_back(p);

    NDArray::prepareSpecialUse(outputs, inputs);

    BUILD_SINGLE_SELECTOR(dXout.dataType(), graphDisjointUnionBpCuda_,
                          (cumN, cumNnz, dXout, dValsOut, dXs, dVals),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse(outputs, inputs);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
