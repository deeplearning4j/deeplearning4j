/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// CPU implementation of graph_disjoint_union helpers.
//
// Forward:
//   - Concatenate K node-feature matrices along axis 0 → X_combined [sumN, F]
//   - Concatenate K edge-weight arrays              → vals_combined  [sumNnz]
//   - Concatenate K colIdx arrays with offset       → colIdx_combined [sumNnz]
//       colIdx_combined[cumNnz[k] + e] = colIdxs[k][e] + cumN[k]
//   - Stitch K rowPtr arrays with offset            → rowPtr_combined [sumN+1]
//       rowPtr_combined[0] = 0
//       for each graph k, each row i in [0, N_k):
//         rowPtr_combined[cumN[k] + i] = rowPtrs[k][i] + cumNnz[k]
//       rowPtr_combined[sumN] = sumNnz
//   - Fill batchVec: batchVec[cumN[k] .. cumN[k+1]) = k
//
// Backward:
//   - Slice dX_combined rows [cumN[k], cumN[k+1]) → dXs[k]
//   - Slice dVals_combined [cumNnz[k], cumNnz[k+1]) → dVals[k]
//

#include <ops/declarable/helpers/sparse_graph_batch.h>
#include <system/op_boilerplate.h>
#include <cstring>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward helpers (typed)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void graphDisjointUnionFwd_(
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
    const sd::LongType K = static_cast<sd::LongType>(Xs.size());
    const sd::LongType F = Xout.sizeAt(1);

    X*  xoBuf  = Xout.bufferAsT<X>();
    X*  voBuf  = valsOut.bufferAsT<X>();
    I*  ciBuf  = colIdxOut.bufferAsT<I>();
    I*  rpBuf  = rowPtrOut.bufferAsT<I>();
    int* bvBuf = batchVec.bufferAsT<int>();

    sd::LongType cumN   = 0;
    sd::LongType cumNnz = 0;

    rpBuf[0] = static_cast<I>(0);

    for (sd::LongType k = 0; k < K; ++k) {
        const sd::LongType Nk   = Xs[k]->sizeAt(0);
        const sd::LongType nnzk = vals[k]->lengthOf();

        // --- copy X_k rows into Xout ---
        const X*  xkBuf  = Xs[k]->bufferAsT<X>();
        const sd::LongType xks0 = Xs[k]->strideAt(0);
        const sd::LongType xks1 = Xs[k]->strideAt(1);
        const sd::LongType xos0 = Xout.strideAt(0);
        const sd::LongType xos1 = Xout.strideAt(1);

        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType i = 0; i < Nk; ++i) {
            for (sd::LongType f = 0; f < F; ++f) {
                xoBuf[(cumN + i) * xos0 + f * xos1] = xkBuf[i * xks0 + f * xks1];
            }
            bvBuf[cumN + i] = static_cast<int>(k);
        }

        // --- copy vals_k ---
        const X* vkBuf = vals[k]->bufferAsT<X>();
        const sd::LongType vks0 = vals[k]->strideAt(0);
        const sd::LongType vos0 = valsOut.strideAt(0);
        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType e = 0; e < nnzk; ++e) {
            voBuf[(cumNnz + e) * vos0] = vkBuf[e * vks0];
        }

        // --- copy and offset colIdx_k ---
        const I* cikBuf = colIdxs[k]->bufferAsT<I>();
        const sd::LongType ciks0 = colIdxs[k]->strideAt(0);
        const sd::LongType cios0 = colIdxOut.strideAt(0);
        const I  colOffset = static_cast<I>(cumN);
        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType e = 0; e < nnzk; ++e) {
            ciBuf[(cumNnz + e) * cios0] = cikBuf[e * ciks0] + colOffset;
        }

        // --- stitch rowPtr_k (skip the first element since rowPtr[0] = 0 always) ---
        const I* rpkBuf = rowPtrs[k]->bufferAsT<I>();
        const sd::LongType rpks0 = rowPtrs[k]->strideAt(0);
        const sd::LongType rpos0 = rowPtrOut.strideAt(0);
        const I  nnzOffset = static_cast<I>(cumNnz);
        // write rowPtr_combined[cumN+1 .. cumN+Nk] from rowPtrs[k][1..Nk]
        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType i = 0; i < Nk; ++i) {
            rpBuf[(cumN + i + 1) * rpos0] = rpkBuf[(i + 1) * rpks0] + nnzOffset;
        }

        cumN   += Nk;
        cumNnz += nnzk;
    }
    // final element (should already be set by the last graph's last row)
    rpBuf[cumN] = static_cast<I>(cumNnz);
}

// ────────────────────────────────────────────────────────────────────────────
// Backward helpers (typed)
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static void graphDisjointUnionBp_(
    const std::vector<sd::LongType>& cumN,
    const std::vector<sd::LongType>& cumNnz,
    NDArray& dXout,
    NDArray& dValsOut,
    std::vector<NDArray*>& dXs,
    std::vector<NDArray*>& dVals)
{
    const sd::LongType K = static_cast<sd::LongType>(dXs.size());
    const sd::LongType F = dXout.sizeAt(1);

    const X* dxoBuf  = dXout.bufferAsT<X>();
    const X* dvoeBuf = dValsOut.bufferAsT<X>();
    const sd::LongType dxos0 = dXout.strideAt(0);
    const sd::LongType dxos1 = dXout.strideAt(1);
    const sd::LongType dvos0 = dValsOut.strideAt(0);

    for (sd::LongType k = 0; k < K; ++k) {
        const sd::LongType Nk   = cumN[k + 1]   - cumN[k];
        const sd::LongType nnzk = cumNnz[k + 1] - cumNnz[k];

        X* dxkBuf = dXs[k]->bufferAsT<X>();
        const sd::LongType dxks0 = dXs[k]->strideAt(0);
        const sd::LongType dxks1 = dXs[k]->strideAt(1);

        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType i = 0; i < Nk; ++i) {
            for (sd::LongType f = 0; f < F; ++f) {
                dxkBuf[i * dxks0 + f * dxks1] = dxoBuf[(cumN[k] + i) * dxos0 + f * dxos1];
            }
        }

        X* dvkBuf = dVals[k]->bufferAsT<X>();
        const sd::LongType dvks0 = dVals[k]->strideAt(0);
        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType e = 0; e < nnzk; ++e) {
            dvkBuf[e * dvks0] = dvoeBuf[(cumNnz[k] + e) * dvos0];
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
    // Collect all inputs and outputs for synchronization
    std::vector<NDArray*> inputs;
    for (auto* p : Xs)      inputs.push_back(p);
    for (auto* p : vals)    inputs.push_back(p);
    for (auto* p : colIdxs) inputs.push_back(p);
    for (auto* p : rowPtrs) inputs.push_back(p);

    NDArray::preparePrimaryUse({&Xout, &valsOut, &colIdxOut, &rowPtrOut, &batchVec}, inputs);

    BUILD_DOUBLE_SELECTOR(Xout.dataType(), colIdxOut.dataType(),
                          graphDisjointUnionFwd_,
                          (Xs, vals, colIdxs, rowPtrs, Xout, valsOut, colIdxOut, rowPtrOut, batchVec),
                          SD_FLOAT_TYPES, SD_INDEXING_TYPES);

    NDArray::registerPrimaryUse({&Xout, &valsOut, &colIdxOut, &rowPtrOut, &batchVec}, inputs);
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

    NDArray::preparePrimaryUse(outputs, inputs);

    BUILD_SINGLE_SELECTOR(dXout.dataType(), graphDisjointUnionBp_,
                          (cumN, cumNnz, dXout, dValsOut, dXs, dVals),
                          SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse(outputs, inputs);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
