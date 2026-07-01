/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Helpers for graph_disjoint_union: assembles K variable-size graphs (each with
// node features X_k [N_k, F] and CSR adjacency vals_k/colIdx_k/rowPtr_k) into a
// single block-diagonal batched graph for message-passing without cross-graph leakage.
//
// Forward op outputs:
//   X_combined    [sumN,   F  ]  float  – concatenated node features
//   vals_combined [sumNnz     ]  float  – concatenated edge weights
//   colIdx_combined[sumNnz    ]  int    – col indices offset by cumulative node count
//   rowPtr_combined[sumN+1    ]  int    – row pointers offset by cumulative nnz
//   batchVec      [sumN       ]  int32  – node → graph index (0-based)
//
// Backward (graph_disjoint_union_bp):
//   Splits dX_combined and dVals_combined back into K per-graph gradients.
//   Structural arrays (colIdx, rowPtr) receive zero gradient.
//

#ifndef SAMEDIFF_SPARSE_GRAPH_BATCH_H
#define SAMEDIFF_SPARSE_GRAPH_BATCH_H

#include <ops/declarable/helpers/helpers.h>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Forward: build block-diagonal graph batch.
 *
 * @param Xs         K node-feature matrices, Xs[k] shape [N_k, F]
 * @param vals       K edge-weight arrays, vals[k] shape [nnz_k]
 * @param colIdxs    K column-index arrays, colIdxs[k] shape [nnz_k]  (int)
 * @param rowPtrs    K row-pointer arrays, rowPtrs[k] shape [N_k+1]   (int)
 * @param Xout       output [sumN, F] node features
 * @param valsOut    output [sumNnz] edge weights
 * @param colIdxOut  output [sumNnz] shifted column indices           (int)
 * @param rowPtrOut  output [sumN+1] stitched row pointers            (int)
 * @param batchVec   output [sumN]   node→graph index                 (int32)
 */
SD_LIB_HIDDEN void graph_disjoint_union_fwd(
    const std::vector<NDArray*>& Xs,
    const std::vector<NDArray*>& vals,
    const std::vector<NDArray*>& colIdxs,
    const std::vector<NDArray*>& rowPtrs,
    NDArray& Xout,
    NDArray& valsOut,
    NDArray& colIdxOut,
    NDArray& rowPtrOut,
    NDArray& batchVec);

/**
 * Backward: split combined gradients back into per-graph gradients.
 *
 * @param cumN       cumulative node offsets: cumN[k] = sum N_0..N_{k-1}, cumN[K] = sumN
 * @param cumNnz     cumulative nnz offsets: cumNnz[k] = sum nnz_0..nnz_{k-1}
 * @param dXout      upstream gradient [sumN, F]
 * @param dValsOut   upstream gradient [sumNnz]
 * @param dXs        output: K per-graph node-feature gradients, dXs[k] shape [N_k, F]
 * @param dVals      output: K per-graph edge-weight gradients, dVals[k] shape [nnz_k]
 */
SD_LIB_HIDDEN void graph_disjoint_union_bp(
    const std::vector<sd::LongType>& cumN,
    const std::vector<sd::LongType>& cumNnz,
    NDArray& dXout,
    NDArray& dValsOut,
    std::vector<NDArray*>& dXs,
    std::vector<NDArray*>& dVals);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_GRAPH_BATCH_H
