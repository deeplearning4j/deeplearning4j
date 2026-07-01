/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// graph_disjoint_union — builds block-diagonal (disjoint) batch from K variable-size graphs.
//
// Input layout (4*K tensors):
//   [0 .. K-1]       X_k     [N_k, F]   float  — node features per graph
//   [K .. 2K-1]      vals_k  [nnz_k]    float  — CSR edge weights per graph
//   [2K .. 3K-1]     colIdx_k[nnz_k]    int    — CSR column indices per graph
//   [3K .. 4K-1]     rowPtr_k[N_k+1]    int    — CSR row pointers per graph
// IArgs:
//   [0] K  — number of graphs in the batch
// Outputs:
//   [0] X_combined     [sumN,   F ]  float
//   [1] vals_combined  [sumNnz   ]   float
//   [2] colIdx_combined[sumNnz   ]   int    (column indices shifted by cumulative node count)
//   [3] rowPtr_combined[sumN+1   ]   int    (row pointers shifted by cumulative nnz)
//   [4] batchVec       [sumN     ]   INT32  (node → graph index, 0-based)
//
// graph_disjoint_union_bp — backward pass.
//
// Inputs (4*K + 2):
//   [0..K-1]         X_k     (forward inputs, shapes needed)
//   [K..2K-1]        vals_k  (forward inputs, shapes needed)
//   [2K..3K-1]       colIdx_k (structural, shape only)
//   [3K..4K-1]       rowPtr_k (structural, shape only)
//   [4K]             dX_combined  [sumN, F]  upstream grad
//   [4K+1]           dVals_combined[sumNnz]   upstream grad
// IArgs:
//   [0] K
// Outputs: K dX_k [N_k, F], then K dVals_k [nnz_k]
//   (structural inputs colIdx, rowPtr receive zero gradients, not output)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_graph_disjoint_union)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_graph_batch.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_IMPL(graph_disjoint_union, -2, 5, false, 0, 1) {
    const LongType K = INT_ARG(0);
    REQUIRE_TRUE(K >= 1, 0, "graph_disjoint_union: K must be >= 1, got %lld", (long long)K);
    REQUIRE_TRUE(block.width() == 4 * K, 0,
                 "graph_disjoint_union: expected 4*K=%lld inputs, got %d",
                 (long long)(4 * K), block.width());

    // Partition inputs
    std::vector<NDArray*> Xs, vals, colIdxs, rowPtrs;
    for (LongType k = 0; k < K; ++k) {
        Xs.push_back(INPUT_VARIABLE(k));
        vals.push_back(INPUT_VARIABLE(K + k));
        colIdxs.push_back(INPUT_VARIABLE(2 * K + k));
        rowPtrs.push_back(INPUT_VARIABLE(3 * K + k));
    }

    auto Xout     = OUTPUT_VARIABLE(0);
    auto valsOut  = OUTPUT_VARIABLE(1);
    auto colIdxOut = OUTPUT_VARIABLE(2);
    auto rowPtrOut = OUTPUT_VARIABLE(3);
    auto batchVec  = OUTPUT_VARIABLE(4);

    rowPtrOut->nullify();
    batchVec->nullify();

    helpers::graph_disjoint_union_fwd(
        Xs, vals, colIdxs, rowPtrs,
        *Xout, *valsOut, *colIdxOut, *rowPtrOut, *batchVec);

    return Status::OK;
}

DECLARE_SHAPE_FN(graph_disjoint_union) {
    const LongType K = INT_ARG(0);
    REQUIRE_TRUE(K >= 1, 0, "graph_disjoint_union: K must be >= 1");

    const LongType F = shape::sizeAt(inputShape->at(0), 1);  // feature dim from X_0
    auto floatDtype = ArrayOptions::dataType(inputShape->at(0));
    auto idxDtype   = ArrayOptions::dataType(inputShape->at(2 * K));  // colIdx_0 dtype

    LongType sumN   = 0;
    LongType sumNnz = 0;
    for (LongType k = 0; k < K; ++k) {
        sumN   += shape::sizeAt(inputShape->at(k), 0);          // X_k rows
        sumNnz += shape::length(inputShape->at(K + k));         // vals_k length
    }

    auto sh0 = ConstantShapeHelper::getInstance().createShapeInfo(floatDtype, 'c', {sumN,   F});
    auto sh1 = ConstantShapeHelper::getInstance().createShapeInfo(floatDtype, 'c', {sumNnz});
    auto sh2 = ConstantShapeHelper::getInstance().createShapeInfo(idxDtype,   'c', {sumNnz});
    auto sh3 = ConstantShapeHelper::getInstance().createShapeInfo(idxDtype,   'c', {sumN + 1});
    auto sh4 = ConstantShapeHelper::getInstance().createShapeInfo(DataType::INT32, 'c', {sumN});

    return SHAPELIST(sh0, sh1, sh2, sh3, sh4);
}

DECLARE_TYPES(graph_disjoint_union) {
    getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(ANY);
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_IMPL(graph_disjoint_union_bp, -2, -2, false, 0, 1) {
    const LongType K = INT_ARG(0);
    REQUIRE_TRUE(K >= 1, 0, "graph_disjoint_union_bp: K must be >= 1");

    // Upstream gradients are at positions 4K and 4K+1
    auto dXout    = INPUT_VARIABLE(4 * K);
    auto dValsOut = INPUT_VARIABLE(4 * K + 1);

    // Build cumulative node / nnz vectors from forward input shapes
    std::vector<LongType> cumN(K + 1, 0);
    std::vector<LongType> cumNnz(K + 1, 0);
    for (LongType k = 0; k < K; ++k) {
        cumN[k + 1]   = cumN[k]   + INPUT_VARIABLE(k)->sizeAt(0);
        cumNnz[k + 1] = cumNnz[k] + INPUT_VARIABLE(K + k)->lengthOf();
    }

    // Output gradients
    std::vector<NDArray*> dXs, dVals;
    for (LongType k = 0; k < K; ++k) {
        dXs.push_back(OUTPUT_NULLIFIED(k));
        dVals.push_back(OUTPUT_NULLIFIED(K + k));
    }

    helpers::graph_disjoint_union_bp(cumN, cumNnz, *dXout, *dValsOut, dXs, dVals);
    return Status::OK;
}

DECLARE_SHAPE_FN(graph_disjoint_union_bp) {
    const LongType K = INT_ARG(0);
    auto shapeList = SHAPELIST();
    // dXs: same shapes as X_k inputs
    for (LongType k = 0; k < K; ++k) {
        shapeList->push_back(CONSTANT(inputShape->at(k)));
    }
    // dVals: same shapes as vals_k inputs
    for (LongType k = 0; k < K; ++k) {
        shapeList->push_back(CONSTANT(inputShape->at(K + k)));
    }
    return shapeList;
}

DECLARE_TYPES(graph_disjoint_union_bp) {
    getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(ANY);
}

}  // namespace ops
}  // namespace sd
#endif  // NOT_EXCLUDED(OP_graph_disjoint_union)
