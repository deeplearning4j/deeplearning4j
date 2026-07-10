/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//
// multi_lora_matmul - Batched LoRA GEMM with per-row adapter selection
//
// For multi-LoRA serving: each row in the batch can use a different LoRA adapter.
// Y[i] = X[i] @ W + alpha * X[i] @ A[adapter[i]] @ B[adapter[i]]
//
// CUTLASS grouped GEMM handles the variable-adapter batching efficiently.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_multi_lora_matmul)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(multi_lora_matmul, 5, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);          // [batch, in_features]
    auto baseWeight = INPUT_VARIABLE(1);     // [in_features, out_features]
    auto loraAWeights = INPUT_VARIABLE(2);   // [num_adapters, in_features, rank]
    auto loraBWeights = INPUT_VARIABLE(3);   // [num_adapters, rank, out_features]
    auto adapterIds = INPUT_VARIABLE(4);     // [batch] INT64 adapter ID per row
    auto output = OUTPUT_VARIABLE(0);        // [batch, out_features]

    float alpha = T_ARG_OR(0, 1.0f);        // LoRA scaling factor

    auto batch = input->sizeAt(0);
    auto inFeatures = input->sizeAt(1);
    auto outFeatures = baseWeight->sizeAt(1);
    auto numAdapters = loraAWeights->sizeAt(0);
    auto rank = loraAWeights->sizeAt(2);

    // Base GEMM: Y = X @ W
    MmulHelper::mmul(input, baseWeight, output, 1.0f, 0.0f);

    // LoRA contribution per row
    for (int b = 0; b < batch; ++b) {
        int adapterId = static_cast<int>(adapterIds->e<sd::LongType>(b));
        if (adapterId < 0 || adapterId >= numAdapters) {
            continue;  // No adapter for this row
        }

        // Get this row's input
        auto inputRowSub = (*input)({b, b+1, 0, 0});
        std::vector<sd::LongType> inputRowShape = {1, inFeatures};
        auto inputRow = inputRowSub->reshape('c', inputRowShape);

        // Get adapter weights for this adapter
        auto loraASub = (*loraAWeights)({(sd::LongType)adapterId, (sd::LongType)(adapterId+1), 0, 0, 0, 0});
        std::vector<sd::LongType> loraAShape = {inFeatures, rank};
        auto loraA = loraASub->reshape('c', loraAShape);
        auto loraBSub = (*loraBWeights)({(sd::LongType)adapterId, (sd::LongType)(adapterId+1), 0, 0, 0, 0});
        std::vector<sd::LongType> loraBShape = {rank, outFeatures};
        auto loraB = loraBSub->reshape('c', loraBShape);

        // LoRA: X @ A @ B * alpha
        auto intermediate = NDArrayFactory::create('c', std::vector<sd::LongType>{1, rank}, input->dataType(), input->getContext());
        MmulHelper::mmul(inputRow, loraA, intermediate, 1.0f, 0.0f);

        auto loraOutput = NDArrayFactory::create('c', std::vector<sd::LongType>{1, outFeatures}, input->dataType(), input->getContext());
        MmulHelper::mmul(intermediate, loraB, loraOutput, alpha, 0.0f);

        // Add the LoRA contribution into this output row. Reshape loraOutput to the output
        // row view's EXACT shape so the in-place add is always a same-shape add and never
        // triggers a broadcast (the raw sub-array shape can differ from [1, outFeatures],
        // which made the += throw "broadcast would change the array shape").
        auto outputRow = (*output)({b, b+1, 0, 0});
        auto* outputRowShape = outputRow->getShapeAsVector();
        auto loraRowMatched = loraOutput->reshape('c', *outputRowShape);
        delete outputRowShape;
        *outputRow += *loraRowMatched;
        delete loraRowMatched;

        delete inputRowSub;
        delete inputRow;
        delete loraASub;
        delete loraA;
        delete loraBSub;
        delete loraB;
        delete outputRow;
        delete intermediate;
        delete loraOutput;
    }

    return sd::Status::OK;
}

DECLARE_TYPES(multi_lora_matmul) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})    // input
        ->setAllowedInputTypes(1, {ALL_FLOATS})    // base weight
        ->setAllowedInputTypes(2, {ALL_FLOATS})    // lora A
        ->setAllowedInputTypes(3, {ALL_FLOATS})    // lora B
        ->setAllowedInputTypes(4, {INT64, INT32})  // adapter IDs
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(multi_lora_matmul) {
    auto inShape = inputShape->at(0);
    auto weightShape = inputShape->at(1);

    auto batch = shape::sizeAt(inShape, 0);
    auto outFeatures = shape::sizeAt(weightShape, 1);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inShape), 'c', {batch, outFeatures});

    return SHAPELIST(outputShape);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_multi_lora_matmul)

// ─────────────────────────────────────────────────────────────────────────────
// multi_lora_matmul_bp — Backward pass for multi_lora_matmul.
//
// Forward recap:
//   Y[b] = X[b] @ W  +  alpha * X[b] @ A[id[b]] @ B[id[b]]
//
//   inputs[0] = X           [batch, in_features]
//   inputs[1] = W           [in_features, out_features]  (frozen in QLoRA context)
//   inputs[2] = loraAWeights[num_adapters, in_features, rank]
//   inputs[3] = loraBWeights[num_adapters, rank, out_features]
//   inputs[4] = adapterIds  [batch]  INT32/INT64
//   inputs[5] = gradOut     [batch, out_features]
//
// Outputs (3):
//   0: dX            [batch, in_features]    — gradient w.r.t. input
//   1: dLoraAWeights [num_adapters, in_features, rank]
//   2: dLoraBWeights [num_adapters, rank, out_features]
//
// (No gradient for W or adapterIds — W is frozen, adapterIds is discrete.)
// ─────────────────────────────────────────────────────────────────────────────
#if NOT_EXCLUDED(OP_multi_lora_matmul_bp)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(multi_lora_matmul_bp, 6, 3, false, 0, 0) {
    auto input       = INPUT_VARIABLE(0);  // [batch, in_features]
    auto baseWeight  = INPUT_VARIABLE(1);  // [in_features, out_features]
    auto loraAWeights = INPUT_VARIABLE(2); // [num_adapters, in_features, rank]
    auto loraBWeights = INPUT_VARIABLE(3); // [num_adapters, rank, out_features]
    auto adapterIds  = INPUT_VARIABLE(4);  // [batch] INT32/INT64
    auto gradOut     = INPUT_VARIABLE(5);  // [batch, out_features]

    auto dX          = OUTPUT_VARIABLE(0); // [batch, in_features]
    auto dLoraA      = OUTPUT_VARIABLE(1); // [num_adapters, in_features, rank]
    auto dLoraB      = OUTPUT_VARIABLE(2); // [num_adapters, rank, out_features]

    float alpha = T_ARG_OR(0, 1.0f);

    if (input->isEmpty() || gradOut->isEmpty()) {
        return sd::Status::OK;
    }

    auto batch       = input->sizeAt(0);
    auto inFeatures  = input->sizeAt(1);
    auto outFeatures = baseWeight->sizeAt(1);
    auto numAdapters = loraAWeights->sizeAt(0);
    auto rank        = loraAWeights->sizeAt(2);

    // Initialize gradients to zero
    double zeroD = 0.0;
    dX->assign(zeroD);
    dLoraA->assign(zeroD);
    dLoraB->assign(zeroD);

    // ── Part 1: dX from base path = gradOut @ Wᵀ ─────────────────────────────
    // W is [in_features, out_features], so Wᵀ is [out_features, in_features]
    // dX_base[batch, in_features] = gradOut[batch, out_features] @ Wᵀ
    {
        auto Wt = baseWeight->transpose();  // [out_features, in_features]
        MmulHelper::mmul(gradOut, Wt, dX, 1.0f, 0.0f);
        delete Wt;
    }

    // ── Part 2: per-row LoRA gradients ────────────────────────────────────────
    for (int b = 0; b < batch; ++b) {
        int adapterId = static_cast<int>(adapterIds->e<sd::LongType>(b));
        if (adapterId < 0 || adapterId >= numAdapters) {
            continue;
        }

        // Slice input row: [1, in_features]
        auto inputRowSub = (*input)({b, b+1, 0, 0});
        std::vector<sd::LongType> rowShape = {1, inFeatures};
        auto inputRow = inputRowSub->reshape('c', rowShape);

        // Slice gradOut row: [1, out_features]
        auto gradRowSub = (*gradOut)({b, b+1, 0, 0});
        std::vector<sd::LongType> gradRowShape = {1, outFeatures};
        auto gradRow = gradRowSub->reshape('c', gradRowShape);

        // Get this adapter's A and B slices
        auto loraASub = (*loraAWeights)({(sd::LongType)adapterId, (sd::LongType)(adapterId+1), 0, 0, 0, 0});
        std::vector<sd::LongType> loraAShape = {inFeatures, rank};
        auto loraA = loraASub->reshape('c', loraAShape);  // [in_features, rank]

        auto loraBSub = (*loraBWeights)({(sd::LongType)adapterId, (sd::LongType)(adapterId+1), 0, 0, 0, 0});
        std::vector<sd::LongType> loraBShape = {rank, outFeatures};
        auto loraB = loraBSub->reshape('c', loraBShape);  // [rank, out_features]

        // intermediate = inputRow[1,in_features] @ loraA[in_features,rank] → [1,rank]
        auto intermediate = NDArrayFactory::create('c', std::vector<sd::LongType>{1, rank},
                                                   input->dataType(), input->getContext());
        MmulHelper::mmul(inputRow, loraA, intermediate, 1.0f, 0.0f);

        // ── dX_lora contribution for this row ──
        // dX_lora[1, in_features] = gradRow[1,out_features] @ loraBᵀ[out_features,rank]
        //                           @ loraAᵀ[rank,in_features]
        // Compute step1 = gradRow @ loraBᵀ → [1, rank]
        auto loraBT = loraB->transpose();  // [out_features, rank]
        auto step1 = NDArrayFactory::create('c', std::vector<sd::LongType>{1, rank},
                                             input->dataType(), input->getContext());
        MmulHelper::mmul(gradRow, loraBT, step1, 1.0f, 0.0f);
        delete loraBT;

        // dXrow[1, in_features] = step1[1,rank] @ loraAᵀ[rank, in_features]
        auto loraAT = loraA->transpose();  // [rank, in_features]
        auto dXrow = NDArrayFactory::create('c', std::vector<sd::LongType>{1, inFeatures},
                                             input->dataType(), input->getContext());
        MmulHelper::mmul(step1, loraAT, dXrow, alpha, 0.0f);
        delete loraAT;

        // Add to dX row b — reshape dXrow to the slice's exact shape to avoid any broadcast.
        auto dXRowSlice = (*dX)({b, b+1, 0, 0});
        auto* dXRowSliceShape = dXRowSlice->getShapeAsVector();
        auto dXrowMatched = dXrow->reshape('c', *dXRowSliceShape);
        delete dXRowSliceShape;
        *dXRowSlice += *dXrowMatched;
        delete dXrowMatched;

        delete dXrow;
        delete step1;

        // ── dLoraB contribution: dLoraB[rank, out_features] += alpha * intermediateᵀ @ gradRow
        // Slice out the adapter's dLoraB: [rank, out_features]
        // Accumulate into the RAW adapter slice view so the write reaches dLoraB. Reshaping
        // the slice itself to [rank,out] returns a COPY, so a += on that copy is silently
        // lost (the adapter gradient stays zero); instead reshape the delta to the slice's
        // shape and add into the view.
        auto dLoraBSub = (*dLoraB)({(sd::LongType)adapterId, (sd::LongType)(adapterId+1), 0, 0, 0, 0});

        // delta = intermediateᵀ[rank,1] @ gradRow[1,out_features] → [rank,out_features]
        auto intermediateT = intermediate->transpose();
        auto dLoraBDelta = NDArrayFactory::create('c', std::vector<sd::LongType>{rank, outFeatures},
                                                   input->dataType(), input->getContext());
        MmulHelper::mmul(intermediateT, gradRow, dLoraBDelta, alpha, 0.0f);
        delete intermediateT;

        auto* dLoraBSubShape = dLoraBSub->getShapeAsVector();
        auto dLoraBDeltaMatched = dLoraBDelta->reshape('c', *dLoraBSubShape);
        delete dLoraBSubShape;
        *dLoraBSub += *dLoraBDeltaMatched;
        delete dLoraBDeltaMatched;
        delete dLoraBDelta;
        delete dLoraBSub;

        // ── dLoraA contribution: dLoraA[in_features, rank] += alpha * inputRowᵀ @ step1_saved
        // step1 was deleted above; recompute: step1 = gradRow @ loraBᵀ → [1,rank]
        loraBT = loraB->transpose();
        auto step1b = NDArrayFactory::create('c', std::vector<sd::LongType>{1, rank},
                                              input->dataType(), input->getContext());
        MmulHelper::mmul(gradRow, loraBT, step1b, 1.0f, 0.0f);
        delete loraBT;

        // dLoraA delta: inputRowᵀ[in_features,1] @ step1b[1,rank] → [in_features, rank]
        auto dLoraASub = (*dLoraA)({(sd::LongType)adapterId, (sd::LongType)(adapterId+1), 0, 0, 0, 0});

        auto inputRowT = inputRow->transpose();
        auto dLoraADelta = NDArrayFactory::create('c', std::vector<sd::LongType>{inFeatures, rank},
                                                   input->dataType(), input->getContext());
        MmulHelper::mmul(inputRowT, step1b, dLoraADelta, alpha, 0.0f);
        delete inputRowT;
        delete step1b;

        // Same raw-slice-view accumulation as dLoraB above (a reshaped slice would be a copy).
        auto* dLoraASubShape = dLoraASub->getShapeAsVector();
        auto dLoraADeltaMatched = dLoraADelta->reshape('c', *dLoraASubShape);
        delete dLoraASubShape;
        *dLoraASub += *dLoraADeltaMatched;
        delete dLoraADeltaMatched;
        delete dLoraADelta;
        delete dLoraASub;

        // Cleanup row temporaries
        delete intermediate;
        delete inputRowSub;
        delete inputRow;
        delete gradRowSub;
        delete gradRow;
        delete loraASub;
        delete loraA;
        delete loraBSub;
        delete loraB;
        delete dXRowSlice;
    }

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(multi_lora_matmul_bp) {
    // Outputs: dX, dLoraAWeights, dLoraBWeights
    return SHAPELIST(
        CONSTANT(inputShape->at(0)),   // dX: same as input
        CONSTANT(inputShape->at(2)),   // dLoraA: same as loraAWeights
        CONSTANT(inputShape->at(3))    // dLoraB: same as loraBWeights
    );
}

DECLARE_TYPES(multi_lora_matmul_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})    // input
        ->setAllowedInputTypes(1, {ALL_FLOATS})    // base weight
        ->setAllowedInputTypes(2, {ALL_FLOATS})    // loraAWeights
        ->setAllowedInputTypes(3, {ALL_FLOATS})    // loraBWeights
        ->setAllowedInputTypes(4, {INT64, INT32})  // adapterIds
        ->setAllowedInputTypes(5, {ALL_FLOATS})    // gradOut
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_multi_lora_matmul_bp)
