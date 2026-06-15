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
        auto intermediate = NDArrayFactory::create(input->dataType(), 'c', {1, rank}, input->getContext());
        MmulHelper::mmul(inputRow, loraA, intermediate, 1.0f, 0.0f);

        auto loraOutput = NDArrayFactory::create(input->dataType(), 'c', {1, outFeatures}, input->getContext());
        MmulHelper::mmul(intermediate, loraB, loraOutput, alpha, 0.0f);

        // Add to base output
        auto outputRow = (*output)({b, b+1, 0, 0});
        *outputRow += *loraOutput;

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

#endif
