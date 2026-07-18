/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//
// quantize_q4_0 / quantize_q8_0 — pack a float tensor into raw GGML block bytes.
// Exact inverse of ggml_dequantize. Output is a flat UINT8 byte buffer.
//

#include <system/op_boilerplate.h>

#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/ggml_quantize.h>

namespace sd {
namespace ops {

static constexpr int GGML_QK = 32;

static Status quantizeImpl(graph::Context& block, NDArray* input, NDArray* output, int quantType) {
    const LongType n = input->lengthOf();
    REQUIRE_TRUE(n % GGML_QK == 0, 0,
                 "quantize: element count (%lld) must be a multiple of %d", (long long)n, GGML_QK);
    if (input->isEmpty()) return Status::OK;

    NDArray* xf = input->dataType() == DataType::FLOAT32 ? input : input->cast(DataType::FLOAT32);
    helpers::ggmlQuantize(block.launchContext(), xf, output, quantType);
    if (xf != input) delete xf;
    return Status::OK;
}

static sd::ShapeList* quantizeShape(const ShapeList* inputShape, int blockBytes) {
    auto inShape = inputShape->at(0);
    const LongType n = shape::length(inShape);
    const LongType numBlocks = n / GGML_QK;
    std::vector<sd::LongType> outShape = {numBlocks * blockBytes};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(DataType::UINT8, 'c', outShape));
}

#if NOT_EXCLUDED(OP_quantize_q4_0)
CUSTOM_OP_IMPL(quantize_q4_0, 1, 1, false, 0, 0) {
    return quantizeImpl(block, INPUT_VARIABLE(0), OUTPUT_VARIABLE(0), 0);  // GGML_QUANT_Q4_0
}
DECLARE_TYPES(quantize_q4_0) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({DataType::UINT8})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(quantize_q4_0) { return quantizeShape(inputShape, 18); }
#endif

#if NOT_EXCLUDED(OP_quantize_q8_0)
CUSTOM_OP_IMPL(quantize_q8_0, 1, 1, false, 0, 0) {
    return quantizeImpl(block, INPUT_VARIABLE(0), OUTPUT_VARIABLE(0), 4);  // GGML_QUANT_Q8_0
}
DECLARE_TYPES(quantize_q8_0) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({DataType::UINT8})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(quantize_q8_0) { return quantizeShape(inputShape, 34); }
#endif

}  // namespace ops
}  // namespace sd
