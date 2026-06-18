/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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
// Per-layer KL divergence op for adaptive GGUF quantization sensitivity analysis.
//
// Computes KL(P||Q) where P = softmax(ref/T) and Q = softmax(quant/T),
// then averages over all rows (batch * seq positions).
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_kl_divergence_per_layer)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/kl_divergence_per_layer.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(kl_divergence_per_layer, 2, 1, false, 1, 0) {
    auto referenceLogits = INPUT_VARIABLE(0);
    auto quantizedLogits = INPUT_VARIABLE(1);
    auto output          = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(referenceLogits->rankOf() >= 2, 0,
                 "kl_divergence_per_layer: reference_logits must have rank >= 2, got %d",
                 referenceLogits->rankOf());
    REQUIRE_TRUE(quantizedLogits->rankOf() >= 2, 0,
                 "kl_divergence_per_layer: quantized_logits must have rank >= 2, got %d",
                 quantizedLogits->rankOf());
    REQUIRE_TRUE(referenceLogits->isSameShape(quantizedLogits), 0,
                 "kl_divergence_per_layer: reference and quantized logits must have the same shape");

    double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
    if (temperature <= 0.0) temperature = 1.0;

    helpers::klDivergencePerLayer(referenceLogits, quantizedLogits, output,
                                   temperature, block.launchContext());
    return sd::Status::OK;
}

DECLARE_TYPES(kl_divergence_per_layer) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(kl_divergence_per_layer) {
    // Output is always a scalar of the same float type as input
    auto dt = sd::ArrayOptions::dataType(inputShape->at(0));
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(dt));
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_kl_divergence_per_layer)
