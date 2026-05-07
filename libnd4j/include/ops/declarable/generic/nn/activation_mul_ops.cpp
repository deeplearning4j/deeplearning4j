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

#include <system/op_boilerplate.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/activation_mul.h>

namespace sd {
namespace ops {

// ─── silu_and_mul ───────────────────────────────────────────────────────────

#if NOT_EXCLUDED(OP_silu_and_mul)

/**
 * silu_and_mul - Fused SiLU (Swish) activation with element-wise multiply.
 *
 * Computes: output = silu(gate) * up = (gate * sigmoid(gate)) * up
 *
 * This is the core SwiGLU computation used in LLaMA, Qwen, Gemma, Mistral.
 *
 * Input:
 *   0: gate [batch, ..., dim] — gate projection output
 *   1: up   [batch, ..., dim] — up projection output (same shape)
 *
 * Output:
 *   0: result [batch, ..., dim] — silu(gate) * up
 */
CUSTOM_OP_IMPL(silu_and_mul, 2, 1, false, 0, 0) {
    auto gate = INPUT_VARIABLE(0);
    auto up = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(gate->isSameShape(up), 0,
                 "silu_and_mul: gate and up must have same shape, got %s vs %s",
                 ShapeUtils::shapeAsString(gate).c_str(),
                 ShapeUtils::shapeAsString(up).c_str());

    helpers::siluAndMul(block.launchContext(), gate, up, output);
    return sd::Status::OK;
}

DECLARE_TYPES(silu_and_mul) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(silu_and_mul) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

#endif

// ─── gelu_and_mul ───────────────────────────────────────────────────────────

#if NOT_EXCLUDED(OP_gelu_and_mul)

/**
 * gelu_and_mul - Fused GELU activation with element-wise multiply.
 *
 * Computes: output = gelu(gate) * up
 * Uses exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * Used in GEGLU architectures (GPT-NeoX, Phi).
 *
 * Input:
 *   0: gate [batch, ..., dim]
 *   1: up   [batch, ..., dim]
 *
 * Output:
 *   0: result [batch, ..., dim]
 *
 * Int args:
 *   0: useTanhApprox (0=exact GELU, 1=tanh approximation, default: 0)
 */
CUSTOM_OP_IMPL(gelu_and_mul, 2, 1, false, 0, 0) {
    auto gate = INPUT_VARIABLE(0);
    auto up = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(gate->isSameShape(up), 0,
                 "gelu_and_mul: gate and up must have same shape, got %s vs %s",
                 ShapeUtils::shapeAsString(gate).c_str(),
                 ShapeUtils::shapeAsString(up).c_str());

    int useTanh = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    if (useTanh) {
        helpers::geluTanhAndMul(block.launchContext(), gate, up, output);
    } else {
        helpers::geluAndMul(block.launchContext(), gate, up, output);
    }

    return sd::Status::OK;
}

DECLARE_TYPES(gelu_and_mul) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(gelu_and_mul) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

#endif

}  // namespace ops
}  // namespace sd
