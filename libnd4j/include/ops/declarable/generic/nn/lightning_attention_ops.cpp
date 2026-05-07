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
#if NOT_EXCLUDED(OP_lightning_attention)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/lightning_attention.h>

namespace sd {
namespace ops {

/**
 * Lightning Attention — O(N) linear attention using cumulative sum recurrence.
 *
 * Instead of computing the full N×N attention matrix, linear attention
 * uses a running state accumulator:
 *   S_i = sum_{j<=i} (K_j^T * V_j)
 *   O_i = scale * Q_i @ S_i
 *
 * Complexity: O(N * d²) instead of O(N² * d).
 *
 * Inputs:
 *   0: query  [batch, heads, seqLen, headDim]
 *   1: key    [batch, heads, seqLen, headDim]
 *   2: value  [batch, heads, seqLen, headDim]
 *
 * Output:
 *   0: attention output [batch, heads, seqLen, headDim]
 *
 * Float args:
 *   0: scale (default: 1/sqrt(headDim))
 */
CUSTOM_OP_IMPL(lightning_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(query->rankOf() == 4, 0,
                 "lightning_attention: query must be rank 4 [batch, heads, seqLen, headDim], got %lld",
                 (long long)query->rankOf());
    REQUIRE_TRUE(key->rankOf() == 4, 0,
                 "lightning_attention: key must be rank 4, got %lld",
                 (long long)key->rankOf());
    REQUIRE_TRUE(value->rankOf() == 4, 0,
                 "lightning_attention: value must be rank 4, got %lld",
                 (long long)value->rankOf());

    auto headDim = query->sizeAt(3);
    double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : (1.0 / std::sqrt(static_cast<double>(headDim)));

#if defined(SD_CUDA)
    helpers::lightningAttentionCuda(block.launchContext(), query, key, value, output, scale);
#else
    helpers::lightningAttentionCpu(block.launchContext(), query, key, value, output, scale);
#endif

    return sd::Status::OK;
}

DECLARE_TYPES(lightning_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(lightning_attention) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif
