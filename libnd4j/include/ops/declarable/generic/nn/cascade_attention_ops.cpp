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
#if NOT_EXCLUDED(OP_cascade_attention)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/cascade_attention.h>

namespace sd {
namespace ops {

/**
 * Cascade Attention — chunked attention for long-context KV cache.
 *
 * Splits KV into chunks, computes per-chunk attention, merges with log-sum-exp.
 *
 * Inputs:
 *   0: query  [batch, heads, queryLen, headDim]
 *   1: key    [batch, heads, kvLen, headDim]
 *   2: value  [batch, heads, kvLen, headDim]
 *
 * Output:
 *   0: attention output [batch, heads, queryLen, headDim]
 *
 * Int args:
 *   0: chunkSize (default: 512)
 *
 * Float args:
 *   0: scale (default: 1/sqrt(headDim))
 */
CUSTOM_OP_IMPL(cascade_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(query->rankOf() == 4, 0,
                 "cascade_attention: query must be rank 4 [batch, heads, queryLen, headDim], got %lld",
                 (long long)query->rankOf());
    REQUIRE_TRUE(key->rankOf() == 4, 0,
                 "cascade_attention: key must be rank 4, got %lld",
                 (long long)key->rankOf());
    REQUIRE_TRUE(value->rankOf() == 4, 0,
                 "cascade_attention: value must be rank 4, got %lld",
                 (long long)value->rankOf());

    int chunkSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 512;
    REQUIRE_TRUE(chunkSize > 0, 0, "cascade_attention: chunkSize must be positive, got %d", chunkSize);

    auto headDim = query->sizeAt(3);
    double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : (1.0 / std::sqrt(static_cast<double>(headDim)));

#if defined(SD_CUDA)
    helpers::cascadeAttentionCuda(block.launchContext(), query, key, value, output, chunkSize, scale);
#else
    helpers::cascadeAttentionCpu(block.launchContext(), query, key, value, output, chunkSize, scale);
#endif

    return sd::Status::OK;
}

DECLARE_TYPES(cascade_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(cascade_attention) {
    // Output shape = query shape
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif
