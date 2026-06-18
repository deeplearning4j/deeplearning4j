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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "vlmUtils.h"

#if HAVE_VLM

namespace sd {
namespace ops {
namespace platforms {

/**
 * Vision encoder operation using GGML
 *
 * Encodes image features through a vision transformer backbone.
 * Input: image patches [B, num_patches, patch_dim]
 * Output: encoded features [B, num_patches, hidden_dim]
 */
static void visionEncodeVlm(NDArray* patches, NDArray* weights, NDArray* bias,
                            NDArray* output, int numHeads, int hiddenDim) {
    vlmUtils::GgmlContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_patches = vlmUtils::createGgmlTensor(ctx, patches, "patches");
    struct ggml_tensor* ggml_weights = vlmUtils::createGgmlTensor(ctx, weights, "weights");

    // Linear projection: patches * weights
    struct ggml_tensor* projected = ggml_mul_mat(ctx, ggml_weights, ggml_patches);

    // Add bias if provided
    struct ggml_tensor* ggml_output = projected;
    if (bias != nullptr && !bias->isEmpty()) {
        struct ggml_tensor* ggml_bias = vlmUtils::createGgmlTensor(ctx, bias, "bias");
        ggml_output = ggml_add(ctx, projected, ggml_bias);
    }

    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    vlmUtils::executeGgmlGraph(ctx, graph);

    vlmUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_vision_encode, ENGINE_CPU) {
    auto patches = INPUT_VARIABLE(0);   // [B, num_patches, patch_dim]
    auto weights = INPUT_VARIABLE(1);   // [hidden_dim, patch_dim]
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);   // [B, num_patches, hidden_dim]

    if (patches->isEmpty()) return sd::Status::OK;

    int numHeads = INT_ARG(0);
    int hiddenDim = weights->sizeAt(0);

    visionEncodeVlm(patches, weights, bias, output, numHeads, hiddenDim);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_vision_encode, ENGINE_CPU) {
    auto patches = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM VISION_ENCODE OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [patches, weights, output] {
                return vlmUtils::isSupportedType(patches->dataType()) &&
                       vlmUtils::isSupportedType(weights->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(patches->rankOf(), RANK_MSG_INPUT0), 3);
    req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), 2);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
