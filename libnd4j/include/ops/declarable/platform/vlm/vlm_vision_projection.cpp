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
 * Vision projection operation for VLM using GGML
 *
 * Projects vision features from vision encoder space to language model space.
 * This is a key component in VLMs like LLaVA that bridges the vision-language gap.
 *
 * Inputs:
 *   vision_features: [B, num_patches, vision_dim] - Features from vision encoder
 *   projection_weights: [language_dim, vision_dim] - Projection matrix
 *   projection_bias: [language_dim] (optional) - Projection bias
 *
 * Output: [B, num_patches, language_dim] - Projected features
 *
 * Args:
 *   INT_ARG(0): activation_type (0=none, 1=gelu, 2=silu, 3=relu)
 */
static void visionProjectionVlm(NDArray* visionFeatures, NDArray* weights, NDArray* bias,
                                 NDArray* output, int activationType) {
    vlmUtils::GgmlContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_vision = vlmUtils::createGgmlTensor(ctx, visionFeatures, "vision");
    struct ggml_tensor* ggml_weights = vlmUtils::createGgmlTensor(ctx, weights, "weights");

    // Linear projection: vision @ weights.T
    struct ggml_tensor* projected = ggml_mul_mat(ctx, ggml_weights, ggml_vision);

    // Add bias if provided
    if (bias != nullptr && !bias->isEmpty()) {
        struct ggml_tensor* ggml_bias = vlmUtils::createGgmlTensor(ctx, bias, "bias");
        projected = ggml_add(ctx, projected, ggml_bias);
    }

    // Apply activation
    struct ggml_tensor* ggml_output = projected;
    switch (activationType) {
        case 1:  // GELU
            ggml_output = ggml_gelu(ctx, projected);
            break;
        case 2:  // SiLU
            ggml_output = ggml_silu(ctx, projected);
            break;
        case 3:  // ReLU
            ggml_output = ggml_relu(ctx, projected);
            break;
        default:
            // No activation
            break;
    }

    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    vlmUtils::executeGgmlGraph(ctx, graph);

    vlmUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_vision_projection, ENGINE_CPU) {
    auto visionFeatures = INPUT_VARIABLE(0);  // [B, num_patches, vision_dim]
    auto weights = INPUT_VARIABLE(1);          // [language_dim, vision_dim]
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);          // [B, num_patches, language_dim]

    if (visionFeatures->isEmpty()) return sd::Status::OK;

    int activationType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    visionProjectionVlm(visionFeatures, weights, bias, output, activationType);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_vision_projection, ENGINE_CPU) {
    auto visionFeatures = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM VISION_PROJECTION OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [visionFeatures, weights, output] {
                return vlmUtils::isSupportedType(visionFeatures->dataType()) &&
                       vlmUtils::isSupportedType(weights->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(visionFeatures->rankOf(), RANK_MSG_INPUT0), 3);
    req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), 2);

    // Vision dim must match weights input dim
    req.expectEq(makeInfoVariable(visionFeatures->sizeAt(2), "VISION_DIM"),
                 makeInfoVariable(weights->sizeAt(1), "WEIGHTS_IN_DIM"));

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
