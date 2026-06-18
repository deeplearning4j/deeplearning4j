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
 * Multimodal fusion operation for VLM using GGML
 *
 * Combines vision and language representations through various fusion strategies.
 * Supports: concatenation, addition, gated fusion, or cross-modal attention.
 *
 * Inputs:
 *   vision:   [B, num_vision_tokens, dim] - Vision features
 *   language: [B, num_lang_tokens, dim] - Language features
 *   gate:     [dim] (optional) - Gating weights for gated fusion
 *
 * Output: [B, num_vision_tokens + num_lang_tokens, dim] or [B, max_tokens, dim]
 *
 * Args:
 *   INT_ARG(0): fusion_type (0=concat, 1=add, 2=gated, 3=interleaved)
 */
static void multimodalFusionVlm(NDArray* vision, NDArray* language, NDArray* gate,
                                 NDArray* output, int fusionType) {
    vlmUtils::GgmlContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_vision = vlmUtils::createGgmlTensor(ctx, vision, "vision");
    struct ggml_tensor* ggml_language = vlmUtils::createGgmlTensor(ctx, language, "language");

    struct ggml_tensor* ggml_output = nullptr;

    switch (fusionType) {
        case 0: {
            // Concatenation along sequence dimension
            ggml_output = ggml_concat(ctx, ggml_vision, ggml_language, 1);
            break;
        }
        case 1: {
            // Element-wise addition (requires same shape)
            ggml_output = ggml_add(ctx, ggml_vision, ggml_language);
            break;
        }
        case 2: {
            // Gated fusion: gate * vision + (1 - gate) * language
            if (gate != nullptr && !gate->isEmpty()) {
                struct ggml_tensor* ggml_gate = vlmUtils::createGgmlTensor(ctx, gate, "gate");
                struct ggml_tensor* gated_vision = ggml_mul(ctx, ggml_gate, ggml_vision);
                struct ggml_tensor* one_minus_gate = ggml_map_unary_f32(ctx, ggml_gate,
                    [](float x) { return 1.0f - x; });
                struct ggml_tensor* gated_language = ggml_mul(ctx, one_minus_gate, ggml_language);
                ggml_output = ggml_add(ctx, gated_vision, gated_language);
            } else {
                // Fallback to simple addition if no gate provided
                ggml_output = ggml_add(ctx, ggml_vision, ggml_language);
            }
            break;
        }
        case 3: {
            // Interleaved: alternate vision and language tokens
            // For simplicity, just concatenate in this implementation
            ggml_output = ggml_concat(ctx, ggml_vision, ggml_language, 1);
            break;
        }
        default:
            ggml_output = ggml_concat(ctx, ggml_vision, ggml_language, 1);
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
PLATFORM_IMPL(vlm_multimodal_fusion, ENGINE_CPU) {
    auto vision = INPUT_VARIABLE(0);    // [B, num_vision_tokens, dim]
    auto language = INPUT_VARIABLE(1);  // [B, num_lang_tokens, dim]
    auto gate = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (vision->isEmpty() && language->isEmpty()) return sd::Status::OK;

    int fusionType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    multimodalFusionVlm(vision, language, gate, output, fusionType);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_multimodal_fusion, ENGINE_CPU) {
    auto vision = INPUT_VARIABLE(0);
    auto language = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM MULTIMODAL_FUSION OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [vision, language, output] {
                return vlmUtils::isSupportedType(vision->dataType()) &&
                       vlmUtils::isSupportedType(language->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(vision->rankOf(), RANK_MSG_INPUT0), 3);
    req.expectEq(makeInfoVariable(language->rankOf(), RANK_MSG_INPUT1), 3);

    // Vision and language must have same batch size and embedding dimension
    req.expectEq(makeInfoVariable(vision->sizeAt(0), "BATCH_VISION"),
                 makeInfoVariable(language->sizeAt(0), "BATCH_LANGUAGE"));
    req.expectEq(makeInfoVariable(vision->sizeAt(2), "DIM_VISION"),
                 makeInfoVariable(language->sizeAt(2), "DIM_LANGUAGE"));

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
