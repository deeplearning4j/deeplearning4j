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
 * Image embedding operation using GGML
 *
 * Converts image patches to embeddings via linear projection.
 * Input: image patches [B, num_patches, patch_dim]
 * Weights: embedding matrix [embed_dim, patch_dim]
 * Output: embeddings [B, num_patches, embed_dim]
 */
static void imageEmbedVlm(NDArray* patches, NDArray* weights, NDArray* output) {
    vlmUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_patches = vlmUtils::createGgmlTensor(ctx, patches, "patches");
    struct ggml_tensor* ggml_weights = vlmUtils::createGgmlTensor(ctx, weights, "weights");

    // Matrix multiplication: patches @ weights.T
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_weights, ggml_patches);
    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    vlmUtils::executeGgmlGraph(ctx, graph);

    vlmUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_image_embed, ENGINE_CPU) {
    auto patches = INPUT_VARIABLE(0);   // [B, num_patches, patch_dim]
    auto weights = INPUT_VARIABLE(1);   // [embed_dim, patch_dim]
    auto output = OUTPUT_VARIABLE(0);   // [B, num_patches, embed_dim]

    if (patches->isEmpty()) return sd::Status::OK;

    imageEmbedVlm(patches, weights, output);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_image_embed, ENGINE_CPU) {
    auto patches = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM IMAGE_EMBED OP");

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
