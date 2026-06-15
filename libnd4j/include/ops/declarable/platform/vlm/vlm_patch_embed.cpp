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
 * Patch embedding operation
 *
 * Extracts patches from images and projects them to embeddings.
 * This is typically used as the first layer in Vision Transformers.
 *
 * Input: images [B, C, H, W] in NCHW format
 * Weights: convolution weights [embed_dim, C, patch_size, patch_size]
 * Bias: optional bias [embed_dim]
 * Output: patch embeddings [B, num_patches, embed_dim]
 *
 * Args:
 *   INT_ARG(0): patch_size (e.g., 14, 16, 32)
 *   INT_ARG(1): stride (usually same as patch_size)
 */
static void patchEmbedVlm(NDArray* images, NDArray* weights, NDArray* bias,
                          NDArray* output, int patchSize, int stride) {
    vlmUtils::GgmlContextGuard ctx(256 * 1024 * 1024);

    // Get dimensions
    const auto batch = images->sizeAt(0);
    const auto channels = images->sizeAt(1);
    const auto height = images->sizeAt(2);
    const auto width = images->sizeAt(3);
    const auto embedDim = weights->sizeAt(0);

    // Create GGML tensors
    struct ggml_tensor* ggml_images = vlmUtils::createGgmlTensor(ctx, images, "images");
    struct ggml_tensor* ggml_weights = vlmUtils::createGgmlTensor(ctx, weights, "weights");

    // Perform 2D convolution with stride=patch_size (non-overlapping patches)
    struct ggml_tensor* conv_output = ggml_conv_2d(ctx, ggml_weights, ggml_images,
                                                    stride, stride,  // stride
                                                    0, 0,            // padding
                                                    1, 1);           // dilation

    // Add bias if provided
    struct ggml_tensor* ggml_output = conv_output;
    if (bias != nullptr && !bias->isEmpty()) {
        struct ggml_tensor* ggml_bias = vlmUtils::createGgmlTensor(ctx, bias, "bias");
        ggml_output = ggml_add(ctx, conv_output, ggml_bias);
    }

    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    vlmUtils::executeGgmlGraph(ctx, graph);

    vlmUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_patch_embed, ENGINE_CPU) {
    auto images = INPUT_VARIABLE(0);    // [B, C, H, W]
    auto weights = INPUT_VARIABLE(1);   // [embed_dim, C, patch_size, patch_size]
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);   // [B, num_patches, embed_dim]

    if (images->isEmpty()) return sd::Status::OK;

    int patchSize = INT_ARG(0);
    int stride = block.getIArguments()->size() > 1 ? INT_ARG(1) : patchSize;

    patchEmbedVlm(images, weights, bias, output, patchSize, stride);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_patch_embed, ENGINE_CPU) {
    auto images = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM PATCH_EMBED OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [images, weights, output] {
                return vlmUtils::isSupportedType(images->dataType()) &&
                       vlmUtils::isSupportedType(weights->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(images->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
