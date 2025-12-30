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
 * Image preprocessing operation for VLM
 *
 * Normalizes and prepares images for VLM input.
 * Applies channel-wise normalization: (pixel - mean) / std
 *
 * Input: images [B, C, H, W] in range [0, 1] or [0, 255]
 * Output: normalized images [B, C, H, W]
 *
 * Args:
 *   T_ARG(0-2): mean_r, mean_g, mean_b
 *   T_ARG(3-5): std_r, std_g, std_b
 *   INT_ARG(0): input_scale (0 = [0,1], 1 = [0,255])
 */
static void imagePreprocessVlm(NDArray* input, NDArray* output,
                                float meanR, float meanG, float meanB,
                                float stdR, float stdG, float stdB,
                                bool scaleFrom255) {
    vlmUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    const auto batch = input->sizeAt(0);
    const auto channels = input->sizeAt(1);
    const auto height = input->sizeAt(2);
    const auto width = input->sizeAt(3);

    auto inputBuf = input->bufferAsT<float>();
    auto outputBuf = output->bufferAsT<float>();

    float means[3] = {meanR, meanG, meanB};
    float stds[3] = {stdR, stdG, stdB};
    float scale = scaleFrom255 ? 1.0f / 255.0f : 1.0f;

    // Apply normalization per channel
    for (LongType b = 0; b < batch; ++b) {
        for (LongType c = 0; c < channels && c < 3; ++c) {
            float mean = means[c];
            float std = stds[c];

            for (LongType h = 0; h < height; ++h) {
                for (LongType w = 0; w < width; ++w) {
                    LongType idx = ((b * channels + c) * height + h) * width + w;
                    float pixel = inputBuf[idx] * scale;
                    outputBuf[idx] = (pixel - mean) / std;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_image_preprocess, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);   // [B, C, H, W]
    auto output = OUTPUT_VARIABLE(0); // [B, C, H, W]

    if (input->isEmpty()) return sd::Status::OK;

    // Get normalization parameters
    float meanR = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.485f;
    float meanG = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.456f;
    float meanB = block.getTArguments()->size() > 2 ? T_ARG(2) : 0.406f;
    float stdR = block.getTArguments()->size() > 3 ? T_ARG(3) : 0.229f;
    float stdG = block.getTArguments()->size() > 4 ? T_ARG(4) : 0.224f;
    float stdB = block.getTArguments()->size() > 5 ? T_ARG(5) : 0.225f;

    bool scaleFrom255 = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : false;

    imagePreprocessVlm(input, output, meanR, meanG, meanB, stdR, stdG, stdB, scaleFrom255);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_image_preprocess, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM IMAGE_PREPROCESS OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [input, output] {
                return vlmUtils::isSupportedType(input->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
