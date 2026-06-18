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

#include <cmath>

namespace sd {
namespace ops {
namespace platforms {

/**
 * 2D Position Encoding for Vision Transformers
 *
 * Applies 2D sinusoidal positional encoding to image patch embeddings.
 * Unlike 1D position encoding, this encodes both row and column positions.
 *
 * Input: patch embeddings [B, num_patches, dim]
 * Output: embeddings with position encoding [B, num_patches, dim]
 *
 * Args:
 *   INT_ARG(0): grid_height (number of patches in height)
 *   INT_ARG(1): grid_width (number of patches in width)
 *   T_ARG(0): temperature (default 10000.0)
 *   INT_ARG(2): encoding_type (0=sinusoidal, 1=learnable placeholder)
 */
static void position2DEncodeVlm(NDArray* patches, NDArray* output,
                                 int gridHeight, int gridWidth, float temperature) {
    const auto batch = patches->sizeAt(0);
    const auto numPatches = patches->sizeAt(1);
    const auto dim = patches->sizeAt(2);

    // Copy input to output first
    output->assign(patches);

    auto outputBuf = output->bufferAsT<float>();

    // Generate 2D sinusoidal position encoding
    const int halfDim = dim / 2;
    const int quarterDim = halfDim / 2;

    for (LongType b = 0; b < batch; ++b) {
        for (LongType p = 0; p < numPatches; ++p) {
            int py = p / gridWidth;   // Row position
            int px = p % gridWidth;   // Column position

            LongType outOffset = (b * numPatches + p) * dim;

            // Height position encoding (first half of first half)
            for (int i = 0; i < quarterDim; ++i) {
                float freq = 1.0f / std::pow(temperature, (2.0f * i) / quarterDim);
                outputBuf[outOffset + i] += std::sin(py * freq);
                outputBuf[outOffset + quarterDim + i] += std::cos(py * freq);
            }

            // Width position encoding (second half of first half and beyond)
            for (int i = 0; i < quarterDim; ++i) {
                float freq = 1.0f / std::pow(temperature, (2.0f * i) / quarterDim);
                outputBuf[outOffset + halfDim + i] += std::sin(px * freq);
                outputBuf[outOffset + halfDim + quarterDim + i] += std::cos(px * freq);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_2d_position_encode, ENGINE_CPU) {
    auto patches = INPUT_VARIABLE(0);  // [B, num_patches, dim]
    auto output = OUTPUT_VARIABLE(0);  // [B, num_patches, dim]

    if (patches->isEmpty()) return sd::Status::OK;

    int gridHeight = INT_ARG(0);
    int gridWidth = INT_ARG(1);
    float temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;

    position2DEncodeVlm(patches, output, gridHeight, gridWidth, temperature);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_2d_position_encode, ENGINE_CPU) {
    auto patches = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM 2D_POSITION_ENCODE OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [patches, output] {
                return vlmUtils::isSupportedType(patches->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(patches->rankOf(), RANK_MSG_INPUT0), 3);

    // Verify num_patches matches grid_height * grid_width
    if (block.getIArguments()->size() >= 2) {
        int gridHeight = INT_ARG(0);
        int gridWidth = INT_ARG(1);
        req.expectEq(makeInfoVariable(patches->sizeAt(1), "NUM_PATCHES"),
                     makeInfoVariable(static_cast<LongType>(gridHeight * gridWidth), "GRID_SIZE"));
    }

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
