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
// vision_encode_patches: ViT-style patch extraction from images.
// Splits an image into non-overlapping patches and flattens each patch
// into a 1-D feature vector. This is the first step of ViT-style vision
// encoding used in VLMs such as LLaVA, Idefics, and SmolDocling.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_vision_encode_patches)

#include <ops/declarable/headers/vlm.h>
#include <helpers/ShapeUtils.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// vision_encode_patches
//
// Input:
//   0: image  [batch, channels, height, width]  — input image(s)
// Int args:
//   0: patch_size  — size of each square patch (e.g. 14)
//   1: hidden_dim  — unused in the extract step; kept for API symmetry with
//                    downstream projection ops (default: 0 = patch_size^2 * channels)
// Outputs:
//   0: patches [batch, num_patches, patch_size*patch_size*channels]
//   1: num_patches_per_image  — INT64 scalar (= (H/patch_size)*(W/patch_size))
//////////////////////////////////////////////////////////////////////////
template <typename T>
static Status vision_encode_patches_impl(NDArray* image, NDArray* patches, NDArray* numPatchesOut,
                                          int patchSize) {
    const LongType batch    = image->sizeAt(0);
    const LongType channels = image->sizeAt(1);
    const LongType height   = image->sizeAt(2);
    const LongType width    = image->sizeAt(3);

    const LongType numPatchesH = height / patchSize;
    const LongType numPatchesW = width  / patchSize;
    const LongType numPatches  = numPatchesH * numPatchesW;
    const LongType patchDim    = channels * patchSize * patchSize;

    // Write the scalar num_patches output
    double numPatchesVal = static_cast<double>(numPatches);
    numPatchesOut->assign(numPatchesVal);

    auto imageBuf   = image->bufferAsT<T>();
    auto patchesBuf = patches->bufferAsT<T>();

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < batch; ++b) {
        for (LongType ph = 0; ph < numPatchesH; ++ph) {
            for (LongType pw = 0; pw < numPatchesW; ++pw) {
                const LongType patchIdx   = ph * numPatchesW + pw;
                const LongType outOffset  = (b * numPatches + patchIdx) * patchDim;

                for (LongType c = 0; c < channels; ++c) {
                    for (LongType i = 0; i < patchSize; ++i) {
                        const LongType inRow    = ph * patchSize + i;
                        const LongType inBase   = ((b * channels + c) * height + inRow) * width
                                                   + pw * patchSize;
                        const LongType patchBase = (c * patchSize + i) * patchSize;
                        PRAGMA_OMP_SIMD
                        for (LongType j = 0; j < patchSize; ++j) {
                            patchesBuf[outOffset + patchBase + j] = imageBuf[inBase + j];
                        }
                    }
                }
            }
        }
    }

    return Status::OK;
}

CUSTOM_OP_IMPL(vision_encode_patches, 1, 2, false, 0, 2) {
    auto image = INPUT_VARIABLE(0);
    auto patches = OUTPUT_VARIABLE(0);
    auto numPatchesOut = OUTPUT_VARIABLE(1);

    REQUIRE_TRUE(image->rankOf() == 4, 0,
        "vision_encode_patches: expected 4-D image [batch, channels, H, W], got rank %d",
        image->rankOf());

    int patchSize = INT_ARG(0);
    // INT_ARG(1) is hidden_dim — not used in patch extraction

    REQUIRE_TRUE(patchSize > 0, 0,
        "vision_encode_patches: patch_size must be positive, got %d", patchSize);

    const LongType height = image->sizeAt(2);
    const LongType width  = image->sizeAt(3);

    REQUIRE_TRUE(height % patchSize == 0 && width % patchSize == 0, 0,
        "vision_encode_patches: image dimensions (%lld x %lld) must be divisible by "
        "patch_size (%d)", (long long)height, (long long)width, patchSize);

    auto imageType = image->dataType();
    BUILD_SINGLE_SELECTOR(imageType, return vision_encode_patches_impl,
                          (image, patches, numPatchesOut, patchSize), SD_FLOAT_TYPES);
}

DECLARE_SHAPE_FN(vision_encode_patches) {
    auto inShape  = inputShape->at(0);
    auto dtype    = ArrayOptions::dataType(inShape);

    int patchSize = INT_ARG(0);

    const LongType batch    = shape::sizeAt(inShape, static_cast<LongType>(0));
    const LongType channels = shape::sizeAt(inShape, static_cast<LongType>(1));
    const LongType height   = shape::sizeAt(inShape, static_cast<LongType>(2));
    const LongType width    = shape::sizeAt(inShape, static_cast<LongType>(3));

    const LongType numPatches = (height / patchSize) * (width / patchSize);
    const LongType patchDim   = channels * patchSize * patchSize;

    // Output 0: [batch, num_patches, patch_dim]
    auto patchShape = ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, numPatches, patchDim});

    // Output 1: scalar INT64
    auto numPatchesShape = ConstantShapeHelper::getInstance().scalarShapeInfo(DataType::INT64);

    return SHAPELIST(patchShape, numPatchesShape);
}

DECLARE_TYPES(vision_encode_patches) {
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->setAllowedOutputTypes(1, {DataType::INT64});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_vision_encode_patches)
