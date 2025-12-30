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
// Generic (fallback) implementations for VLM operations.
// These are used when platform-specific helpers (GGML/VLM) are not available.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_vlm_vision_encode) || NOT_EXCLUDED(OP_vlm_image_embed) || \
    NOT_EXCLUDED(OP_vlm_patch_embed) || NOT_EXCLUDED(OP_vlm_cross_attention) || \
    NOT_EXCLUDED(OP_vlm_multimodal_fusion) || NOT_EXCLUDED(OP_vlm_vision_projection) || \
    NOT_EXCLUDED(OP_vlm_image_preprocess) || NOT_EXCLUDED(OP_vlm_2d_position_encode)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/vlm.h>
#include <helpers/MmulHelper.h>
#include <cmath>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// vlm_vision_encode - Generic implementation using layer normalization
#if NOT_EXCLUDED(OP_vlm_vision_encode)
CUSTOM_OP_IMPL(vlm_vision_encode, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get parameters
    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    // Simple layer normalization as fallback
    // Normalize along the last dimension
    int axis = input->rankOf() - 1;

    auto mean = input->reduceAlongDimension(reduce::Mean, {axis}, true);
    auto variance = input->varianceAlongDimension(variance::SummaryStatsVariance, false, {axis});
    variance.reshapei(mean.shapeInfo());

    // output = (input - mean) / sqrt(variance + eps)
    auto centered = *input - mean;
    auto stddev = (variance + eps).transform(transform::Sqrt);
    output->assign(centered / stddev);

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_vision_encode) {
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
}

DECLARE_TYPES(vlm_vision_encode) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_image_embed - Reshape and project patches
#if NOT_EXCLUDED(OP_vlm_image_embed)
CUSTOM_OP_IMPL(vlm_image_embed, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int patchSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 16;
    int embeddingDim = block.getIArguments()->size() > 1 ? INT_ARG(1) : 768;

    // Input shape: [batch, channels, height, width]
    auto batch = input->sizeAt(0);
    auto channels = input->sizeAt(1);
    auto height = input->sizeAt(2);
    auto width = input->sizeAt(3);

    auto numPatchesH = height / patchSize;
    auto numPatchesW = width / patchSize;
    auto numPatches = numPatchesH * numPatchesW;
    auto patchDim = channels * patchSize * patchSize;

    // For the generic implementation, just reshape
    // Platform helpers will do proper projection
    output->assign(input->reshape(input->ordering(), {batch, numPatches, patchDim}));

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_image_embed) {
    auto inputShape = inputShape->at(0);
    auto dtype = ArrayOptions::dataType(inputShape);

    int patchSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 16;
    int embeddingDim = block.getIArguments()->size() > 1 ? INT_ARG(1) : 768;

    auto batch = shape::sizeAt(inputShape, static_cast<LongType>(0));
    auto channels = shape::sizeAt(inputShape, static_cast<LongType>(1));
    auto height = shape::sizeAt(inputShape, static_cast<LongType>(2));
    auto width = shape::sizeAt(inputShape, static_cast<LongType>(3));

    auto numPatches = (height / patchSize) * (width / patchSize);
    auto patchDim = channels * patchSize * patchSize;

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, numPatches, patchDim}));
}

DECLARE_TYPES(vlm_image_embed) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_patch_embed - Extract and embed patches
#if NOT_EXCLUDED(OP_vlm_patch_embed)
CUSTOM_OP_IMPL(vlm_patch_embed, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int patchSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 16;
    int stride = block.getIArguments()->size() > 1 ? INT_ARG(1) : patchSize;
    bool includeCls = block.getIArguments()->size() > 2 ? INT_ARG(2) != 0 : true;

    auto batch = input->sizeAt(0);
    auto channels = input->sizeAt(1);
    auto height = input->sizeAt(2);
    auto width = input->sizeAt(3);

    auto numPatchesH = (height - patchSize) / stride + 1;
    auto numPatchesW = (width - patchSize) / stride + 1;
    auto numPatches = numPatchesH * numPatchesW + (includeCls ? 1 : 0);
    auto patchDim = channels * patchSize * patchSize;

    // Extract patches
    auto inputBuf = input->bufferAsT<float>();
    auto outputBuf = output->bufferAsT<float>();

    for (LongType b = 0; b < batch; ++b) {
        LongType patchIdx = includeCls ? 1 : 0;

        // Initialize CLS token to zeros if included
        if (includeCls) {
            for (LongType i = 0; i < patchDim; ++i) {
                outputBuf[b * numPatches * patchDim + i] = 0.0f;
            }
        }

        for (LongType ph = 0; ph < numPatchesH; ++ph) {
            for (LongType pw = 0; pw < numPatchesW; ++pw) {
                LongType outOffset = (b * numPatches + patchIdx) * patchDim;

                for (LongType c = 0; c < channels; ++c) {
                    for (LongType i = 0; i < patchSize; ++i) {
                        for (LongType j = 0; j < patchSize; ++j) {
                            LongType inH = ph * stride + i;
                            LongType inW = pw * stride + j;
                            LongType inOffset = ((b * channels + c) * height + inH) * width + inW;
                            LongType patchOffset = (c * patchSize + i) * patchSize + j;
                            outputBuf[outOffset + patchOffset] = inputBuf[inOffset];
                        }
                    }
                }
                ++patchIdx;
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_patch_embed) {
    auto inputShape = inputShape->at(0);
    auto dtype = ArrayOptions::dataType(inputShape);

    int patchSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 16;
    int stride = block.getIArguments()->size() > 1 ? INT_ARG(1) : patchSize;
    bool includeCls = block.getIArguments()->size() > 2 ? INT_ARG(2) != 0 : true;

    auto batch = shape::sizeAt(inputShape, static_cast<LongType>(0));
    auto channels = shape::sizeAt(inputShape, static_cast<LongType>(1));
    auto height = shape::sizeAt(inputShape, static_cast<LongType>(2));
    auto width = shape::sizeAt(inputShape, static_cast<LongType>(3));

    auto numPatchesH = (height - patchSize) / stride + 1;
    auto numPatchesW = (width - patchSize) / stride + 1;
    auto numPatches = numPatchesH * numPatchesW + (includeCls ? 1 : 0);
    auto patchDim = channels * patchSize * patchSize;

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, numPatches, patchDim}));
}

DECLARE_TYPES(vlm_patch_embed) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_cross_attention - Cross attention between vision and language
#if NOT_EXCLUDED(OP_vlm_cross_attention)
CUSTOM_OP_IMPL(vlm_cross_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);    // [batch, seq_len, dim]
    auto key = INPUT_VARIABLE(1);      // [batch, kv_len, dim]
    auto value = INPUT_VARIABLE(2);    // [batch, kv_len, dim]
    auto output = OUTPUT_VARIABLE(0);  // [batch, seq_len, dim]

    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : 8;
    bool isCausal = block.getIArguments()->size() > 1 ? INT_ARG(1) != 0 : false;
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1) / numHeads));

    // Compute Q * K^T
    auto kT = key->transpose();
    auto scores = MmulHelper::mmul(query, &kT, nullptr, 1.0f, 0.0f);

    // Scale
    scores->applyScalar(scalar::Multiply, scale, *scores);

    // Apply causal mask if needed
    if (isCausal) {
        auto seqLen = query->sizeAt(1);
        auto kvLen = key->sizeAt(1);
        for (LongType i = 0; i < seqLen; ++i) {
            for (LongType j = i + 1; j < kvLen; ++j) {
                // Mask future positions
                scores->p(i, j, -1e9f);
            }
        }
    }

    // Softmax
    scores->applyTransform(transform::SoftMax, *scores);

    // Attention output: scores * V
    MmulHelper::mmul(scores, value, output, 1.0f, 0.0f);

    delete scores;
    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_cross_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(queryShape), shape::order(queryShape),
        shape::rank(queryShape), shape::shapeOf(queryShape)));
}

DECLARE_TYPES(vlm_cross_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_multimodal_fusion - Fusion of vision and language
#if NOT_EXCLUDED(OP_vlm_multimodal_fusion)
CUSTOM_OP_IMPL(vlm_multimodal_fusion, 2, 1, false, 0, 0) {
    auto vision = INPUT_VARIABLE(0);
    auto language = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    int fusionType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    switch (fusionType) {
        case 0: {  // Concatenation along sequence dimension
            std::vector<const NDArray*> inputs = {vision, language};
            helpers::concat(block.launchContext(), inputs, *output, 1);
            break;
        }
        case 1:  // Addition (requires same shape)
            output->assign(*vision + *language);
            break;
        case 2:  // Element-wise multiplication
            output->assign(*vision * *language);
            break;
        default:
            std::vector<const NDArray*> inputs = {vision, language};
            helpers::concat(block.launchContext(), inputs, *output, 1);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_multimodal_fusion) {
    auto visionShape = inputShape->at(0);
    auto languageShape = inputShape->at(1);
    auto dtype = ArrayOptions::dataType(visionShape);

    int fusionType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    if (fusionType == 0) {  // Concatenation
        auto batch = shape::sizeAt(visionShape, static_cast<LongType>(0));
        auto visionLen = shape::sizeAt(visionShape, static_cast<LongType>(1));
        auto languageLen = shape::sizeAt(languageShape, static_cast<LongType>(1));
        auto dim = shape::sizeAt(visionShape, static_cast<LongType>(2));

        return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
            dtype, 'c', {batch, visionLen + languageLen, dim}));
    } else {
        // Same shape as input for add/multiply
        return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
            dtype, shape::order(visionShape),
            shape::rank(visionShape), shape::shapeOf(visionShape)));
    }
}

DECLARE_TYPES(vlm_multimodal_fusion) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_vision_projection - Project vision features
#if NOT_EXCLUDED(OP_vlm_vision_projection)
CUSTOM_OP_IMPL(vlm_vision_projection, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    // Matrix multiplication: output = input @ weights
    MmulHelper::mmul(input, weights, output, 1.0f, 0.0f);

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_vision_projection) {
    auto inputShape = inputShape->at(0);
    auto weightsShape = inputShape->at(1);
    auto dtype = ArrayOptions::dataType(inputShape);

    auto batch = shape::sizeAt(inputShape, static_cast<LongType>(0));
    auto seqLen = shape::sizeAt(inputShape, static_cast<LongType>(1));
    auto outDim = shape::sizeAt(weightsShape, static_cast<LongType>(1));

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, seqLen, outDim}));
}

DECLARE_TYPES(vlm_vision_projection) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_image_preprocess - Image preprocessing
#if NOT_EXCLUDED(OP_vlm_image_preprocess)
CUSTOM_OP_IMPL(vlm_image_preprocess, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Default ImageNet normalization
    float meanR = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.485f;
    float meanG = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.456f;
    float meanB = block.getTArguments()->size() > 2 ? T_ARG(2) : 0.406f;
    float stdR = block.getTArguments()->size() > 3 ? T_ARG(3) : 0.229f;
    float stdG = block.getTArguments()->size() > 4 ? T_ARG(4) : 0.224f;
    float stdB = block.getTArguments()->size() > 5 ? T_ARG(5) : 0.225f;

    // Assuming input is [batch, channels, height, width]
    auto channels = input->sizeAt(1);

    if (channels == 3) {
        // Per-channel normalization
        NDArray means('c', {1, 3, 1, 1}, {meanR, meanG, meanB}, input->dataType(), block.launchContext());
        NDArray stds('c', {1, 3, 1, 1}, {stdR, stdG, stdB}, input->dataType(), block.launchContext());
        output->assign((*input - means) / stds);
    } else {
        // Generic normalization
        float meanAvg = (meanR + meanG + meanB) / 3.0f;
        float stdAvg = (stdR + stdG + stdB) / 3.0f;
        output->assign((*input - meanAvg) / stdAvg);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_image_preprocess) {
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
}

DECLARE_TYPES(vlm_image_preprocess) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// vlm_2d_position_encode - 2D position encoding
#if NOT_EXCLUDED(OP_vlm_2d_position_encode)
CUSTOM_OP_IMPL(vlm_2d_position_encode, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);   // [batch, num_patches, dim]
    auto output = OUTPUT_VARIABLE(0);

    int height = block.getIArguments()->size() > 0 ? INT_ARG(0) : 14;
    int width = block.getIArguments()->size() > 1 ? INT_ARG(1) : 14;
    float temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;

    // Copy input to output
    output->assign(input);

    auto batch = input->sizeAt(0);
    auto numPatches = input->sizeAt(1);
    auto dim = input->sizeAt(2);

    auto outputBuf = output->bufferAsT<float>();

    int halfDim = dim / 2;
    int quarterDim = halfDim / 2;

    for (LongType b = 0; b < batch; ++b) {
        for (LongType p = 0; p < numPatches && p < height * width; ++p) {
            int py = p / width;
            int px = p % width;

            LongType outOffset = (b * numPatches + p) * dim;

            // Height position encoding
            for (int i = 0; i < quarterDim; ++i) {
                float freq = 1.0f / std::pow(temperature, (2.0f * i) / quarterDim);
                outputBuf[outOffset + i] += std::sin(py * freq);
                outputBuf[outOffset + quarterDim + i] += std::cos(py * freq);
            }

            // Width position encoding
            for (int i = 0; i < quarterDim; ++i) {
                float freq = 1.0f / std::pow(temperature, (2.0f * i) / quarterDim);
                outputBuf[outOffset + halfDim + i] += std::sin(px * freq);
                outputBuf[outOffset + halfDim + quarterDim + i] += std::cos(px * freq);
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(vlm_2d_position_encode) {
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
}

DECLARE_TYPES(vlm_2d_position_encode) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

}  // namespace ops
}  // namespace sd

#endif
