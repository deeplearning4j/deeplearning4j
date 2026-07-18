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

//
// @author Eclipse Deeplearning4j
//
// Reference: "Deformable Convolutional Networks" (Dai et al., 2017)
//            "Deformable ConvNets v2" (Zhu et al., 2019)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_deformable_conv2d)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/deformable_conv.h>

namespace sd {
namespace ops {

/**
 * deformable_conv2d - Deformable Convolution 2D
 *
 * Implements deformable convolution where learned offsets are added to
 * the regular sampling grid, allowing the convolution to adapt to
 * geometric transformations in the input.
 *
 * Inputs:
 *   0: input [batch, channels, height, width] (NCHW) or [batch, height, width, channels] (NHWC)
 *   1: weights [out_channels, in_channels/groups, kernel_h, kernel_w]
 *   2: offset [batch, 2*kernel_h*kernel_w*offset_groups, out_h, out_w]
 *   3: bias [out_channels] (optional)
 *   4: mask [batch, kernel_h*kernel_w*offset_groups, out_h, out_w] (optional, for v2)
 *
 * Integer args:
 *   0: kernel_h
 *   1: kernel_w
 *   2: stride_h
 *   3: stride_w
 *   4: pad_h
 *   5: pad_w
 *   6: dilation_h
 *   7: dilation_w
 *   8: groups
 *   9: offset_groups (deformable groups)
 *   10: isNCHW (1 for NCHW, 0 for NHWC)
 *
 * Outputs:
 *   0: output [batch, out_channels, out_h, out_w] or [batch, out_h, out_w, out_channels]
 */
CUSTOM_OP_IMPL(deformable_conv2d, 3, 1, false, 0, 11) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto offset = INPUT_VARIABLE(2);
    auto bias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto mask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

    auto output = OUTPUT_VARIABLE(0);

    // Get integer arguments
    auto kH = INT_ARG(0);
    auto kW = INT_ARG(1);
    auto sH = INT_ARG(2);
    auto sW = INT_ARG(3);
    auto pH = INT_ARG(4);
    auto pW = INT_ARG(5);
    auto dH = INT_ARG(6);
    auto dW = INT_ARG(7);
    auto groups = INT_ARG(8);
    auto offsetGroups = INT_ARG(9);
    auto isNCHW = INT_ARG(10) != 0;

    REQUIRE_TRUE(input->rankOf() == 4, 0,
                 "deformable_conv2d: input must be rank 4, got %i", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0,
                 "deformable_conv2d: weights must be rank 4, got %i", weights->rankOf());
    REQUIRE_TRUE(offset->rankOf() == 4, 0,
                 "deformable_conv2d: offset must be rank 4, got %i", offset->rankOf());

    sd::LongType batchSize, inChannels, inputH, inputW;
    if (isNCHW) {
        batchSize = input->sizeAt(0);
        inChannels = input->sizeAt(1);
        inputH = input->sizeAt(2);
        inputW = input->sizeAt(3);
    } else {
        batchSize = input->sizeAt(0);
        inputH = input->sizeAt(1);
        inputW = input->sizeAt(2);
        inChannels = input->sizeAt(3);
    }

    auto outChannels = weights->sizeAt(0);
    auto outputH = (inputH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    auto outputW = (inputW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

    // Validate offset shape
    REQUIRE_TRUE(offset->sizeAt(0) == batchSize, 0,
                 "deformable_conv2d: offset batch size must match input");
    REQUIRE_TRUE(offset->sizeAt(1) == 2 * kH * kW * offsetGroups, 0,
                 "deformable_conv2d: offset channels must be 2*kH*kW*offset_groups");

    // Validate mask shape if provided
    if (mask != nullptr && !mask->isEmpty()) {
        REQUIRE_TRUE(mask->sizeAt(0) == batchSize, 0,
                     "deformable_conv2d: mask batch size must match input");
        REQUIRE_TRUE(mask->sizeAt(1) == kH * kW * offsetGroups, 0,
                     "deformable_conv2d: mask channels must be kH*kW*offset_groups");
    }

    // Handle NHWC format by permuting to NCHW for internal processing
    NDArray* inputNCHW = const_cast<NDArray*>(input);
    NDArray* outputNCHW = output;
    NDArray* inputPermuted = nullptr;
    NDArray* outputTemp = nullptr;

    if (!isNCHW) {
        // Permute input from NHWC to NCHW
        std::vector<sd::LongType> permDims = {0, 3, 1, 2};
        inputPermuted = new NDArray(input->permute(permDims, false, false));
        inputNCHW = inputPermuted;

        // Create temporary output in NCHW format
        std::vector<sd::LongType> outShape = {batchSize, outChannels, outputH, outputW};
        outputTemp = new NDArray('c', outShape, input->dataType(), block.launchContext());
        outputNCHW = outputTemp;
    }

    // Use helper for CPU/GPU implementation
    helpers::deformableConv2d(block.launchContext(),
                               inputNCHW, weights, offset, bias, mask, outputNCHW,
                               kH, kW, sH, sW, pH, pW, dH, dW, groups, offsetGroups);

    // Convert back to NHWC if needed
    if (!isNCHW) {
        std::vector<sd::LongType> permDimsBack = {0, 2, 3, 1};
        auto permutedBack = outputNCHW->permute(permDimsBack, false, false);
        output->assign(permutedBack);
        delete permutedBack;
        delete inputPermuted;
        delete outputTemp;
    }

    return sd::Status::OK;
}

DECLARE_TYPES(deformable_conv2d) {
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(deformable_conv2d) {
    auto inShape = inputShape->at(0);
    auto weightsShape = inputShape->at(1);
    auto inputType = sd::ArrayOptions::dataType(inShape);

    auto kH = INT_ARG(0);
    auto kW = INT_ARG(1);
    auto sH = INT_ARG(2);
    auto sW = INT_ARG(3);
    auto pH = INT_ARG(4);
    auto pW = INT_ARG(5);
    auto dH = INT_ARG(6);
    auto dW = INT_ARG(7);
    auto isNCHW = INT_ARG(10) != 0;

    sd::LongType batchSize, inH, inW;
    if (isNCHW) {
        batchSize = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
        inH = shape::sizeAt(inShape, static_cast<sd::LongType>(2));
        inW = shape::sizeAt(inShape, static_cast<sd::LongType>(3));
    } else {
        batchSize = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
        inH = shape::sizeAt(inShape, static_cast<sd::LongType>(1));
        inW = shape::sizeAt(inShape, static_cast<sd::LongType>(2));
    }

    auto outChannels = shape::sizeAt(weightsShape, static_cast<sd::LongType>(0));
    auto outputH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    auto outputW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

    std::vector<sd::LongType> outputShape;
    if (isNCHW) {
        outputShape = {batchSize, outChannels, outputH, outputW};
    } else {
        outputShape = {batchSize, outputH, outputW, outChannels};
    }

    auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', outputShape);
    return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
