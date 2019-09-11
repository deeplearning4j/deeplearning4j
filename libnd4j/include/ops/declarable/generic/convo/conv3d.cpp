/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/


//
// @author Yurii Shyrma, created on 05.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_conv3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/addBias.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {

#ifdef HAVE_MKLDNN
using namespace mkldnn;
#endif

CUSTOM_OP_IMPL(conv3dnew, 2, 1, false, 0, 13) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC] always
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM CONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, weights, bias, output})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("conv3dnew"));
        }

        if (streams[0].checkAndReset({input, weights, bias}, {output}, {}, {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode, isNCDHW})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc conv_src_md(empty), conv_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_weights_md(empty), user_bias_md(empty), user_dst_md(empty);
            mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;

            ConvolutionUtils::getMKLDNNMemoryDescConv3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode, isNCDHW,
                    bS, iC, iD, iH, iW, oC, oD, oH, oW, input, nullptr, weights, nullptr, bias, output,
                    &conv_src_md, nullptr, &conv_weights_md, nullptr, &conv_bias_md, &conv_dst_md,
                    &user_src_md, nullptr, &user_weights_md, nullptr, &user_bias_md, &user_dst_md,
                    conv_strides, conv_padding, conv_padding_r);

            auto conv_desc = bias != nullptr
                    ? convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                    : convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

            auto engine = streams[0].getEngine();
            auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray*>(input)->buffer());
            auto user_weights_memory = mkldnn::memory({user_weights_md, engine}, const_cast<NDArray*>(weights)->buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());

            auto conv_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(conv_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                conv_src_memory = mkldnn::memory(conv_prim_desc.src_primitive_desc());
                streams[0].addMemory(conv_src_memory);
                streams[0].addOperation(reorder(user_src_memory, conv_src_memory));
            }

            auto conv_weights_memory = user_weights_memory;
            streams[0].addMemory(user_weights_memory);
            if (mkldnn::memory::primitive_desc(conv_prim_desc.weights_primitive_desc())
                    != user_weights_memory.get_primitive_desc()) {
                conv_weights_memory = mkldnn::memory(conv_prim_desc.weights_primitive_desc());
                streams[0].addMemory(conv_weights_memory);
                streams[0].addOperation(reorder(user_weights_memory, conv_weights_memory));
            }

            auto conv_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                conv_dst_memory = mkldnn::memory(conv_prim_desc.dst_primitive_desc());
                streams[0].addMemory(conv_dst_memory);
            }

            if (bias != nullptr) {
                auto conv_bias_memory = mkldnn::memory(conv_prim_desc.bias_primitive_desc(), bias->buffer());
                streams[0].addMemory(conv_bias_memory);
                streams[0].addOperation(convolution_forward(conv_prim_desc, conv_src_memory, conv_weights_memory, conv_bias_memory, conv_dst_memory));
            } else {
                streams[0].addOperation(convolution_forward(conv_prim_desc, conv_src_memory, conv_weights_memory, conv_dst_memory));
            }

            if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(conv_dst_memory, user_dst_memory));
            }
        }

        streams[0].submitAndWait();
        return Status::OK();
    }
#endif
    nd4j_debug("MKL-DNN is not used for conv3dnew!\n", 0);

    std::vector<int> permutForOutput;

    if (isNCDHW)
        permutForOutput    = {0,2,3,4,1};                                        // [bS, oC, oD, oH, oW] -> [bS, oD, oH, oW, oC]
    else
        input = new NDArray(input->permute({0,4,1,2,3}));

    NDArray columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, input->dataType(), block.launchContext());
    ConvolutionUtils::vol2col(block, *input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                 // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]
    // [bS, iC, kD, kH, kW, oD, oH, oW] x [kD, kH, kW, iC, oC] = [bS, oD, oH, oW, oC]
    MmulHelper::tensorDot(&columns, weights, output, {1,2,3,4}, {3,0,1,2}, permutForOutput);

    if(bias)
        // output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
        helpers::addBias(block, *output, *bias, *output, isNCDHW);

     if(!isNCDHW)
        delete input;

    return Status::OK();
}

   DECLARE_TYPES(conv3dnew) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                ->setAllowedInputTypes(1, {ALL_FLOATS})
                ->setAllowedInputTypes(2, {ALL_FLOATS})
                ->setAllowedOutputTypes({ALL_FLOATS});
    }

DECLARE_SHAPE_FN(conv3dnew) {

    auto inputShapeInfo   = inputShape->at(0);                                  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weightsShapeInfo = inputShape->at(1);                                  // [kD, kH, kW, iC, oC] always
    auto biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;    // [oC]

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID;
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW

    const int rank = 5;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM CONV3D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM CONV3D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo);

    int indIOioC, indIiD, indWoC(4);
    if(!isNCDHW) {
        indIOioC = 4; indIiD = 1;
    }
    else {
        indIOioC = 1; indIiD = 2;
    }

    int bS = inputShapeInfo[1];                           // batch size
    int iD = inputShapeInfo[indIiD+1];                    // input depth
    int iH = inputShapeInfo[indIiD+2];                    // input height
    int iW = inputShapeInfo[indIiD+3];                    // input width
    int iC = inputShapeInfo[indIOioC+1];                  // input channels
    int oC = weightsShapeInfo[indWoC+1];                  // output channels

    std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "CUSTOM CONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM CONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    Nd4jLong* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), Nd4jLong);

    outputShapeInfo[0] = rank;
    outputShapeInfo[1] = bS;
    if (isNCDHW) {
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = oC;
    }

    ShapeUtils::updateStridesAndType(outputShapeInfo, weightsShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(CONSTANT(outputShapeInfo));
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(conv3dnew_bp, 3, 2, false, 0, 13) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC] always
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, iC, oC] always
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D_BP OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 5, 0, "CUSTOM CONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !", gradO->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID
    int isNDHWC  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNDHWC, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0,  "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, weights, bias, gradO, gradI, gradW, gradB})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("conv3dnew_bp_weights"));
            streams.push_back(MKLDNNStream("conv3dnew_bp_data"));
        }

        bool resetW = streams[0].checkAndReset({input, weights, bias, gradO}, {gradI, gradW, gradB}, {}, {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode, isNDHWC});
        bool resetI = streams[1].checkAndReset({input, weights, bias, gradO}, {gradI, gradW, gradB}, {}, {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode, isNDHWC});
        if (resetW || resetI) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc conv_src_md(empty), conv_diff_src_md(empty), conv_weights_md(empty),
                                 conv_diff_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_weights_md(empty),
                                 user_diff_weights_md(empty), user_bias_md(empty), user_dst_md(empty);
            mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;

            ConvolutionUtils::getMKLDNNMemoryDescConv3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode, isNDHWC,
                    bS, iC, iD, iH, iW, oC, oD, oH, oW, input, gradI, weights, gradW, gradB, gradO,
                    &conv_src_md, &conv_diff_src_md, &conv_weights_md, &conv_diff_weights_md, &conv_bias_md, &conv_dst_md,
                    &user_src_md, &user_diff_src_md, &user_weights_md, &user_diff_weights_md, &user_bias_md, &user_dst_md,
                    conv_strides, conv_padding, conv_padding_r);

            auto conv_desc = gradB != nullptr
                    ? convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                    : convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

            auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, streams[0].getEngine());

            if (gradW != nullptr) {
                auto convW_desc = gradB != nullptr
                        ? convolution_backward_weights::desc(
                                convolution_direct, conv_src_md, conv_diff_weights_md, conv_bias_md,
                                conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                        : convolution_backward_weights::desc(
                                convolution_direct, conv_src_md, conv_diff_weights_md,
                                conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

                auto engine = streams[0].getEngine();
                auto convW_prim_desc = convolution_backward_weights::primitive_desc(convW_desc, engine, conv_prim_desc);
                auto userW_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray*>(input)->buffer());
                auto userW_weights_memory = mkldnn::memory({user_diff_weights_md, engine}, gradW->buffer());
                auto userW_dst_memory = mkldnn::memory({user_dst_md, engine}, const_cast<NDArray*>(gradO)->buffer());

                auto convW_src_memory = userW_src_memory;
                streams[0].addMemory(userW_src_memory);
                if (mkldnn::memory::primitive_desc(convW_prim_desc.src_primitive_desc())
                        != userW_src_memory.get_primitive_desc()) {
                    convW_src_memory = mkldnn::memory(convW_prim_desc.src_primitive_desc());
                    streams[0].addMemory(convW_src_memory);
                    streams[0].addOperation(reorder(userW_src_memory, convW_src_memory));
                }

                auto convW_weights_memory = userW_weights_memory;
                streams[0].addMemory(userW_weights_memory);
                if (mkldnn::memory::primitive_desc(convW_prim_desc.diff_weights_primitive_desc())
                        != userW_weights_memory.get_primitive_desc()) {
                    convW_weights_memory = mkldnn::memory(convW_prim_desc.diff_weights_primitive_desc());
                    streams[0].addMemory(convW_weights_memory);
                }

                auto convW_dst_memory = userW_dst_memory;
                streams[0].addMemory(userW_dst_memory);
                if (mkldnn::memory::primitive_desc(convW_prim_desc.diff_dst_primitive_desc())
                        != userW_dst_memory.get_primitive_desc()) {
                    convW_dst_memory = mkldnn::memory(convW_prim_desc.diff_dst_primitive_desc());
                    streams[0].addMemory(convW_dst_memory);
                    streams[0].addOperation(reorder(userW_dst_memory, convW_dst_memory));
                }

                if (gradB != nullptr) {
                    auto convW_bias_memory = mkldnn::memory(convW_prim_desc.diff_bias_primitive_desc(), gradB->buffer());
                    streams[0].addMemory(convW_bias_memory);
                    streams[0].addOperation(convolution_backward_weights(convW_prim_desc, convW_src_memory, convW_dst_memory, convW_weights_memory, convW_bias_memory));
                } else {
                    streams[0].addOperation(convolution_backward_weights(convW_prim_desc, convW_src_memory, convW_dst_memory, convW_weights_memory));
                }

                if (mkldnn::memory::primitive_desc(convW_prim_desc.diff_weights_primitive_desc())
                        != userW_weights_memory.get_primitive_desc()) {
                    streams[0].addOperation(reorder(convW_weights_memory, userW_weights_memory));
                }
            }

            if (gradI != nullptr) {
                auto convI_desc =
                        convolution_backward_data::desc(
                                convolution_direct, conv_diff_src_md, conv_weights_md,
                                conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

                auto engine = streams[1].getEngine();
                auto convI_prim_desc = convolution_backward_data::primitive_desc(convI_desc, engine, conv_prim_desc);
                auto userI_src_memory = mkldnn::memory({user_diff_src_md, engine}, gradI->buffer());
                auto userI_weights_memory = mkldnn::memory({user_weights_md, engine}, const_cast<NDArray*>(weights)->buffer());
                auto userI_dst_memory = mkldnn::memory({user_dst_md, engine}, const_cast<NDArray*>(gradO)->buffer());

                auto convI_src_memory = userI_src_memory;
                streams[1].addMemory(userI_src_memory);
                if (mkldnn::memory::primitive_desc(convI_prim_desc.diff_src_primitive_desc())
                        != userI_src_memory.get_primitive_desc()) {
                    convI_src_memory = mkldnn::memory(convI_prim_desc.diff_src_primitive_desc());
                    streams[1].addMemory(convI_src_memory);
                }

                auto convI_weights_memory = userI_weights_memory;
                streams[1].addMemory(userI_weights_memory);
                if (mkldnn::memory::primitive_desc(convI_prim_desc.weights_primitive_desc())
                        != userI_weights_memory.get_primitive_desc()) {
                    convI_weights_memory = mkldnn::memory(convI_prim_desc.weights_primitive_desc());
                    streams[1].addMemory(convI_weights_memory);
                    streams[1].addOperation(reorder(userI_weights_memory, convI_weights_memory));
                }

                auto convI_dst_memory = userI_dst_memory;
                streams[1].addMemory(userI_dst_memory);
                if (mkldnn::memory::primitive_desc(convI_prim_desc.diff_dst_primitive_desc())
                        != userI_dst_memory.get_primitive_desc()) {
                    convI_dst_memory = mkldnn::memory(convI_prim_desc.diff_dst_primitive_desc());
                    streams[1].addMemory(convI_dst_memory);
                    streams[1].addOperation(reorder(userI_dst_memory, convI_dst_memory));
                }

                streams[1].addOperation(convolution_backward_data(convI_prim_desc, convI_dst_memory, convI_weights_memory, convI_src_memory));

                if (mkldnn::memory::primitive_desc(convI_prim_desc.diff_src_primitive_desc())
                        != userI_src_memory.get_primitive_desc()) {
                    streams[1].addOperation(reorder(convI_src_memory, userI_src_memory));
                }
            }
        }

        if (gradW != nullptr) {
            streams[0].submitAndWait();
        }
        if (gradI != nullptr) {
            streams[1].submitAndWait();
        }
        return Status::OK();
    }
#endif
    nd4j_debug("MKL-DNN is not used for conv3dnew_bp!\n", 0);

    std::vector<int> gradOaxesForDot;

    if(!isNDHWC) {
        gradOaxesForDot  = {0,1,2,3};                                           // bS, oD, oH, oW
        input = new NDArray(input->permute({0,4,1,2,3}));                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        gradI = new NDArray(gradI->permute({0,4,1,2,3}));                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
    }
    else {
        gradOaxesForDot  = {0,2,3,4};                                           // bS, oD, oH, oW
    }

    // ----- calculation of gradW and gradB ----- //
    NDArray columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, input->dataType(), block.launchContext());
    ConvolutionUtils::vol2col(block, *input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                   // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]
    MmulHelper::tensorDot(&columns, gradO, gradW, {0,5,6,7}, gradOaxesForDot, {3,0,1,2,4});     // [bS, iC, kD, kH, kW, oD, oH, oW] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [iC, kD, kH, kW, oC]

    //----- calculation of gradO -----//
    if(gradB) {
        if(gradB->rankOf() == 2)
            gradB = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}));
        gradO->reduceAlongDimension(reduce::Sum, gradB, gradOaxesForDot);                          // sum over bS oD oH oW
        if(gradB != OUTPUT_VARIABLE(2))
            delete gradB;
    }

    //----- calculation of gradI -----//
    MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, {2,3,4,1,0,5,6,7});   // [kD, kH, kW, iC, oC] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [kD, kH, kW, iC, bS, oD, oH, oW]
    ConvolutionUtils::col2vol(block, columns, *gradI, sD, sH, sW, pD, pH, pW, dD, dH, dW);                   // columns [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to  [bS, iC, iD, iH, iW]

    if(!isNDHWC) {
        delete input;
        delete gradI;
    }

    return Status::OK();
}

   DECLARE_TYPES(conv3dnew_bp) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                ->setAllowedInputTypes(1, {ALL_FLOATS})
                ->setAllowedInputTypes(2, {ALL_FLOATS})
                ->setAllowedInputTypes(3, {ALL_FLOATS})
                ->setAllowedOutputTypes({ALL_FLOATS});
    }


DECLARE_SHAPE_FN(conv3dnew_bp) {

    Nd4jLong* inputShapeInfo   = inputShape->at(0);                                              // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    Nd4jLong* weightsShapeInfo = inputShape->at(1);                                              // [kD, kH, kW, iC, oC] always
    Nd4jLong* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;                // [oC]
    Nd4jLong* gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);      // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID
    int isNDHWC  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW

    const int rank = 5;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM CONV3D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM CONV3D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM CONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradOShapeInfo);

    int indIOioC, indIiD, indWoC(4);
    if(!isNDHWC) {
        indIOioC = 4; indIiD = 1;
    }
    else {
        indIOioC = 1; indIiD = 2;
    }

    int bS = inputShapeInfo[1];                           // batch size
    int iD = inputShapeInfo[indIiD+1];                    // input depth
    int iH = inputShapeInfo[indIiD+2];                    // input height
    int iW = inputShapeInfo[indIiD+3];                    // input width
    int iC = inputShapeInfo[indIOioC+1];                  // input channels
    int oC = weightsShapeInfo[indWoC+1];                  // output channels

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIiD,indIiD+1,indIiD+2}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
    REQUIRE_TRUE(expectedGradOShape   == ShapeUtils::shapeAsString(gradOShapeInfo),   0, "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    auto gradIshapeInfo = ShapeBuilders::copyShapeInfoAndType(inputShapeInfo,   gradOShapeInfo, false, block.getWorkspace());
    auto gradWshapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, gradOShapeInfo, false, block.getWorkspace());

    if(biasShapeInfo) {
        auto gradBshapeInfo = ShapeBuilders::copyShapeInfoAndType(biasShapeInfo, gradOShapeInfo, false, block.getWorkspace());
        return SHAPELIST(CONSTANT(gradIshapeInfo), CONSTANT(gradWshapeInfo), CONSTANT(gradBshapeInfo));
    }

    return SHAPELIST(CONSTANT(gradIshapeInfo), CONSTANT(gradWshapeInfo));
}
}
}

#endif
