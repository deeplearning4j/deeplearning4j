/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace nd4j      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void deconv2TFdBackPropMKLDNN(const NDArray* weights, const NDArray* gradO, NDArray* gradI,
                                    const int bS, const int iC, const int iH, const int iW, const int oC, const int oH, const int oW,
                                    const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW) {

    // gradI [bS, iH, iW, iC], mkl doesn't support ndhwc format
    // weights [oC, iC, kH, kW] always, mkl doesn't support weights format [kH, kW, iC, oC]
    // gradO [bS, oH, oW, oC]

    dnnl::memory::dims strides   = { sH, sW };
    dnnl::memory::dims dilation  = { dH - 1, dW - 1 };
    dnnl::memory::dims padding   = { pH, pW };
    dnnl::memory::dims padding_r = { (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW };

    // weights type
    dnnl::memory::data_type wType = weights->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // gradO type
    dnnl::memory::data_type gradOType = gradO->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // gradI type
    dnnl::memory::data_type gradIType = gradI->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;

    dnnl::memory::format_tag xFormat = dnnl::memory::format_tag::nchw;      // isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
    dnnl::memory::format_tag wFormat = dnnl::memory::format_tag::oihw;

    dnnl::memory::dims xDims = {bS, iC, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oH, oW};

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, gradOType, dnnl::memory::format_tag::any);

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormat);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->stridesOf()[0];
    w_user_md.data.format_desc.blocking.strides[1] = weights->stridesOf()[1];
    w_user_md.data.format_desc.blocking.strides[2] = weights->stridesOf()[2];
    w_user_md.data.format_desc.blocking.strides[3] = weights->stridesOf()[3];

    // gradO
    dnnl::memory::desc gradO_mkl_md  = dnnl::memory::desc(zDims, gradOType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, gradOType, xFormat);
    gradO_user_md.data.format_kind = dnnl_blocked;    // overrides format
    gradO_user_md.data.format_desc.blocking.strides[0] = gradO->stridesOf()[0];
    gradO_user_md.data.format_desc.blocking.strides[1] = gradO->stridesOf()[1];
    gradO_user_md.data.format_desc.blocking.strides[2] = gradO->stridesOf()[2];
    gradO_user_md.data.format_desc.blocking.strides[3] = gradO->stridesOf()[3];

    // gradI
    dnnl::memory::desc gradI_mkl_md  = dnnl::memory::desc(xDims, gradIType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, gradIType, xFormat);
    gradI_user_md.data.format_kind = dnnl_blocked;    // overrides format
    gradI_user_md.data.format_desc.blocking.strides[0] = gradI->stridesOf()[0];
    gradI_user_md.data.format_desc.blocking.strides[1] = gradI->stridesOf()[1];
    gradI_user_md.data.format_desc.blocking.strides[2] = gradI->stridesOf()[2];
    gradI_user_md.data.format_desc.blocking.strides[3] = gradI->stridesOf()[3];


    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // forward primitive description
    dnnl::convolution_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto, x_mkl_md, w_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::convolution_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // backward data primitive description
    dnnl::convolution_backward_data::desc op_data_bp_desc(dnnl::algorithm::convolution_auto, gradI_mkl_md, w_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::convolution_backward_data::primitive_desc op_data_bp_prim_desc(op_data_bp_desc, engine, op_ff_prim_desc);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    dnnl::stream stream(engine);

    // provide memory buffers and check whether reorder is required

    // weights
    auto w_user_mem = dnnl::memory(w_user_md, engine, weights->getBuffer());
    const bool wReorder = op_data_bp_prim_desc.weights_desc() != w_user_mem.get_desc();
    auto w_mkl_mem = wReorder ? dnnl::memory(op_data_bp_prim_desc.weights_desc(), engine) : w_user_mem;
    if (wReorder)
        dnnl::reorder(w_user_mem, w_mkl_mem).execute(stream, w_user_mem, w_mkl_mem);
    args[DNNL_ARG_WEIGHTS] = w_mkl_mem;

    // gradO
    auto gradO_user_mem = dnnl::memory(gradO_user_md, engine, gradO->getBuffer());
    const bool gradOReorder = op_data_bp_prim_desc.diff_dst_desc() != gradO_user_mem.get_desc();
    auto gradO_mkl_mem = gradOReorder ? dnnl::memory(op_data_bp_prim_desc.diff_dst_desc(), engine) : gradO_user_mem;
    if (gradOReorder)
        dnnl::reorder(gradO_user_mem, gradO_mkl_mem).execute(stream, gradO_user_mem, gradO_mkl_mem);
    args[DNNL_ARG_DIFF_DST] = gradO_mkl_mem;

    // gradI
    auto gradI_user_mem = dnnl::memory(gradI_user_md, engine, gradI->getBuffer());
    const bool gradIReorder = op_data_bp_prim_desc.diff_src_desc() != gradI_user_mem.get_desc();
    auto gradI_mkl_mem = gradIReorder ? dnnl::memory(op_data_bp_prim_desc.diff_src_desc(), engine) : gradI_user_mem;
    args[DNNL_ARG_DIFF_SRC] = gradI_mkl_mem;

    // run backward data calculations
    dnnl::convolution_backward_data(op_data_bp_prim_desc).execute(stream, args);

    // reorder gradI if necessary
    if (gradIReorder)
        dnnl::reorder(gradI_mkl_mem, gradI_user_mem).execute(stream, gradI_mkl_mem, gradI_user_mem);

    stream.wait();

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}



//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d_tf) {

    auto gradO      = INPUT_VARIABLE(2);                                                // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    auto weights    = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] always
    auto gradIShape = INPUT_VARIABLE(0);                                                // [4] - shape of input of conv2d (that is shape of gradI)

    auto gradI = OUTPUT_VARIABLE(0);                                                  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // INT_ARG(9): 1-NHWC, 0-NCHW

    const int rank = gradO->rankOf();

    REQUIRE_TRUE(weights->rankOf() == rank, 0, "CUSTOM DECONV2D_TF OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradIShape->rankOf() == 1, 0, "CUSTOM DECONV2D_TF OP: rank of array with output shape must be equal to 1, but got %i instead !", gradIShape->rankOf());
    REQUIRE_TRUE(gradIShape->lengthOf() == rank, 0, "CUSTOM DECONV2D_TF OP: length of array with output shape must be equal to 4, but got %i instead !", gradIShape->lengthOf());

    int indIOioC, indIiH, indWoC(3), indOoH;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indOoH = 1;
    }
    else {
        indIOioC = 1; indIiH = 2; indOoH = 2;
    }

    std::vector<Nd4jLong> gradIShapeVector = gradIShape->template asVectorT<Nd4jLong>();

    const int bS = gradIShapeVector[0];                     // batch size
    const int iH = gradIShapeVector[indIiH];                // input height
    const int iW = gradIShapeVector[indIiH+1];              // input width
    const int iC = gradIShapeVector[indIOioC];              // input channels
    const int oC = weights->sizeAt(indWoC);                 // output channels
    const int oH = gradO->sizeAt(indOoH);                   // input height
    const int oW = gradO->sizeAt(indOoH);                   // input width

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::vector<Nd4jLong> expectedGradOShape   = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1});
    std::vector<Nd4jLong> expectedWeightsShape = {kH, kW, iC, oC};
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "CUSTOM DECONV2D_TF OP: wrong shape of input array, basing on array with output shape expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM DECONV2D_TF OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    // mkl supports only [oC, iC, kH, kW] for weights
    weights = new NDArray(weights->permute({3,2,0,1}));        // [kH, kW, iC, oC] -> [oC, iC, kH, kW]

    // mkl supports NCHW format only
    if(!isNCHW) {
        gradI = new NDArray(gradI->permute({0,3,1,2}));    // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
        gradO = new NDArray(gradO->permute({0,3,1,2}));    // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    }

    deconv2TFdBackPropMKLDNN(weights, gradO,  gradI, bS, iC, iH, iW, oC, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW);

    delete weights;

    if(!isNCHW) {
        delete gradI;
        delete gradO;
    }

    // ConvolutionUtils::conv2dBP(block, &input, weights, nullptr, gradO, gradI, nullptr, nullptr, kH,kW,sH,sW,pH,pW,dH,dW,isSameMode,isNCHW);

    return Status::OK();
}

PLATFORM_CHECK(deconv2d_tf) {
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] always
    auto gradO   = INPUT_VARIABLE(2);                                                // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    auto gradI   = OUTPUT_VARIABLE(0);                                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW), gradI


    const DataType wType = weights->dataType();
    const DataType gradOType = gradO->dataType();
    const DataType gradIType = gradI->dataType();

    return block.isUseMKLDNN() && ((wType==DataType::FLOAT32 || wType==DataType::BFLOAT16) && (gradOType==DataType::FLOAT32 || gradOType==DataType::BFLOAT16) && (gradIType==DataType::FLOAT32 || gradIType==DataType::BFLOAT16));
}

}
}
}
