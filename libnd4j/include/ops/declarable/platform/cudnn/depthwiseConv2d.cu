/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace sd      {
namespace ops       {
namespace platforms {


//////////////////////////////////////////////////////////////////////////
static void depthwiseConv2dCUDNN(const LaunchContext* context,
                        const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output,
                        const int kH, const int kW,
                        const int sH, const int sW,
                        const int pH, const int pW,
                        const int dH, const int dW,
                        const int paddingMode, const bool isNCHW) {

    // cudnn supports only following case: mC = 1, oC = iC (groupCount == iC)

    // input [bS, iC, iH, iW] nchw or [bS, iH, iW, iC] nhwc
    // weights [iC, mC, kH, kW]
    // bias [oC], may be nullptr
    // output [bS, oC, oH, oW] nchw or [bS, oH, oW, oC] nhwc
    // oC = iC*mC

    int bS, iC, iH, iW, mC, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;           // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(1);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: can't set stream for cuDNN", err);

    cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(x, format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(x, cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC), input->strideAt(indIiH), input->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for input failed", err);

    // weights descriptor
    cudnnFilterDescriptor_t w;
    cudnnCreateFilterDescriptor(&w);
    err = cudnnSetFilter4dDescriptor(w, cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, iC, mC, kH, kW);
    if(err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnSetFilter4dDescriptor failed", err);

    // output descriptor
    cudnnTensorDescriptor_t z;
    cudnnCreateTensorDescriptor(&z);
    if(output->ews() == 1 && output->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(z, format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
    else
        err = cudnnSetTensor4dDescriptorEx(z, cudnnDataType(output->dataType()), bS, oC, oH, oW, output->strideAt(0), output->strideAt(indIOioC), output->strideAt(indOoH), output->strideAt(indOoH + 1));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for output failed", err);

    // description of convolution
    cudnnConvolutionDescriptor_t conv;
    cudnnCreateConvolutionDescriptor(&conv);
    err = cudnnSetConvolution2dDescriptor(conv, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnSetConvolution2dDescriptor failed", err);
    err = cudnnSetConvolutionGroupCount(conv, iC);  // set number of groups (depthwise mode) in description of convolution, groupCount == iC
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnSetConvolutionGroupCount failed", err);

    // algorithm description
    cudnnConvolutionFwdAlgo_t algo;
    err = cudnnGetConvolutionForwardAlgorithm(*handle, x, w, conv, z, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnGetConvolutionForwardAlgorithm failed", err);

    // allocate auxiliary device memory, abbreviation ws means workspace
    size_t wsSize;
    err = cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize);
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnGetConvolutionForwardWorkspaceSize failed", err);
    void* wsData;
    auto cudaErr = cudaMalloc(&wsData, wsSize);
    if (cudaErr != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudaMalloc for auxiliary workspace memory failed", cudaErr);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({output}, {input, weights, bias});

    // run calculation
    err = cudnnConvolutionForward(*handle, alpha, x, input->getSpecialBuffer(), w, weights->getSpecialBuffer(), conv, algo, wsData, wsSize, beta, z, output->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnConvolutionForward failed", err);

    // add bias if it is present
    if (bias != nullptr) {

        cudnnTensorDescriptor_t b;
        cudnnCreateTensorDescriptor(&b);
        // err = cudnnSetTensor4dDescriptor(b, format, cudnnDataType(bias->dataType()), 1, isNCHW ? bias->lengthOf() : 1, 1, isNCHW ? 1: bias->lengthOf());
        err = cudnnSetTensor4dDescriptor(b, CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);
        if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnSetTensor4dDescriptor for bias failed", err);
        err = cudnnAddTensor(*handle, alpha, b, bias->getSpecialBuffer(), alpha, z, output->specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnAddTensor bias failed", err);
    }

    // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    // if (cudaErr != 0)
    //     throw cuda_exception::build("depthwiseConv2dCUDNN: cudaStreamSynchronize failed !", cudaErr);

    cudaErr = cudaFree(wsData);
    if (cudaErr != 0) throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudaFree for auxiliary workspace memory failed", cudaErr);

    NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void depthwiseConv2dBpCUDNN(const LaunchContext* context,
                                const NDArray* input, const NDArray* weights, const NDArray* gradO,
                                NDArray* gradI, NDArray* gradW, NDArray* gradB,
                                const int kH, const int kW,
                                const int sH, const int sW,
                                const int pH, const int pW,
                                const int dH, const int dW,
                                const int paddingMode, const bool isNCHW) {

    // cudnn supports only following case: mC = 1, oC = iC (groupCount == iC)

    // input, gradI [bS, iC, iH, iW] nchw or [bS, iH, iW, iC] nhwc
    // weights, gradW [iC, mC, kH, kW]
    // gradB [oC], may be nullptr
    // gradO [bS, oC, oH, oW] nchw or [bS, oH, oW, oC] nhwc
    // oC = iC*mC

    int bS, iC, iH, iW, mC, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;           // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(1);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: can't set stream for cuDNN", err);

    cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(x, format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(x, cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC), input->strideAt(indIiH), input->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for input failed", err);

    // gradO descriptor
    cudnnTensorDescriptor_t dz;
    cudnnCreateTensorDescriptor(&dz);
    if(gradO->ews() == 1 && gradO->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(dz, format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
    else
        err = cudnnSetTensor4dDescriptorEx(dz, cudnnDataType(gradO->dataType()), bS, oC, oH, oW, gradO->strideAt(0), gradO->strideAt(indIOioC), gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for gradO failed", err);

    // gradI descriptor
    cudnnTensorDescriptor_t dx;
    cudnnCreateTensorDescriptor(&dx);
    if(gradI->ews() == 1 && gradI->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(dx, format, cudnnDataType(gradI->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(dx, cudnnDataType(gradI->dataType()), bS, iC, iH, iW, gradI->strideAt(0), gradI->strideAt(indIOioC), gradI->strideAt(indIiH), gradI->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for gradI failed", err);

    // gradW descriptor
    cudnnFilterDescriptor_t dw;
    cudnnCreateFilterDescriptor(&dw);
    err = cudnnSetFilter4dDescriptor(dw, cudnnDataType(gradW->dataType()), CUDNN_TENSOR_NCHW, iC, mC, kH, kW);
    if(err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetFilter4dDescriptor gradW failed", err);

    // description of convolution
    cudnnConvolutionDescriptor_t conv;
    cudnnCreateConvolutionDescriptor(&conv);
    err = cudnnSetConvolution2dDescriptor(conv, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetConvolution2dDescriptor failed", err);
    err = cudnnSetConvolutionGroupCount(conv, iC);  // set number of groups (depthwise mode) in description of convolution, groupCount == iC
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetConvolutionGroupCount failed", err);

    // gradW algorithm description
    cudnnConvolutionBwdFilterAlgo_t algoGradW;
    err = cudnnGetConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algoGradW);
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnGetConvolutionBackwardFilterAlgorithm failed", err);

    // gradI algorithm description
    cudnnConvolutionBwdDataAlgo_t algoGradI;
    err = cudnnGetConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algoGradI);
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnGetConvolutionBackwardDataAlgorithm failed", err);

    // allocate auxiliary device memory for gradW calculation, abbreviation ws means workspace
    size_t wsGradWSize;
    err = cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, x, dz, conv, dw, algoGradW, &wsGradWSize);
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnGetConvolutionBackwardFilterWorkspaceSize failed", err);
    void* wsGradWData;
    auto cudaErr = cudaMalloc(&wsGradWData, wsGradWSize);
    if (cudaErr != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudaMalloc for auxiliary workspace memory wsGradWData failed", cudaErr);

    // allocate auxiliary device memory for gradI calculation, abbreviation ws means workspace
    size_t wsGradISize;
    err = cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, dw, dz, conv, dx, algoGradI, &wsGradISize);
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnGetConvolutionBackwardDataWorkspaceSize failed", err);
    void* wsGradIData;
    cudaErr = cudaMalloc(&wsGradIData, wsGradISize);
    if (cudaErr != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudaMalloc for auxiliary workspace memory wsGradIData failed", cudaErr);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});

    // run calculation for gradB (if not nullptr)
    if(gradB != nullptr) {
        cudnnTensorDescriptor_t db;
        cudnnCreateTensorDescriptor(&db);
        // err = cudnnSetTensor4dDescriptor(db, format, cudnnDataType(gradB->dataType()), 1, isNCHW ? gradB->lengthOf() : 1, 1, isNCHW ? 1: gradB->lengthOf());
        err = cudnnSetTensor4dDescriptor(db, CUDNN_TENSOR_NCHW, cudnnDataType(gradB->dataType()), 1, oC, 1, 1);
        if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnSetTensor4dDescriptor for gradB failed", err);

        err = cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->getSpecialBuffer(), beta, db, gradB->getSpecialBuffer());
        if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnConvolutionBackwardBias failed", err);
    }

    // run calculation for gradW
    err = cudnnConvolutionBackwardFilter(*handle, alpha, x, input->getSpecialBuffer(), dz, gradO->getSpecialBuffer(), conv, algoGradW, wsGradWData, wsGradWSize, beta, dw, gradW->getSpecialBuffer());
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnConvolutionBackwardFilter failed", err);

    // run calculation for gradI
    err = cudnnConvolutionBackwardData(*handle, alpha, dw, weights->getSpecialBuffer(), dz, gradO->getSpecialBuffer(), conv, algoGradI, wsGradIData, wsGradISize, beta, dx, gradI->getSpecialBuffer());
    if (err != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudnnConvolutionBackwardData failed", err);

    // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    // if (cudaErr != 0)
    //     throw cuda_exception::build("depthwiseConv2dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

    cudaErr = cudaFree(wsGradWData);
    if (cudaErr != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudaFree for auxiliary workspace memory wsGradWData failed", cudaErr);
    cudaErr = cudaFree(wsGradIData);
    if (cudaErr != 0) throw sd::cuda_exception::build("depthwiseConv2dBpCUDNN: cudaFree for auxiliary workspace memory wsGradIData failed", cudaErr);

    NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(depthwise_conv2d, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC] = iC*mC

    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "DEPTHWISECONV2D CUDNN OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "DEPTHWISECONV2D CUDNN OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());

    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                               // 0-VALID, 1-SAME
    int isNCHW      = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;      // INT_ARG(9): 0-NCHW,  1-NHWC
    int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;         // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, mC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "DEPTHWISECONV2D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    REQUIRE_TRUE(output->sizeAt(indIOioC) == iC*mC, 0, "DEPTHWISECONV2D CUDNN OP: the output_channels must be equal to input_channels * channels_multiplier = %i !", iC*mC);
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "DEPTHWISECONV2D CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    std::vector<int> wPermut;     // cudnn support format {oC, iC/groupCount, kH, kW} only, mC = 1, oC = iC (groupCount == iC) that is {iC, mC, kH, kW} in our case
    if(0 == wFormat)
        wPermut = {2,3,0,1};         // kH, kW, iC, mC -> iC, mC, kH, kW
    else if(1 == wFormat)
        wPermut = {1,0,2,3};         // mC, iC, kH, kW -> iC, mC, kH, kW
    else
        wPermut = {3,0,1,2};         // mC, kH, kW, iC -> iC, mC, kH, kW

    NDArray* newWeights = new NDArray(weights->ordering(), {iC, mC, kH, kW}, weights->dataType(), weights->getContext());
    newWeights->assign(weights->permute(wPermut));

    NDArray* newInput = input;
    NDArray* newGradI = nullptr;
    if(paddingMode == 1) // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
        checkConv2dCUDNNPadAsymmetric(newInput, newGradI, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);

    depthwiseConv2dCUDNN(block.launchContext(), newInput, newWeights, bias, output, kH,kW,sH,sW,pH,pW,dH,dW, paddingMode, isNCHW);

    if(newInput != input)
        delete newInput;

    delete newWeights;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(depthwise_conv2d, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC] = iC*mC

    const int paddingMode = INT_ARG(8);                                  // 0-VALID, 1-SAME, 2-CAUSAL
    const int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;       // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

    const int mC = weights->sizeAt(0 == wFormat ? 3 : 0);

    const bool badInputType   = input->dataType()   != DataType::DOUBLE && input->dataType()   != DataType::FLOAT32 && input->dataType()   != DataType::HALF;
    const bool badWeightsType = weights->dataType() != DataType::DOUBLE && weights->dataType() != DataType::FLOAT32 && weights->dataType() != DataType::HALF;
    const bool badBiasType    = bias == nullptr ? false : (bias->dataType() != DataType::DOUBLE && bias->dataType() != DataType::FLOAT32 && bias->dataType() != DataType::HALF);

    return mC == 1 && paddingMode != 2 && !badInputType && !badWeightsType && !badBiasType;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(depthwise_conv2d_bp, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC] = [iC*mC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "DEPTHWISECONV2D_BP CUDNN OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "DEPTHWISECONV2D_BP CUDNN OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 4, 0,   "DEPTHWISECONV2D_BP CUDNN OP: rank of output gradients (next epsilon) array must be equal to 4, but got %i instead !", gradO->rankOf());

    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                               // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // INT_ARG(9): 1-NHWC, 0-NCHW
    int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;         // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

    std::vector<Nd4jLong> expectedGradOShape   = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1});
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, mC);
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "DEPTHWISECONV2D_BP CUDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "DEPTHWISECONV2D_BP CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "DEPTHWISECONV2D_BP CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    std::vector<int> wPermut, gradWPermut;     // cudnn support format {oC, iC/groupCount, kH, kW} only, mC = 1, oC = iC (groupCount == iC) that is {iC, mC, kH, kW}
    if(0 == wFormat) {
        wPermut = {2,3,0,1};         // kH, kW, iC, mC -> iC, mC, kH, kW
        gradWPermut = {2,3,0,1};     // iC, mC, kH, kW -> kH, kW, iC, mC
    }
    else if(1 == wFormat) {
        wPermut = {1,0,2,3};         // mC, iC, kH, kW -> iC, mC, kH, kW
        gradWPermut = {1,0,2,3};     // iC, mC, kH, kW -> mC, iC, kH, kW
    }
    else {
        wPermut = {3,0,1,2};         // mC, kH, kW, iC -> iC, mC, kH, kW
        gradWPermut = {1,2,3,0};     // iC, mC, kH, kW -> mC, kH, kW, iC
    }

    NDArray* newGradW   = new NDArray(gradW->ordering(),   {iC, mC, kH, kW}, gradW->dataType(),   gradW->getContext());
    NDArray* newWeights = new NDArray(weights->ordering(), {iC, mC, kH, kW}, weights->dataType(), weights->getContext());

    newWeights->assign(weights->permute(wPermut));

    NDArray* newInput = input;
    NDArray* newGradI = gradI;
    if(paddingMode == 1) // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
        checkConv2dCUDNNPadAsymmetric(newInput, newGradI, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);

    depthwiseConv2dBpCUDNN(block.launchContext(), newInput, newWeights, gradO,   newGradI, newGradW, gradB, kH,kW,sH,sW,pH,pW,dH,dW,paddingMode,isNCHW);

    newGradW->permutei(gradWPermut);
    gradW->assign(newGradW);

    if(newInput != input) {

        if(isNCHW)
            gradI->assign((*newGradI)({0,0,  0,0,  0,gradI->sizeAt(2),  0,gradI->sizeAt(3)}));
        else
            gradI->assign((*newGradI)({0,0,  0,gradI->sizeAt(1),  0,gradI->sizeAt(2),  0,0}));

        delete newInput;
        delete newGradI;
    }

    delete newWeights;
    delete newGradW;

    return Status::OK();
}

PLATFORM_CHECK(depthwise_conv2d_bp, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC] = [iC*mC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

    const int paddingMode = INT_ARG(8);                                             // 0-VALID, 1-SAME, 2-CAUSAL
    const int isNCHW      = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;    // INT_ARG(9): 0-NCHW, 1-NHWC
    const int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;       // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

    const int mC = weights->sizeAt(0 == wFormat ? 3 : 0);

    const bool badInputType   = input->dataType()   != DataType::DOUBLE && input->dataType()   != DataType::FLOAT32 && input->dataType()   != DataType::HALF;
    const bool badWeightsType = weights->dataType() != DataType::DOUBLE && weights->dataType() != DataType::FLOAT32 && weights->dataType() != DataType::HALF;
    const bool badGradOType   = gradO->dataType()   != DataType::DOUBLE && gradO->dataType()   != DataType::FLOAT32 && gradO->dataType()   != DataType::HALF;
    const bool badBiasType    = bias == nullptr ? false : (bias->dataType() != DataType::DOUBLE && bias->dataType() != DataType::FLOAT32 && bias->dataType() != DataType::HALF);

    return mC == 1 && isNCHW && paddingMode != 2 && !badInputType && !badWeightsType && !badGradOType && !badBiasType;
}


}
}
}
