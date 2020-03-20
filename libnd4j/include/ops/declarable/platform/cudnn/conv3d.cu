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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace sd      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void conv3dCUDNN(const LaunchContext* context,
                        const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output,
                        const int kD, const int kH, const int kW,
                        const int sD, const int sH, const int sW,
                        const int pD, const int pH, const int pW,
                        const int dD, const int dH, const int dW,
                        const int paddingMode, const bool isNCDHW, const int wFormat) {

    // cudnn support only one format for weights {oC,iC,kD,kH,kW}

    const int numDims = 5;

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: can't set stream for cuDNN", err);

    const std::vector<int> pads        = {pD, pH, pW};
    const std::vector<int> filtStrides = {sD, sH, sW};
    const std::vector<int> dilations   = {dD, dH, dW};

    const std::vector<int> xShape   = {bS, iC, iD, iH, iW};
    const std::vector<int> zShape   = {bS, oC, oD, oH, oW};
    const std::vector<int> wShape   = {oC, iC, kD, kH, kW};
    const std::vector<int> bShape   = {1, oC, 1, 1, 1};         // {1, (isNCDHW ? oC : 1), 1, 1, (isNCDHW ? 1 : oC)};

    const std::vector<int> xStrides = {(int)input->strideAt(0), (int)input->strideAt(1), (int)input->strideAt(2), (int)input->strideAt(3), (int)input->strideAt(4)};
    const std::vector<int> zStrides = {(int)output->strideAt(0), (int)output->strideAt(1), (int)output->strideAt(2), (int)output->strideAt(3), (int)output->strideAt(4)};

    cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(x, format, cudnnDataType(input->dataType()), numDims, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(x, cudnnDataType(input->dataType()), numDims, xShape.data(), xStrides.data());
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for input failed", err);

    // weights descriptor
    cudnnFilterDescriptor_t w;
    cudnnCreateFilterDescriptor(&w);
    err = cudnnSetFilterNdDescriptor(w, cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, numDims, wShape.data());
    if(err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnSetFilterNdDescriptor failed", err);

    // output descriptor
    cudnnTensorDescriptor_t z;
    cudnnCreateTensorDescriptor(&z);
    if(output->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(z, format, cudnnDataType(output->dataType()), numDims, zShape.data());
    else
        err = cudnnSetTensorNdDescriptor(z, cudnnDataType(output->dataType()), numDims, zShape.data(), zStrides.data());
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for output failed", err);

    // description of convolution
    cudnnConvolutionDescriptor_t conv;
    cudnnCreateConvolutionDescriptor(&conv);
    err = cudnnSetConvolutionNdDescriptor(conv, numDims-2, pads.data(), filtStrides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnSetConvolutionNdDescriptor failed", err);

    // algorithm description
    cudnnConvolutionFwdAlgo_t algo;
    err = cudnnGetConvolutionForwardAlgorithm(*handle, x, w, conv, z, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnGetConvolutionForwardAlgorithm failed", err);

    // allocate auxiliary device memory, abbreviation ws means workspace
    size_t wsSize;
    err = cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize);
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnGetConvolutionForwardWorkspaceSize failed", err);
    void* wsData;
    auto cudaErr = cudaMalloc(&wsData, wsSize);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudaMalloc for auxiliary workspace memory failed", cudaErr);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({output}, {input, weights, bias});

    // run calculation
    err = cudnnConvolutionForward(*handle, alpha, x, input->getSpecialBuffer(), w, weights->getSpecialBuffer(), conv, algo, wsData, wsSize, beta, z, output->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnConvolutionForward failed", err);

    // add bias if it is present
    if (bias != nullptr) {

        cudnnTensorDescriptor_t b;
        cudnnCreateTensorDescriptor(&b);
        err = cudnnSetTensorNdDescriptorEx(b, /*format*/CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), numDims, bShape.data());
        if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnSetTensorNdDescriptor for bias failed", err);
        err = cudnnAddTensor(*handle, alpha, b, bias->getSpecialBuffer(), alpha, z, output->specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudnnAddTensor bias failed", err);
    }

    // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    // if (cudaErr != 0)
    //     throw cuda_exception::build("conv3dCUDNN: cudaStreamSynchronize failed !", cudaErr);

    cudaErr = cudaFree(wsData);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv3dCUDNN: cudaFree for auxiliary workspace memory failed", cudaErr);

    NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void conv3dBpCUDNN(const LaunchContext* context,
                          const NDArray* input, const NDArray* weights, const NDArray* gradO,
                          NDArray* gradI, NDArray* gradW, NDArray* gradB,
                          const int kD, const int kH, const int kW,
                          const int sD, const int sH, const int sW,
                          const int pD, const int pH, const int pW,
                          const int dD, const int dH, const int dW,
                          const int paddingMode, const bool isNCDHW, const int wFormat) {

    // cudnn supports only two formats {oC,iC,kD,kH,kW} and {oC,kD,kH,kW,iC} for weights/gradW

    const int numDims = 5;

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: can't set stream for cuDNN", err);

    const std::vector<int> pads        = {pD, pH, pW};
    const std::vector<int> filtStrides = {sD, sH, sW};
    const std::vector<int> dilations   = {dD, dH, dW};

    const std::vector<int> xShape  = {bS, iC, iD, iH, iW};
    const std::vector<int> dzShape = {bS, oC, oD, oH, oW};
    const std::vector<int> wShape  = {oC, iC, kD, kH, kW};
    const std::vector<int> dbShape = {1, (int)(isNCDHW ? oC : 1), 1, 1, (int)(isNCDHW ? 1 : oC)};

    const std::vector<int> xStrides  = {(int)input->strideAt(0), (int)input->strideAt(1), (int)input->strideAt(2), (int)input->strideAt(3), (int)input->strideAt(4)};
    const std::vector<int> dxStrides = {(int)gradI->strideAt(0), (int)gradI->strideAt(1), (int)gradI->strideAt(2), (int)gradI->strideAt(3), (int)gradI->strideAt(4)};
    const std::vector<int> dzStrides = {(int)gradO->strideAt(0), (int)gradO->strideAt(1), (int)gradO->strideAt(2), (int)gradO->strideAt(3), (int)gradO->strideAt(4)};

    cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(x, format, cudnnDataType(input->dataType()), numDims, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(x, cudnnDataType(input->dataType()), numDims, xShape.data(), xStrides.data());
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for input failed", err);

    // gradO descriptor
    cudnnTensorDescriptor_t dz;
    cudnnCreateTensorDescriptor(&dz);
    if(gradO->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(dz, format, cudnnDataType(gradO->dataType()), numDims, dzShape.data());
    else
        err = cudnnSetTensorNdDescriptor(dz, cudnnDataType(gradO->dataType()), numDims, dzShape.data(), dzStrides.data());
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for gradO failed", err);

    // gradI descriptor
    cudnnTensorDescriptor_t dx;
    cudnnCreateTensorDescriptor(&dx);
    if(gradI->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(dx, format, cudnnDataType(gradI->dataType()), numDims, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(dx, cudnnDataType(gradI->dataType()), numDims, xShape.data(), dxStrides.data());
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for gradI failed", err);

    // gradW descriptor
    cudnnFilterDescriptor_t dw;
    cudnnCreateFilterDescriptor(&dw);
    err = cudnnSetFilterNdDescriptor(dw, cudnnDataType(gradW->dataType()), formatW, numDims, wShape.data());
    if(err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnSetFilterNdDescriptor failed", err);

    // description of convolution
    cudnnConvolutionDescriptor_t conv;
    cudnnCreateConvolutionDescriptor(&conv);
    err = cudnnSetConvolutionNdDescriptor(conv, numDims-2, pads.data(), filtStrides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnSetConvolutionNdDescriptor failed", err);

    // gradW algorithm description
    cudnnConvolutionBwdFilterAlgo_t algoGradW;
    err = cudnnGetConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algoGradW);
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnGetConvolutionBackwardFilterAlgorithm failed", err);

    // gradI algorithm description
    cudnnConvolutionBwdDataAlgo_t algoGradI;
    err = cudnnGetConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algoGradI);
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnGetConvolutionBackwardDataAlgorithm failed", err);

    // allocate auxiliary device memory for gradW calculation, abbreviation ws means workspace
    size_t wsGradWSize;
    err = cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, x, dz, conv, dw, algoGradW, &wsGradWSize);
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnGetConvolutionBackwardFilterWorkspaceSize failed", err);
    void* wsGradWData;
    auto cudaErr = cudaMalloc(&wsGradWData, wsGradWSize);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudaMalloc for auxiliary workspace memory wsGradWData failed", cudaErr);

    // allocate auxiliary device memory for gradI calculation, abbreviation ws means workspace
    size_t wsGradISize;
    err = cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, dw, dz, conv, dx, algoGradI, &wsGradISize);
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnGetConvolutionBackwardDataWorkspaceSize failed", err);
    void* wsGradIData;
    cudaErr = cudaMalloc(&wsGradIData, wsGradISize);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudaMalloc for auxiliary workspace memory wsGradIData failed", cudaErr);

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
        err = cudnnSetTensorNdDescriptorEx(db, format, cudnnDataType(gradB->dataType()), numDims, dbShape.data());
        if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnSetTensorNdDescriptor for gradB failed", err);

        err = cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->getSpecialBuffer(), beta, db, gradB->getSpecialBuffer());
        if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnConvolutionBackwardBias failed", err);
    }

    // run calculation for gradW
    err = cudnnConvolutionBackwardFilter(*handle, alpha, x, input->getSpecialBuffer(), dz, gradO->getSpecialBuffer(), conv, algoGradW, wsGradWData, wsGradWSize, beta, dw, gradW->getSpecialBuffer());
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnConvolutionBackwardFilter failed", err);

    // run calculation for gradI
    err = cudnnConvolutionBackwardData(*handle, alpha, dw, weights->getSpecialBuffer(), dz, gradO->getSpecialBuffer(), conv, algoGradI, wsGradIData, wsGradISize, beta, dx, gradI->getSpecialBuffer());
    if (err != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudnnConvolutionBackwardData failed", err);

    // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    // if (cudaErr != 0)
    //     throw cuda_exception::build("conv3dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

    cudaErr = cudaFree(wsGradWData);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudaFree for auxiliary workspace memory wsGradWData failed", cudaErr);
    cudaErr = cudaFree(wsGradIData);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv3dBpCUDNN: cudaFree for auxiliary workspace memory wsGradIData failed", cudaErr);

    NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv3dnew, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CONV3D CUDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CONV3D CUDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

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
    int paddingMode = INT_ARG(12);                                              // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;         // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    REQUIRE_TRUE(paddingMode < 2, 0, "CONV3D CUDNN OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW, paddingMode);

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CONV3D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CONV3D CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    NDArray* newWeights = weights; // cudnn support only one format {oC,iC,kD,kH,kW}
    if(1 != wFormat) {
        newWeights = new NDArray(weights->ordering(), {oC, iC, kD, kH, kW}, weights->dataType(), weights->getContext());
        newWeights->assign(weights->permute(0 == wFormat ? std::vector<int>({4,3,0,1,2}) : std::vector<int>({0,4,1,2,3})));  // kD, kH, kW, iC, oC  --> oC, iC, kD, kH, kW   or oC, kD, kH, kW, iC  --> oC, iC, kD, kH, kW
    }

    NDArray* newInput = input;
    NDArray* newGradI = nullptr;
    if(paddingMode == 1) // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
        checkConv3dCUDNNPadAsymmetric(newInput, newGradI, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW);

    conv3dCUDNN(block.launchContext(), newInput, newWeights, bias, output, kD,kH,kW,sD,sH,sW,pD,pH,pW,dD,dH,dW, paddingMode, isNCDHW, wFormat);

    if(newInput != input)
        delete newInput;

    if(1 != wFormat)
        delete newWeights;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv3dnew, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    int paddingMode = INT_ARG(12);                                       // 0-SAME,  1-VALID

    const bool badInputType   = input->dataType()   != DataType::DOUBLE && input->dataType()   != DataType::FLOAT32 && input->dataType()   != DataType::HALF;
    const bool badWeightsType = weights->dataType() != DataType::DOUBLE && weights->dataType() != DataType::FLOAT32 && weights->dataType() != DataType::HALF;
    const bool badBiasType    = bias == nullptr ? false : (bias->dataType() != DataType::DOUBLE && bias->dataType() != DataType::FLOAT32 && bias->dataType() != DataType::HALF);

    return paddingMode != 2 && !badInputType && !badWeightsType && !badBiasType;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv3dnew_bp, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CONV3D_BP CUDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CONV3D_BP CUDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 5, 0,   "CONV3D_BP CUDNN OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !", gradO->rankOf());

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
    int paddingMode = INT_ARG(12);                                              // 1-SAME,  0-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;         // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

    REQUIRE_TRUE(paddingMode < 2, 0, "CONV3D_BP CUDNN OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");

    std::vector<Nd4jLong> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2});
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "CONV3D_BP CUDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(gradW->isSameShape(expectedWeightsShape), 0, "CONV3D_BP CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CONV3D_BP CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW, paddingMode);

    NDArray *newWeights = weights, *newGradW = gradW; // cudnn support only two formats {oC,iC,kD,kH,kW} and {oC,kD,kH,kW,iC}
    if(0 == wFormat) {
        newGradW   = new NDArray(gradW->ordering(),   isNCDHW ? std::vector<Nd4jLong>({oC, iC, kD, kH, kW}) : std::vector<Nd4jLong>({oC, kD, kH, kW, iC}), gradW->dataType(),   gradW->getContext());
        newWeights = new NDArray(weights->ordering(), isNCDHW ? std::vector<Nd4jLong>({oC, iC, kD, kH, kW}) : std::vector<Nd4jLong>({oC, kD, kH, kW, iC}), weights->dataType(), weights->getContext());
        newWeights->assign(weights->permute(isNCDHW ? std::vector<int>({4,3,0,1,2}) : std::vector<int>({4,0,1,2,3}))); // (kD, kH, kW, iC, oC  --> oC, iC, kD, kH, kW) or (kD, kH, kW, iC, oC  --> oC, kD, kH, kW, iC)
    }

    NDArray* newInput = input;
    NDArray* newGradI = gradI;
    if(paddingMode == 1) // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
        checkConv3dCUDNNPadAsymmetric(newInput, newGradI, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW);

    conv3dBpCUDNN(block.launchContext(), newInput, newWeights, gradO,   newGradI, newGradW, gradB, kD,kH,kW,sD,sH,sW,pD,pH,pW,dD,dH,dW,paddingMode,isNCDHW,wFormat);

    if(0 == wFormat) {
        newGradW->permutei(isNCDHW ? std::vector<int>({2,3,4,1,0}) : std::vector<int>({1,2,3,4,0})); // (oC, iC, kD, kH, kW --> kD, kH, kW, iC, oC) or (oC, kD, kH, kW, iC --> kD, kH, kW, iC, oC)
        gradW->assign(newGradW);
    }


    if(newInput != input) {

        if(isNCDHW)
            gradI->assign((*newGradI)({0,0,  0,0,  0,gradI->sizeAt(2),  0,gradI->sizeAt(3),  0,gradI->sizeAt(4)}));
        else
            gradI->assign((*newGradI)({0,0,  0,gradI->sizeAt(1),  0,gradI->sizeAt(2),  0,gradI->sizeAt(3),  0,0}));

        delete newInput;
        delete newGradI;
    }

    if(0 == wFormat) {
        delete newWeights;
        delete newGradW;
    }

    return Status::OK();
}

PLATFORM_CHECK(conv3dnew_bp, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    int paddingMode = INT_ARG(12);                                              // 1-SAME,  0-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW

    const bool badInputType   = input->dataType()   != DataType::DOUBLE && input->dataType()   != DataType::FLOAT32 && input->dataType()   != DataType::HALF;
    const bool badWeightsType = weights->dataType() != DataType::DOUBLE && weights->dataType() != DataType::FLOAT32 && weights->dataType() != DataType::HALF;
    const bool badGradOType   = gradO->dataType()   != DataType::DOUBLE && gradO->dataType()   != DataType::FLOAT32 && gradO->dataType()   != DataType::HALF;
    const bool badBiasType    = bias == nullptr ? false : (bias->dataType() != DataType::DOUBLE && bias->dataType() != DataType::FLOAT32 && bias->dataType() != DataType::HALF);

    return isNCDHW && paddingMode != 2 && !badInputType && !badWeightsType && !badGradOType && !badBiasType;
}

}
}
}
