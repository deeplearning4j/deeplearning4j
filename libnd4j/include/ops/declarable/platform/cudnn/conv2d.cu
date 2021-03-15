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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace sd      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void conv2dCUDNN(const LaunchContext* context,
                        const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output,
                        const int kH, const int kW,
                        const int sH, const int sW,
                        const int pH, const int pW,
                        const int dH, const int dW,
                        const int paddingMode, const bool isNCHW, const int wFormat) {

    // cudnn support only two formats for weights {oC,iC,kH,kW} and {oC,kH,kW,iC}

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: can't set stream for cuDNN", err);

    cudnnTensorFormat_t format  = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(x, format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(x, cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC), input->strideAt(indIiH), input->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for input failed", err);

    // weights descriptor
    cudnnFilterDescriptor_t w;
    cudnnCreateFilterDescriptor(&w);
    err = cudnnSetFilter4dDescriptor(w, cudnnDataType(weights->dataType()), formatW, oC, iC, kH, kW);
    if(err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnSetFilter4dDescriptor failed", err);

    // output descriptor
    cudnnTensorDescriptor_t z;
    cudnnCreateTensorDescriptor(&z);
    if(output->ews() == 1 && output->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(z, format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
    else
        err = cudnnSetTensor4dDescriptorEx(z, cudnnDataType(output->dataType()), bS, oC, oH, oW, output->strideAt(0), output->strideAt(indIOioC), output->strideAt(indOoH), output->strideAt(indOoH + 1));
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for output failed", err);

    // description of convolution
    cudnnConvolutionDescriptor_t conv;
    cudnnCreateConvolutionDescriptor(&conv);
    err = cudnnSetConvolution2dDescriptor(conv, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnSetConvolution2dDescriptor failed", err);

    // algorithm description
    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int count = 0;
    //err = cudnnGetConvolutionForwardAlgorithm(*handle, x, w, conv, z, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    err = cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf);
    if (err != 0 || count == 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnGetConvolutionForwardAlgorithm failed", err);
    algo = algoPerf.algo;


    // allocate auxiliary device memory, abbreviation ws means workspace
    size_t wsSize;
    err = cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize);
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnGetConvolutionForwardWorkspaceSize failed", err);
    void* wsData;
    auto cudaErr = cudaMalloc(&wsData, wsSize);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudaMalloc for auxiliary workspace memory failed", cudaErr);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({output}, {input, weights, bias});

    // run calculation
    err = cudnnConvolutionForward(*handle, alpha, x, input->specialBuffer(), w, weights->specialBuffer(), conv, algo, wsData, wsSize, beta, z, output->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnConvolutionForward failed", err);

    // add bias if it is present
    if (bias != nullptr) {
        cudnnTensorDescriptor_t b;
        cudnnCreateTensorDescriptor(&b);
        // err = cudnnSetTensor4dDescriptor(b, format, cudnnDataType(bias->dataType()), 1, isNCHW ? bias->lengthOf() : 1, 1, isNCHW ? 1: bias->lengthOf());
        err = cudnnSetTensor4dDescriptor(b, CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);
        if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnSetTensor4dDescriptor for bias failed", err);
        err = cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha, z, output->specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudnnAddTensor bias failed", err);
    }

    // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    // if (cudaErr != 0)
    //     throw cuda_exception::build("conv2dCUDNN: cudaStreamSynchronize failed !", cudaErr);

    cudaErr = cudaFree(wsData);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv2dCUDNN: cudaFree for auxiliary workspace memory failed", cudaErr);

    NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void conv2dBpCUDNN(const LaunchContext* context,
                          const NDArray* input, const NDArray* weights, const NDArray* gradO,
                          NDArray* gradI, NDArray* gradW, NDArray* gradB,
                          const int kH, const int kW,
                          const int sH, const int sW,
                          const int pH, const int pW,
                          const int dH, const int dW,
                          const int paddingMode, const bool isNCHW, const int wFormat) {

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: can't set stream for cuDNN", err);

    cudnnTensorFormat_t format  = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(x, format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(x, cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC), input->strideAt(indIiH), input->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for input failed", err);

    // gradO descriptor
    cudnnTensorDescriptor_t dz;
    cudnnCreateTensorDescriptor(&dz);
    if(gradO->ews() == 1 && gradO->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(dz, format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
    else
        err = cudnnSetTensor4dDescriptorEx(dz, cudnnDataType(gradO->dataType()), bS, oC, oH, oW, gradO->strideAt(0), gradO->strideAt(indIOioC), gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for gradO failed", err);

    // gradI descriptor
    cudnnTensorDescriptor_t dx;
    cudnnCreateTensorDescriptor(&dx);
    if(gradI->ews() == 1 && gradI->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(dx, format, cudnnDataType(gradI->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(dx, cudnnDataType(gradI->dataType()), bS, iC, iH, iW, gradI->strideAt(0), gradI->strideAt(indIOioC), gradI->strideAt(indIiH), gradI->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for gradI failed", err);

    // gradW descriptor
    cudnnFilterDescriptor_t dw;
    cudnnCreateFilterDescriptor(&dw);
    err = cudnnSetFilter4dDescriptor(dw, cudnnDataType(gradW->dataType()), formatW, oC, iC, kH, kW);
    if(err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnSetFilter4dDescriptor gradW failed", err);

    // description of convolution
    cudnnConvolutionDescriptor_t conv;
    cudnnCreateConvolutionDescriptor(&conv);
    err = cudnnSetConvolution2dDescriptor(conv, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnSetConvolution2dDescriptor failed", err);

    // gradW algorithm description
    cudnnConvolutionBwdFilterAlgo_t algoGradW;
    cudnnConvolutionBwdFilterAlgoPerf_t algoGradWPerf;
    int count = 0;
    //err = cudnnGetConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algoGradW);
    err = cudnnFindConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, 1, &count, &algoGradWPerf);
    if (err != 0 || count == 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnGetConvolutionBackwardFilterAlgorithm failed", err);
    algoGradW = algoGradWPerf.algo;

    // gradI algorithm description
    cudnnConvolutionBwdDataAlgo_t algoGradI;
    cudnnConvolutionBwdDataAlgoPerf_t algoGradIPerf;
    //err = cudnnGetConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algoGradI);
    err = cudnnFindConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, 1, &count, &algoGradIPerf);
    if (err != 0 || count == 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnGetConvolutionBackwardDataAlgorithm failed", err);
    algoGradI = algoGradIPerf.algo;

    // allocate auxiliary device memory for gradW calculation, abbreviation ws means workspace
    size_t wsGradWSize;
    err = cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, x, dz, conv, dw, algoGradW, &wsGradWSize);
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnGetConvolutionBackwardFilterWorkspaceSize failed", err);
    void* wsGradWData;
    auto cudaErr = cudaMalloc(&wsGradWData, wsGradWSize);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudaMalloc for auxiliary workspace memory wsGradWData failed", cudaErr);

    // allocate auxiliary device memory for gradI calculation, abbreviation ws means workspace
    size_t wsGradISize;
    err = cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, dw, dz, conv, dx, algoGradI, &wsGradISize);
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnGetConvolutionBackwardDataWorkspaceSize failed", err);
    void* wsGradIData;
    cudaErr = cudaMalloc(&wsGradIData, wsGradISize);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudaMalloc for auxiliary workspace memory wsGradIData failed", cudaErr);

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
        if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnSetTensor4dDescriptor for gradB failed", err);

        err = cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->specialBuffer(), beta, db, gradB->specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnConvolutionBackwardBias failed", err);
    }

    // run calculation for gradW
    err = cudnnConvolutionBackwardFilter(*handle, alpha, x, input->specialBuffer(), dz, gradO->specialBuffer(), conv, algoGradW, wsGradWData, wsGradWSize, beta, dw, gradW->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnConvolutionBackwardFilter failed", err);

    // run calculation for gradI
    err = cudnnConvolutionBackwardData(*handle, alpha, dw, weights->specialBuffer(), dz, gradO->specialBuffer(), conv, algoGradI, wsGradIData, wsGradISize, beta, dx, gradI->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudnnConvolutionBackwardData failed", err);

    // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    // if (cudaErr != 0)
    //     throw cuda_exception::build("conv2dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

    cudaErr = cudaFree(wsGradWData);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudaFree for auxiliary workspace memory wsGradWData failed", cudaErr);
    cudaErr = cudaFree(wsGradIData);
    if (cudaErr != 0) throw sd::cuda_exception::build("conv2dBpCUDNN: cudaFree for auxiliary workspace memory wsGradIData failed", cudaErr);

    NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                               // 0-VALID, 1-SAME
    bool isNCHW    = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // INT_ARG(9): 0-NCHW, 1-NHWC
    int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;         // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0)); // filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1)); // filter(kernel) width

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D CUDNN OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D CUDNN OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM CONV2D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV2D CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());
        REQUIRE_TRUE((bias->rankOf() == 1 && bias->strideAt(0) == 1) || (bias->rankOf() == 2 && bias->sizeAt(0) == 1 && bias->strideAt(1) == 1) || (bias->rankOf() == 2 && bias->sizeAt(1) == 1 && bias->strideAt(0) == 1), 0, "CUSTOM CONV2D CUDNN OP: bias array should be contiguous in memory !");
    }

    NDArray* newWeights = weights; // cudnn support only two formats {oC,iC,kH,kW} and {oC,kH,kW,iC}
    if(0 == wFormat) {
        newWeights = new NDArray(weights->ordering(), isNCHW ? std::vector<Nd4jLong>({oC, iC, kH, kW}) : std::vector<Nd4jLong>({oC, kH, kW, iC}), weights->dataType(), weights->getContext());
        newWeights->assign(weights->permute(isNCHW ? std::vector<int>({3,2,0,1}) : std::vector<int>({3,0,1,2}))); // (kH, kW, iC, oC  --> oC, iC, kH, kW) or (kH, kW, iC, oC  --> oC, kH, kW, iC)
    }

    NDArray* newInput = input;
    NDArray* newGradI = nullptr;
    if(paddingMode == 1) // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
        checkConv2dCUDNNPadAsymmetric(newInput, newGradI, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);

    conv2dCUDNN(block.launchContext(), newInput, newWeights, bias, output, kH,kW,sH,sW,pH,pW,dH,dW, paddingMode, isNCHW, wFormat);

    if(newInput != input)
        delete newInput;

    if(0 == wFormat)
        delete newWeights;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv2d, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC] always
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    const int paddingMode = INT_ARG(8);                                  // 0-VALID, 1-SAME, 2-CAUSAL

    const bool badInputType   = input->dataType()   != DataType::DOUBLE && input->dataType()   != DataType::FLOAT32 && input->dataType()   != DataType::HALF;
    const bool badWeightsType = weights->dataType() != DataType::DOUBLE && weights->dataType() != DataType::FLOAT32 && weights->dataType() != DataType::HALF;
    const bool badBiasType    = bias == nullptr ? false : (bias->dataType() != DataType::DOUBLE && bias->dataType() != DataType::FLOAT32 && bias->dataType() != DataType::HALF);

    return paddingMode != 2 && !badInputType && !badWeightsType && !badBiasType;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d_bp, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                               // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // INT_ARG(9): 0-NCHW, 1-NHWC
    int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;         // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D_BP CUDNN OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D_BP CUDNN OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 4, 0, "CUSTOM CONV2D_BP CUDNN OP: rank of output's gradients (next epsilon) array must be equal to 4, but got %i instead !", gradO->rankOf());

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

    std::vector<Nd4jLong> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1});
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "CUSTOM CONV2D_BP CUDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM CONV2D_BP CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV2D_BP CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    NDArray *newWeights = weights, *newGradW = gradW; // cudnn support only two formats {oC,iC,kH,kW} and {oC,kH,kW,iC}
    if(0 == wFormat) {
        newGradW   = new NDArray(gradW->ordering(),   isNCHW ? std::vector<Nd4jLong>({oC, iC, kH, kW}) : std::vector<Nd4jLong>({oC, kH, kW, iC}), gradW->dataType(),   gradW->getContext());
        newWeights = new NDArray(weights->ordering(), isNCHW ? std::vector<Nd4jLong>({oC, iC, kH, kW}) : std::vector<Nd4jLong>({oC, kH, kW, iC}), weights->dataType(), weights->getContext());
        newWeights->assign(weights->permute(isNCHW ? std::vector<int>({3,2,0,1}) : std::vector<int>({3,0,1,2}))); // (kH, kW, iC, oC  --> oC, iC, kH, kW) or (kH, kW, iC, oC  --> oC, kH, kW, iC)
    }

    NDArray* newInput = input;
    NDArray* newGradI = gradI;
    if(paddingMode == 1) // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
        checkConv2dCUDNNPadAsymmetric(newInput, newGradI, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);

    conv2dBpCUDNN(block.launchContext(), newInput, newWeights, gradO,   newGradI, newGradW, gradB, kH,kW,sH,sW,pH,pW,dH,dW,paddingMode,isNCHW,wFormat);

    if(0 == wFormat) {
        newGradW->permutei(isNCHW ? std::vector<int>({2,3,1,0}) : std::vector<int>({1,2,3,0})); // (oC, iC, kH, kW --> kH, kW, iC, oC) or (oC, kH, kW, iC --> kH, kW, iC, oC)
        gradW->assign(newGradW);
    }

    if(newInput != input) {

        if(isNCHW)
            gradI->assign((*newGradI)({0,0,  0,0,  0,gradI->sizeAt(2),  0,gradI->sizeAt(3)}));
        else
            gradI->assign((*newGradI)({0,0,  0,gradI->sizeAt(1),  0,gradI->sizeAt(2),  0,0}));

        delete newInput;
        delete newGradI;
    }

    if(0 == wFormat) {
        delete newWeights;
        delete newGradW;
    }

    return Status::OK();
}

PLATFORM_CHECK(conv2d_bp, ENGINE_CUDA) {

    auto input   = INPUT_VARIABLE(0);                                           // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                           // [kH, kW, iC, oC] always
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;             // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

    const int paddingMode = INT_ARG(8);                                             // 0-VALID, 1-SAME, 2-CAUSAL
    const int isNCHW      = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;    // INT_ARG(9): 0-NCHW, 1-NHWC

    const bool badInputType   = input->dataType()   != DataType::DOUBLE && input->dataType()   != DataType::FLOAT32 && input->dataType()   != DataType::HALF;
    const bool badWeightsType = weights->dataType() != DataType::DOUBLE && weights->dataType() != DataType::FLOAT32 && weights->dataType() != DataType::HALF;
    const bool badGradOType   = gradO->dataType()   != DataType::DOUBLE && gradO->dataType()   != DataType::FLOAT32 && gradO->dataType()   != DataType::HALF;
    const bool badBiasType    = bias == nullptr ? false : (bias->dataType() != DataType::DOUBLE && bias->dataType() != DataType::FLOAT32 && bias->dataType() != DataType::HALF);

    return isNCHW && paddingMode != 2 && !badInputType && !badWeightsType && !badGradOType && !badBiasType;
}







// PLATFORM_IMPL(conv2d, ENGINE_CUDA) {

//     auto handle = reinterpret_cast<cudnnHandle_t *>(block.launchContext()->getCuDnnHandle());
//     auto res = cudnnSetStream(*handle, *block.launchContext()->getCudaStream());
//     if (res != 0)
//         throw sd::cuda_exception::build("Can't set stream for cuDNN", res);

//     auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
//     auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC] always
//     auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

//     auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

//     NDArray::prepareSpecialUse({output}, {input, weights, bias});

//     int sH = INT_ARG(2);                                                        // strides height
//     int sW = INT_ARG(3);                                                        // strides width
//     int pH = INT_ARG(4);                                                        // paddings height
//     int pW = INT_ARG(5);                                                        // paddings width
//     int dH = INT_ARG(6);                                                        // dilations height
//     int dW = INT_ARG(7);                                                        // dilations width
//     int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
//     bool isNCHW    = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // INT_ARG(9): 0-NCHW,  1-NHWC

//     int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0)); // filter(kernel) height
//     int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1)); // filter(kernel) width

//     int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
//     int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
//     ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
//     ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, isSameMode);

//     auto dtype = cudnnDataType(input->dataType());


//     cudnnTensorDescriptor_t src;
//     cudnnCreateTensorDescriptor(&src);
//     res = cudnnSetTensor4dDescriptorEx(src, dtype, input->sizeAt(0), input->sizeAt(1), input->sizeAt(2), input->sizeAt(3), input->strideAt(0), input->strideAt(1), input->strideAt(2), input->strideAt(3));
//     if (res != 0)
//         throw sd::cuda_exception::build("cudnnSetTensor4dDescriptorEx src failed", res);

//     // TODO: we definitely want NHWC here as well
//     cudnnFilterDescriptor_t wght;
//     cudnnCreateFilterDescriptor(&wght);
//     res = cudnnSetFilter4dDescriptor(wght, dtype, CUDNN_TENSOR_NCHW, oC, iC, kH, kW);
//     if (res != 0)
//         throw sd::cuda_exception::build("cudnnSetFilter4dDescriptor failed", res);

//     cudnnConvolutionDescriptor_t cdc;
//     cudnnCreateConvolutionDescriptor(&cdc);
//     res = cudnnSetConvolution2dDescriptor(cdc, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, dtype);
//     if (res != 0)
//         throw sd::cuda_exception::build("cudnnSetConvolution2dDescriptor failed", res);

//     cudnnTensorDescriptor_t dst;
//     cudnnCreateTensorDescriptor(&dst);
//     res = cudnnSetTensor4dDescriptorEx(dst, dtype, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3));
//     if (res != 0)
//         throw sd::cuda_exception::build("cudnnSetTensor4dDescriptorEx dst failed", res);

//     // TODO: workspace algorithms are supposed to be faster, so we should use it here if we have enough memory
//     cudnnConvolutionFwdAlgo_t algo;
//     res = cudnnGetConvolutionForwardAlgorithm(*handle, src, wght, cdc, dst, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &algo);
//     if (res != 0)
//         throw sd::cuda_exception::build("cudnnGetConvolutionForwardAlgorithm failed", res);

//     // TODO: should be float if dtype is half/float, and double otherwise
//     float alpha = 1.0f;
//     float beta = 0.0f;
//     res = cudnnConvolutionForward(*handle, &alpha, src, input->specialBuffer(), wght, weights->specialBuffer(), cdc, algo, nullptr, 0, &beta, dst, output->specialBuffer());
//     if (res != 0)
//         throw sd::cuda_exception::build("cudnnConvolutionForward failed", res);


//     if (bias != nullptr) {
//         cudnnTensorDescriptor_t bs;
//         cudnnCreateTensorDescriptor(&bs);
//         if (isNCHW) {
//             res = cudnnSetTensor4dDescriptor(bs, CUDNN_TENSOR_NCHW, dtype, 1, bias->lengthOf(), 1, 1);
//             if (res != 0)
//                 throw sd::cuda_exception::build("cudnnSetTensor4dDescriptorEx bias NHWC failed", res);
//         } else {
//             res = cudnnSetTensor4dDescriptor(bs, CUDNN_TENSOR_NHWC, dtype, 1, 1, 1, bias->lengthOf());
//             if (res != 0)
//                 throw sd::cuda_exception::build("cudnnSetTensor4dDescriptorEx bias NHWC failed", res);
//         }

//         res = cudnnAddTensor(*handle, &alpha, bs, bias->specialBuffer(), &alpha, dst, output->specialBuffer());
//         if (res != 0)
//             throw sd::cuda_exception::build("cudnnAddTensor failed", res);
//     }


//     NDArray::registerSpecialUse({output}, {input, weights, bias});

//     return Status::OK();
// }


}
}
}
