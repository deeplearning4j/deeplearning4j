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
void checkConv2dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
                                            const int iH, const int iW,
                                            const int oH, const int oW,
                                            const int kH, const int kW,
                                            const int sH, const int sW,
                                            const int pH, const int pW,
                                            const int dH, const int dW,
                                            const bool isNCHW) {

    const auto pHsum = ((oH - 1) * sH + ((kH - 1) * dH + 1) - iH);
    const auto pWsum = ((oW - 1) * sW + ((kW - 1) * dW + 1) - iW);

    const bool isPHasymm = pH != (pHsum - pH);
    const bool isPWasymm = pW != (pWsum - pW);

    if(!isPHasymm && !isPWasymm)
        return;

    std::vector<Nd4jLong> newShape = input->getShapeAsVector();

    const int iHposition = isNCHW ? 2 : 1;

    if(isPHasymm)
        newShape[iHposition] += 1;
    if(isPWasymm)
        newShape[iHposition + 1] += 1;

    NDArray* newInput = new NDArray(input->ordering(), newShape, input->dataType(), input->getContext());

    if(isNCHW)
        (*newInput)({0,0,  0,0,  0,input->sizeAt(2),  0,input->sizeAt(3)}).assign(input);
    else
        (*newInput)({0,0,  0,input->sizeAt(1),  0,input->sizeAt(2),  0,0}).assign(input);

    input = newInput;

    if(gradI != nullptr)
        gradI = new NDArray(gradI->ordering(), newShape, gradI->dataType(), gradI->getContext());
}


//////////////////////////////////////////////////////////////////////////
void checkConv3dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
                                            const int iD, const int iH, const int iW,
                                            const int oD, const int oH, const int oW,
                                            const int kD, const int kH, const int kW,
                                            const int sD, const int sH, const int sW,
                                            const int pD, const int pH, const int pW,
                                            const int dD, const int dH, const int dW,
                                            const bool isNCDHW) {

    const auto pDsum = ((oD - 1) * sD + ((kD - 1) * dD + 1) - iD);
    const auto pHsum = ((oH - 1) * sH + ((kH - 1) * dH + 1) - iH);
    const auto pWsum = ((oW - 1) * sW + ((kW - 1) * dW + 1) - iW);

    const bool isPDasymm = pD != (pDsum - pD);
    const bool isPHasymm = pH != (pHsum - pH);
    const bool isPWasymm = pW != (pWsum - pW);

    if(!isPDasymm && !isPHasymm && !isPWasymm)
        return;

    std::vector<Nd4jLong> newShape = input->getShapeAsVector();

    const int iDposition = isNCDHW ? 2 : 1;

    if(isPDasymm)
        newShape[iDposition] += 1;
    if(isPHasymm)
        newShape[iDposition + 1] += 1;
    if(isPWasymm)
        newShape[iDposition + 2] += 1;

    NDArray* newInput = new NDArray(input->ordering(), newShape, input->dataType(), input->getContext());

    if(isNCDHW)
        (*newInput)({0,0,  0,0,  0,input->sizeAt(2),  0,input->sizeAt(3),  0,input->sizeAt(4)}).assign(input);
    else
        (*newInput)({0,0,  0,input->sizeAt(1),  0,input->sizeAt(2),  0,input->sizeAt(3),  0,0}).assign(input);

    input = newInput;

    if(gradI != nullptr)
        gradI = new NDArray(gradI->ordering(), newShape, gradI->dataType(), gradI->getContext());
}

//////////////////////////////////////////////////////////////////////////
void pooling2dCUDNN(const LaunchContext* context,
                    const NDArray* input, NDArray* output,
                    const int kH, const int kW,
                    const int sH, const int sW,
                    const int pH, const int pW,
                    const int dH, const int dW,
                    const bool isNCHW, const cudnnPoolingMode_t mode) {

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("pooling2dCUDNN: can't set stream for cuDNN", err);

    cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(x, format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(x, cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC), input->strideAt(indIiH), input->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("pooling2dCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for input failed", err);

    // output descriptor
    cudnnTensorDescriptor_t z;
    cudnnCreateTensorDescriptor(&z);
    if(output->ews() == 1 && output->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(z, format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
    else
        err = cudnnSetTensor4dDescriptorEx(z, cudnnDataType(output->dataType()), bS, oC, oH, oW, output->strideAt(0), output->strideAt(indIOioC), output->strideAt(indOoH), output->strideAt(indOoH + 1));
    if (err != 0) throw sd::cuda_exception::build("pooling2dCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for output failed", err);

    // description of pooling
    cudnnPoolingDescriptor_t pooling;
    cudnnCreatePoolingDescriptor(&pooling);
    err = cudnnSetPooling2dDescriptor(pooling, mode, CUDNN_PROPAGATE_NAN, kH, kW, pH, pW, sH, sW);
    if (err != 0) throw sd::cuda_exception::build("pooling2dCUDNN: cudnnSetPooling2dDescriptor failed", err);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({output}, {input});

    // run calculation
    err = cudnnPoolingForward(*handle, pooling, alpha, x, input->specialBuffer(), beta, z, output->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("pooling2dCUDNN: cudnnPoolingForward failed", err);

    auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("pooling2dCUDNN: cudaStreamSynchronize failed !", cudaErr);

    NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
void pooling2dBpCUDNN(const LaunchContext* context,
                    const NDArray* input, const NDArray* gradO,
                          NDArray* gradI,
                    const int kH, const int kW,
                    const int sH, const int sW,
                    const int pH, const int pW,
                    const int dH, const int dW,
                    const bool isNCHW, const cudnnPoolingMode_t mode) {

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: can't set stream for cuDNN", err);

    cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input and gradI descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(x, format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
    else
        err = cudnnSetTensor4dDescriptorEx(x, cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC), input->strideAt(indIiH), input->strideAt(indIiH + 1));
    if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for input/gradI failed", err);

    // gradO descriptor
    cudnnTensorDescriptor_t dz;
    cudnnCreateTensorDescriptor(&dz);
    if(gradO->ews() == 1 && gradO->ordering() == 'c')
        err = cudnnSetTensor4dDescriptor(dz, format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
    else
        err = cudnnSetTensor4dDescriptorEx(dz, cudnnDataType(gradO->dataType()), bS, oC, oH, oW, gradO->strideAt(0), gradO->strideAt(indIOioC), gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));
    if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: cudnnSetTensor4dDescriptor/cudnnSetTensor4dDescriptorEx for gradO failed", err);

    // description of pooling
    cudnnPoolingDescriptor_t pooling;
    cudnnCreatePoolingDescriptor(&pooling);
    err = cudnnSetPooling2dDescriptor(pooling, mode, CUDNN_PROPAGATE_NAN, kH, kW, pH, pW, sH, sW);
    if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: cudnnSetPooling2dDescriptor failed", err);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({gradI}, {input, gradO});

    // run calculation for gradI
    err = cudnnPoolingBackward(*handle, pooling, alpha, dz, gradO->specialBuffer(), dz, gradO->specialBuffer(), x, input->specialBuffer(), beta, x, gradI->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: cudnnPoolingBackward failed", err);

    auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("pooling2dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

    NDArray::registerSpecialUse({gradI}, {input, gradO});
}

//////////////////////////////////////////////////////////////////////////
void pooling3dCUDNN(const LaunchContext* context,
                    const NDArray* input, NDArray* output,
                    const int kD, const int kH, const int kW,
                    const int sD, const int sH, const int sW,
                    const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW,
                    const bool isNCDHW, const cudnnPoolingMode_t mode) {

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("pooling3dCUDNN: can't set stream for cuDNN", err);

    const int numDims = 5;

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    const int pSizes[] = {pD, pH, pW};
    const int sSizes[] = {sD, sH, sW};
    const int kSizes[] = {kD, kH, kW};

    const int xShape[] = {bS, iC, iD, iH, iW};
    const int zShape[] = {bS, oC, oD, oH, oW};

    const int xStrides[] = {(int)input->strideAt(0), (int)input->strideAt(1), (int)input->strideAt(2), (int)input->strideAt(3), (int)input->strideAt(4)};
    const int zStrides[] = {(int)output->strideAt(0), (int)output->strideAt(1), (int)output->strideAt(2), (int)output->strideAt(3), (int)output->strideAt(4)};

    cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensorNdDescriptorEx(x, format, cudnnDataType(input->dataType()), numDims, xShape);
    else
        err = cudnnSetTensorNdDescriptor(x, cudnnDataType(input->dataType()), numDims, xShape, xStrides);
    if (err != 0) throw sd::cuda_exception::build("pooling3dCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for input failed", err);

    // output descriptor
    cudnnTensorDescriptor_t z;
    cudnnCreateTensorDescriptor(&z);
    if(output->ews() == 1 && output->ordering() == 'c')
        err = cudnnSetTensorNdDescriptorEx(z, format, cudnnDataType(output->dataType()), numDims, zShape);
    else
        err = cudnnSetTensorNdDescriptor(z, cudnnDataType(output->dataType()), numDims, zShape, zStrides);
    if (err != 0) throw sd::cuda_exception::build("pooling3dCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for output failed", err);

    // description of pooling
    cudnnPoolingDescriptor_t pooling;
    cudnnCreatePoolingDescriptor(&pooling);
    err = cudnnSetPoolingNdDescriptor(pooling, mode, CUDNN_PROPAGATE_NAN, numDims - 2, kSizes, pSizes, sSizes);
    if (err != 0) throw sd::cuda_exception::build("pooling3dCUDNN: cudnnSetPoolingNdDescriptor failed", err);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({output}, {input});

    // run calculation
    err = cudnnPoolingForward(*handle, pooling, alpha, x, input->specialBuffer(), beta, z, output->specialBuffer());
    if (err != 0) throw sd::cuda_exception::build("pooling3dCUDNN: cudnnPoolingForward failed", err);

    auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("pooling3dCUDNN: cudaStreamSynchronize failed !", cudaErr);

    NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
void pooling3dBpCUDNN(const LaunchContext* context,
                    const NDArray* input, const NDArray* gradO,
                          NDArray* gradI,
                    const int kD, const int kH, const int kW,
                    const int sD, const int sH, const int sW,
                    const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW,
                    const bool isNCDHW, const cudnnPoolingMode_t mode) {

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("pooling3dBpCUDNN: can't set stream for cuDNN", err);

    const int numDims = 5;

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    const int pSizes[] = {pD, pH, pW};
    const int sSizes[] = {sD, sH, sW};
    const int kSizes[] = {kD, kH, kW};

    const int xShape[]  = {bS, iC, iD, iH, iW};
    const int dzShape[] = {bS, oC, oD, oH, oW};

    const int xStrides[]  = {(int)input->strideAt(0), (int)input->strideAt(1), (int)input->strideAt(2), (int)input->strideAt(3), (int)input->strideAt(4)};
    const int dzStrides[] = {(int)gradO->strideAt(0), (int)gradO->strideAt(1), (int)gradO->strideAt(2), (int)gradO->strideAt(3), (int)gradO->strideAt(4)};

    cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

    // input and gradI descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1 && input->ordering() == 'c')
        err = cudnnSetTensorNdDescriptorEx(x, format, cudnnDataType(input->dataType()), numDims, xShape);
    else
        err = cudnnSetTensorNdDescriptor(x, cudnnDataType(input->dataType()), numDims, xShape, xStrides);
    if (err != 0) throw sd::cuda_exception::build("pooling3dBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for input/gradI failed", err);

    // gradO descriptor
    cudnnTensorDescriptor_t dz;
    cudnnCreateTensorDescriptor(&dz);
    if(gradO->ews() == 1 && gradO->ordering() == 'c')
        err = cudnnSetTensorNdDescriptorEx(dz, format, cudnnDataType(gradO->dataType()), numDims, dzShape);
    else
        err = cudnnSetTensorNdDescriptor(dz, cudnnDataType(gradO->dataType()), numDims, dzShape, dzStrides);
    if (err != 0) throw sd::cuda_exception::build("pooling3dBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for gradO failed", err);

    // description of pooling
    cudnnPoolingDescriptor_t pooling;
    cudnnCreatePoolingDescriptor(&pooling);
    err = cudnnSetPoolingNdDescriptor(pooling, mode, CUDNN_PROPAGATE_NAN, numDims - 2, kSizes, pSizes, sSizes);
    if (err != 0) throw sd::cuda_exception::build("pooling3dBpCUDNN: cudnnSetPoolingNdDescriptor failed", err);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* alpha = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* beta  = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    // cudnn maxpool2d_bp api requires ff output as one of input arguments
    if(mode == CUDNN_POOLING_MAX) {

        NDArray temp(gradO);

        NDArray::prepareSpecialUse({gradI}, {input, gradO, &temp});

        // run ff calculation
        err = cudnnPoolingForward(*handle, pooling, alpha, x, input->specialBuffer(), beta, dz, temp.specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("pooling3dCUDNN: cudnnPoolingForward failed", err);

        // run bp calculation for gradI
        err = cudnnPoolingBackward(*handle, pooling, alpha, dz, temp.specialBuffer(), dz, gradO->specialBuffer(), x, input->specialBuffer(), beta, x, gradI->specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: cudnnPoolingBackward failed", err);

        NDArray::registerSpecialUse({gradI}, {input, gradO, &temp});
    }
    else {

        NDArray::prepareSpecialUse({gradI}, {input, gradO});

        // run bp calculation for gradI
        err = cudnnPoolingBackward(*handle, pooling, alpha, dz, gradO->specialBuffer(), dz, gradO->specialBuffer(), x, input->specialBuffer(), beta, x, gradI->specialBuffer());
        if (err != 0) throw sd::cuda_exception::build("pooling2dBpCUDNN: cudnnPoolingBackward failed", err);

        NDArray::registerSpecialUse({gradI}, {input, gradO});
    }

    auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("pooling3dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);
}

}
}
}
