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
//

#ifndef SD_CUDNNUTILS_H
#define SD_CUDNNUTILS_H

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <platform_boilerplate.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <dll.h>

#include <cudnn.h>

namespace nd4j {
namespace ops {
namespace platforms {

    DECLARE_PLATFORM(conv2d, ENGINE_CUDA);
    DECLARE_PLATFORM(conv2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(conv3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CUDA);
    DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(batchnorm, ENGINE_CUDA);
    DECLARE_PLATFORM(batchnorm_bp, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
FORCEINLINE cudnnDataType_t cudnnDataType(nd4j::DataType dataType) {
    switch (dataType) {
        case nd4j::DataType::FLOAT32:
            return CUDNN_DATA_FLOAT;
        case nd4j::DataType::DOUBLE:
            return CUDNN_DATA_DOUBLE;
        case nd4j::DataType::HALF:
            return CUDNN_DATA_HALF;
        case nd4j::DataType::INT32:
            return CUDNN_DATA_INT32;
        case nd4j::DataType::INT8:
            return CUDNN_DATA_INT8;
        default:
            throw datatype_exception::build("Unsupported data type", dataType);
    }
}

//////////////////////////////////////////////////////////////////////////
FORCEINLINE void checkConv2dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
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
FORCEINLINE void checkConv3dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
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

}
}
}

#endif //SD_CUDNNUTILS_H
