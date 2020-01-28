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

namespace nd4j      {
namespace ops       {
namespace platforms {

    DECLARE_PLATFORM(conv2d, ENGINE_CUDA);
    DECLARE_PLATFORM(conv2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(conv3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CUDA);
    DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(batchnorm, ENGINE_CUDA);
    DECLARE_PLATFORM(batchnorm_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(avgpool2d, ENGINE_CUDA);
    DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(maxpool2d, ENGINE_CUDA);
    DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(avgpool3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(maxpool3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_CUDA);

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
void checkConv2dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
                                    const int iH, const int iW,
                                    const int oH, const int oW,
                                    const int kH, const int kW,
                                    const int sH, const int sW,
                                    const int pH, const int pW,
                                    const int dH, const int dW,
                                    const bool isNCHW);

//////////////////////////////////////////////////////////////////////////
void checkConv3dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
                                    const int iD, const int iH, const int iW,
                                    const int oD, const int oH, const int oW,
                                    const int kD, const int kH, const int kW,
                                    const int sD, const int sH, const int sW,
                                    const int pD, const int pH, const int pW,
                                    const int dD, const int dH, const int dW,
                                    const bool isNCDHW);

//////////////////////////////////////////////////////////////////////////
void pooling2dCUDNN(const LaunchContext* context,
                    const NDArray* input, NDArray* output,
                    const int kH, const int kW,
                    const int sH, const int sW,
                    const int pH, const int pW,
                    const int dH, const int dW,
                    const bool isNCHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling2dBpCUDNN(const LaunchContext* context,
                      const NDArray* input, const NDArray* gradO,
                            NDArray* gradI,
                      const int kH, const int kW,
                      const int sH, const int sW,
                      const int pH, const int pW,
                      const int dH, const int dW,
                      const bool isNCHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling3dCUDNN(const LaunchContext* context,
                    const NDArray* input, NDArray* output,
                    const int kD, const int kH, const int kW,
                    const int sD, const int sH, const int sW,
                    const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW,
                    const bool isNCDHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling3dBpCUDNN(const LaunchContext* context,
                    const NDArray* input, const NDArray* gradO,
                          NDArray* gradI,
                    const int kD, const int kH, const int kW,
                    const int sD, const int sH, const int sW,
                    const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW,
                    const bool isNCDHW, const cudnnPoolingMode_t mode);

}
}
}

#endif //SD_CUDNNUTILS_H
