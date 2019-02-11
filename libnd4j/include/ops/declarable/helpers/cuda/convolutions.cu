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

#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <NDArrayFactory.h>
#include <MmulHelper.h>

namespace nd4j {
    namespace ops {

        void ConvolutionUtils::conv2d(nd4j::graph::LaunchContext& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::conv2d(nd4j::graph::LaunchContext& block, const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs) {

        }

        void ConvolutionUtils::conv2dBP(nd4j::graph::LaunchContext& block, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const std::vector<int>& intArgs) {

        }

        void ConvolutionUtils::conv2dBP(nd4j::graph::LaunchContext& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::depthwiseConv2d(nd4j::graph::LaunchContext& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::depthwiseConv2dBP(nd4j::graph::LaunchContext& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::sconv2d(nd4j::graph::LaunchContext& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::vol2col(nd4j::graph::LaunchContext& block, const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

        }

        void ConvolutionUtils::col2vol(nd4j::graph::LaunchContext& block, const NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

        }

        void ConvolutionUtils::upsampling2d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {

        }

        void ConvolutionUtils::upsampling3d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

        }

        void ConvolutionUtils::upsampling2dBP(nd4j::graph::LaunchContext& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {

        }

        void ConvolutionUtils::upsampling3dBP(nd4j::graph::LaunchContext& block, const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {

        }

        template <typename T>
        static __global__ void global_avg_pooling2d() {

        }

        template <typename T>
        static void _avg_pooling2d(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

        }
        BUILD_SINGLE_TEMPLATE(template void _avg_pooling2d, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), FLOAT_TYPES);

        void ConvolutionUtils::pooling2d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {
            switch (poolingMode) {
                case MAX_POOL: {

                    }
                    break;
                case AVG_POOL: {

                    }
                    break;
                case PNORM_POOL: {

                    }
                    break;
                default:
                    throw std::runtime_error("Pooling2D: Unknown PoolingType used");
            }
        }

        void ConvolutionUtils::pooling3d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }

        void ConvolutionUtils::pooling2dBP(nd4j::graph::LaunchContext& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }

        void ConvolutionUtils::pooling3dBP(nd4j::graph::LaunchContext &block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }
    }
}