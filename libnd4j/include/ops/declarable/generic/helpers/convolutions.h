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
// Based on PyTorch - https://github.com/pytorch/pytorch
//

#ifndef LIBND4J_CONVOLUTIONS_H
#define LIBND4J_CONVOLUTIONS_H

#include <NDArray.h>
#include <graph/Context.h>

#ifdef HAVE_MKLDNN
#include <helpers/MKLDNNStream.h>
#endif
#include <LaunchContext.h>

namespace nd4j {
    namespace ops {

        class ConvolutionUtils {
        public:
            static void calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode);

            static void calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode);

            static void calcPadding2D(int& pH, int& pW, int oH, int oW, int inH, int inW, int kH, int kW, int sH, int sW, int dH, int dW);

            static void calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW);

            // calculation of output height and width in 2D deconvolution procedure
            static void calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode);
            // calculation of output height and width in 3D deconvolution procedure
            static void calcOutSizeDeconv3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode);

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static void getSizesAndIndexesConv2d(const bool isNCHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH);
            static void getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH);

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static void getSizesAndIndexesConv3d(const bool isNCDHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iD, int& iH, int& iW, int& oC, int& oD, int& oH, int& oW, int& indIOioC, int& indIOioD, int& indWiC, int& indWoC, int& indWkD);

#ifdef HAVE_MKLDNN
            static void getMKLDNNMemoryDescConv2d(
                    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, bool isNCHW,
                    int bS, int iC, int iH, int iW, int oC, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                    const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                    mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
                    mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
                    mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r);

            static void getMKLDNNMemoryDescConv3d(
                    int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, bool isSameMode, bool isNCDHW,
                    int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                    const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                    mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
                    mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
                    mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r);
#endif
            static void conv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void conv2d(nd4j::graph::Context& block, const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs);

            static void conv2dBP(nd4j::graph::Context& block, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const std::vector<int>& intArgs);

            static void conv2dBP(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void depthwiseConv2d(const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void depthwiseConv2dBP(const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void sconv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void vol2col(const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void col2vol(const NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

#ifdef HAVE_MKLDNN
            static void getMKLDNNMemoryDescPool2d(
                    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int poolingMode, int extraParam0, bool isNCHW,
                    int bS, int iC, int iH, int iW, int oC, int oH, int oW,
                    const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                    mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* pool_dst_md, mkldnn::algorithm& algorithm,
                    mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r);

            static void getMKLDNNMemoryDescPool3d(
                    int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, int poolingMode, int extraParam0, bool isNCDHW,
                    int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW,
                    const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                    mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* pool_dst_md, mkldnn::algorithm& algorithm,
                    mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r);
#endif
            static void upsampling2d(const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW);

            static void upsampling3d(const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW);

            static void upsampling2dBP(const NDArray& gradO, NDArray& gradI, const bool isNCHW);

            static void upsampling3dBP(const NDArray& gradO, NDArray& gradI, const bool isNCDHW);

            static void pooling2d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0);

            static void pooling3d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0);

            static void pooling2dBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0);

            static void pooling3dBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0);
    };

}
}
#endif //LIBND4J_CONVOLUTIONS_H
