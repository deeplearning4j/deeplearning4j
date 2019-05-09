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
#include <execution/LaunchContext.h>

namespace nd4j {
    namespace ops {

        enum PoolingType {
            MAX_POOL = 0,
            AVG_POOL = 1,
            PNORM_POOL = 2,
        };

        class ConvolutionUtils {
        public:
            static inline void calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode) {
                if(isSameMode > 0) {
                    oH = (int) math::nd4j_ceil<double, double>(iH * 1. / sH);
                    oW = (int) math::nd4j_ceil<double, double>(iW * 1. / sW);
                }
                else {
                    oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
                    oW = (iW - (kW + (kW-1)*(dW-1)) + 2*pW)/sW + 1;
                }
            }

            static inline void calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode) {
                if(!isSameMode) {                                           // valid

                    oD = (iD - (kD + (kD - 1) * (dD - 1)) + 2 * pD) / sD + 1;
                    oH = (iH - (kH + (kH - 1) * (dH - 1)) + 2 * pH) / sH + 1;
                    oW = (iW - (kW + (kW - 1) * (dW - 1)) + 2 * pW) / sW + 1;
                }
                else {                                                      // same

                    oD = (int) nd4j::math::nd4j_ceil<double, double>(iD * 1. / sD);
                    oH = (int) nd4j::math::nd4j_ceil<double, double>(iH * 1. / sH);
                    oW = (int) nd4j::math::nd4j_ceil<double, double>(iW * 1. / sW);
                }
            }

            static inline void calcPadding2D(int& pH, int& pW, int oH, int oW, int iH, int iW, int kH, int kW, int sH, int sW, int dH, int dW) {
                int eKH, eKW;
                if (dH == 1 && dW == 1) {
                    eKH = kH;
                    eKW = kW;
                } else {
                    eKH = kH + (kH - 1) * (dH - 1);
                    eKW = kW + (kW - 1) * (dW - 1);
                }

                pH = ((oH - 1) * sH + eKH - iH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
                pW = ((oW - 1) * sW + eKW - iW) / 2;
            }

            static inline void calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW) {
                int eKD, eKH, eKW;
                if (dD == 1 && dH == 1 && dW == 1) {
                    eKD = kD;
                    eKH = kH;
                    eKW = kW;
                } else {
                    eKD = kD + (kD - 1) * (dD - 1);
                    eKH = kH + (kH - 1) * (dH - 1);
                    eKW = kW + (kW - 1) * (dW - 1);
                }

                pD = ((oD - 1) * sD + eKD - iD) / 2;       // Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
                pH = ((oH - 1) * sH + eKH - iH) / 2;
                pW = ((oW - 1) * sW + eKW - iW) / 2;

            }

            // calculation of output height and width in 2D deconvolution procedure
            static inline void calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode) {
                if (isSameMode) {
                    oH = sH * iH;
                    oW = sW * iW;
                }
                else {
                    int ekH, ekW;
                    if (dH == 1 && dW == 1) {
                        ekH = kH;
                        ekW = kW;
                    } else {
                        ekH = kH + (kH - 1) * (dH - 1);
                        ekW = kW + (kW - 1) * (dW - 1);
                    }

                    oH = sH * (iH - 1) + ekH - 2 * pH;
                    oW = sW * (iW - 1) + ekW - 2 * pW;
                }
            }

            // calculation of output height and width in 3D deconvolution procedure
            static inline void calcOutSizeDeconv3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode) {
                if (isSameMode) {
                    oD = sD * iD;
                    oH = sH * iH;
                    oW = sW * iW;
                }
                else {
                    int ekD, ekH, ekW;
                    if (dD == 1 && dH == 1 && dW == 1) {
                        ekD = kD;
                        ekH = kH;
                        ekW = kW;
                    }
                    else {
                        ekD = kD + (kD - 1) * (dD - 1);
                        ekH = kH + (kH - 1) * (dH - 1);
                        ekW = kW + (kW - 1) * (dW - 1);
                    }
                    oD = sD * (iD - 1) + ekD - 2 * pD;
                    oH = sH * (iH - 1) + ekH - 2 * pH;
                    oW = sW * (iW - 1) + ekW - 2 * pW;
                }
            }

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static inline void getSizesAndIndexesConv2d(const bool isNCHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {
                getSizesAndIndexesConv2d(isNCHW, input.getShapeInfo(), output.getShapeInfo(), bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
            }

            static inline void getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {
                // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
                // weights [kH, kW, iC, oC] always
                // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
                indWkH = 0; indWiC = 2; indWoC = 3;

                if(!isNCHW) {
                    indIOioC = 3; indIiH = 1; indOoH = 1;
                }
                else {
                    indIOioC = 1; indIiH = 2; indOoH = 2;
                }

                bS = inShapeInfo[1];                          // batch size
                iC = inShapeInfo[indIOioC+1];                 // input channels
                iH = inShapeInfo[indIiH+1];                   // input height
                iW = inShapeInfo[indIiH+2];                   // input width
                oC = outShapeInfo[indIOioC+1];                // output channels
                oH = outShapeInfo[indOoH+1];                  // output height
                oW = outShapeInfo[indOoH+2];                  // output width
            }

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static inline void getSizesAndIndexesConv3d(const bool isNCDHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iD, int& iH, int& iW, int& oC, int& oD, int& oH, int& oW, int& indIOioC, int& indIOioD, int& indWiC, int& indWoC, int& indWkD) {
                // input   [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
                // weights [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)
                // output  [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

                indWkD = 0; indWiC = 3; indWoC = 4;
                if(!isNCDHW) {
                    indIOioC = 4; indIOioD = 1;
                }
                else {
                    indIOioC = 1; indIOioD = 2;
                }

                bS = input.sizeAt(0);                          // batch size
                iC = input.sizeAt(indIOioC);                   // input channels
                iD = input.sizeAt(indIOioD);                   // input depth
                iH = input.sizeAt(indIOioD+1);                 // input height
                iW = input.sizeAt(indIOioD+2);                 // input width
                oC = output.sizeAt(indIOioC);                  // output channels
                oD = output.sizeAt(indIOioD);                  // output depth
                oH = output.sizeAt(indIOioD+1);                // output height
                oW = output.sizeAt(indIOioD+2);                // output width

            }

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

            static void conv2d(nd4j::LaunchContext  &context, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void conv2d(nd4j::LaunchContext & block, const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs);

            static void conv2dBP(nd4j::LaunchContext & block, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const std::vector<int>& intArgs);

            static void conv2dBP(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void depthwiseConv2d(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void depthwiseConv2dBP(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void sconv2d(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW);

            static void vol2col(nd4j::LaunchContext & block, const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void col2vol(nd4j::LaunchContext & block, const NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void upsampling2d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW);

            static void upsampling3d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW);

            static void upsampling2dBP(nd4j::LaunchContext & block, const NDArray& gradO, NDArray& gradI, const bool isNCHW);

            static void upsampling3dBP(nd4j::LaunchContext & block, const NDArray& gradO, NDArray& gradI, const bool isNCDHW);

            static void pooling2d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0);

            static void pooling3d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0);

            static void pooling2dBP(nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0);

            static void pooling3dBP(nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0);
    };

}
}
#endif //LIBND4J_CONVOLUTIONS_H
