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
// Based on PyTorch - https://github.com/pytorch/pytorch
//

#ifndef LIBND4J_CONVOLUTIONS_H
#define LIBND4J_CONVOLUTIONS_H
#include <array/NDArray.h>
#include <execution/LaunchContext.h>
#include <graph/Context.h>

namespace sd {
namespace ops {

enum PoolingType {
  MAX_POOL = 0,
  AVG_POOL = 1,
  PNORM_POOL = 2,
};

class SD_LIB_HIDDEN ConvolutionUtils {
 public:



  static inline LongType outputHeight(const LongType *inputShapeInfo,bool nchw) {
    if(nchw) {
      return shape::sizeAt(inputShapeInfo, 2);
    } else {
      return shape::sizeAt(inputShapeInfo, 1);
    }
  }

  static inline LongType outputWidth(const LongType *inputShapeInfo,bool nchw) {
    if(nchw) {
      return shape::sizeAt(inputShapeInfo, -1);
    } else {
      return shape::sizeAt(inputShapeInfo, -2);
    }
  }

  static inline LongType inputWidth(const LongType *inputShapeInfo,bool nchw) {
    return outputWidth(inputShapeInfo,nchw);
  }

  static inline LongType inputHeight(const LongType *inputShapeInfo,bool nchw) {
   //time series: this will always be 1.
    if(shape::rank(inputShapeInfo) < 4) {
      return 1;
    }
    if(nchw) {
      return shape::sizeAt(inputShapeInfo, 2);
    } else {
      return shape::sizeAt(inputShapeInfo, 1);
    }
  }

  static inline LongType inChannels(const LongType* inputShapeInfo, int weightFormat) {
    if (weightFormat == 0 ) {  // [kH, kW, iC, oC] or
      return shape::sizeAt(inputShapeInfo, -2);
    } else if(weightFormat == 1) { //[oC, iC, kH, kW]
      return shape::sizeAt(inputShapeInfo, -2);
    } else if (weightFormat == 2) {  // [oC, kH, kW, iC]
      return shape::sizeAt(inputShapeInfo, -1);
    } else {
      THROW_EXCEPTION("Unsupported weight format");
    }
  }

  static inline LongType outChannels(const LongType* inputShapeInfo, int weightFormat) {
    if (weightFormat == 0) {  // [kH, kW, iC, oC]
      return shape::sizeAt(inputShapeInfo, -1);
    } else if (weightFormat == 1 || weightFormat == 2) {  // [oC, iC, kH, kW] or [oC, kH, kW, iC]
      return shape::sizeAt(inputShapeInfo, 0);
    } else {
      THROW_EXCEPTION("Unsupported weight format");
    }
  }

  static inline LongType sizeOfOutChannels(const LongType *shapeInfo,LongType weightsFormat) {
    // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
    if (weightsFormat == 0) {
      return shape::sizeAt(shapeInfo, 3);
    } else if (weightsFormat == 1) {
      return shape::sizeAt(shapeInfo, 0);
    } else {
      return shape::sizeAt(shapeInfo, 0);
    }
  }

  static inline LongType sizeOfInChannels(const LongType *shapeInfo,LongType weightsFormat) {
    // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
    if (weightsFormat == 0) {
      return shape::sizeAt(shapeInfo, 2);
    } else if (weightsFormat == 1) {
      return shape::sizeAt(shapeInfo, 1);
    } else {
      return shape::sizeAt(shapeInfo, 3);
    }
  }
  static inline LongType sizeOfKw(const LongType *shapeInfo,LongType weightFormat) {
    // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
    if (weightFormat == 0) {
      return shape::sizeAt(shapeInfo, 1);
    } else if (weightFormat == 1) {
      return shape::sizeAt(shapeInfo, 3);
    } else {
      return shape::sizeAt(shapeInfo, 2);
    }
  }

  static inline LongType sizeOfKh(const LongType *shapeInfo,LongType weightFormat) {
    // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
    if (weightFormat == 0) {
      return shape::sizeAt(shapeInfo, 0);
    } else if (weightFormat == 1) {
      return shape::sizeAt(shapeInfo, 2);
    } else {
      return shape::sizeAt(shapeInfo, 1);
    }
  }

  static inline LongType calcOutDimConv(const LongType inputDim, const LongType kernelDim, const LongType stride,
                                        const LongType padding, const LongType dilation, const int paddingMode) {


    /**
     * Reference:
     * def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Args:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.

    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {"same", "valid", "full", "causal"}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ["same", "causal"]:
        output_length = input_length
    elif padding == "valid":
        output_length = input_length - dilated_filter_size + 1
    elif padding == "full":
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride



     */
    const LongType dilatedKernelDim = kernelDim + (kernelDim - 1) * (dilation - 1);
    LongType outputLength = 0;

    if (paddingMode == 0) {  // valid
      outputLength = inputDim - dilatedKernelDim + 1;
    } else if (paddingMode == 1 || paddingMode == 2) {  // same
      outputLength = inputDim;
    } else {
      THROW_EXCEPTION("Invalid padding type");
    }

    LongType outputDim = sd::math::sd_floordiv<LongType,LongType,LongType>(outputLength + stride - 1, stride);
    return outputDim;
  }


  static inline void calcOutSizePool2D(LongType& oH, LongType& oW, const LongType kH, const LongType kW,
                                       const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                                       const LongType dH, const LongType dW, const LongType iH, const LongType iW,
                                       const int paddingMode) {
    oH = calcOutDimConv(iH, kH, sH, pH, dH, paddingMode);
    oW = calcOutDimConv(iW, kW, sW, pW, dW, paddingMode);
  }

  static inline void calcOutSizePool3D(LongType& oD, LongType& oH, LongType& oW, const LongType kD, const LongType kH, const LongType kW,
                                       const LongType sD, const LongType sH, const LongType sW,  LongType pD,  LongType pH,
                                       LongType pW, const LongType dD, const LongType dH, const LongType dW, const LongType iD,
                                       const LongType iH, const LongType iW, const int paddingMode) {
    if (paddingMode == 0) {  // valid
      oD = (iD + 2 * pD - (kD - 1) * dD - 1) / sD + 1;
      oH = (iH + 2 * pH - (kH - 1) * dH - 1) / sH + 1;
      oW = (iW + 2 * pW - (kW - 1) * dW - 1) / sW + 1;
    } else if (paddingMode == 1) {  // same
      oD = (iD + sD - 1) / sD;
      oH = (iH + sH - 1) / sH;
      oW = (iW + sW - 1) / sW;

      // Calculate the padding needed to achieve the same output size
      LongType paddingNeededD = ((oD - 1) * sD + (kD - 1) * dD + 1 - iD) / 2;
      LongType paddingNeededH = ((oH - 1) * sH + (kH - 1) * dH + 1 - iH) / 2;
      LongType paddingNeededW = ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2;

      // Update the padding values
      pD = paddingNeededD;
      pH = paddingNeededH;
      pW = paddingNeededW;

      // Recalculate the output depth, height, and width with the updated padding
      oD = (iD + 2 * pD - (kD - 1) * dD - 1) / sD + 1;
      oH = (iH + 2 * pH - (kH - 1) * dH - 1) / sH + 1;
      oW = (iW + 2 * pW - (kW - 1) * dW - 1) / sW + 1;
    } else {  // causal
      // Update the padding values for causal convolution
      pD = (kD - 1) * dD;
      pH = (kH - 1) * dH;
      pW = (kW - 1) * dW;

      // Calculate the output depth, height, and width with the updated padding
      oD = (iD + 2 * pD - (kD - 1) * dD - 1) / sD + 1;
      oH = (iH + 2 * pH - (kH - 1) * dH - 1) / sH + 1;
      oW = (iW + 2 * pW - (kW - 1) * dW - 1) / sW + 1;
    }
  }

  static inline void calcPadding2D(LongType& pH, LongType& pW, LongType oH, LongType oW, LongType iH, LongType iW, LongType kH, LongType kW, LongType sH, LongType sW,
                                   LongType dH, LongType dW, const int paddingMode = 1 /* default is same mode*/) {
    if (paddingMode == 0) {  // valid
      pH = 0;
      pW = 0;
    } else if (paddingMode == 1) {  // same
      const int eKH = (kH - 1) * dH + 1;
      const int eKW = (kW - 1) * dW + 1;

      pH = ((oH - 1) * sH + eKH - iH) / 2;
      pW = ((oW - 1) * sW + eKW - iW) / 2;

      // Handle odd padding cases
      int padBottomH = (oH - 1) * sH + eKH - iH - pH;
      int padBottomW = (oW - 1) * sW + eKW - iW - pW;

      // Adjust padding to ensure symmetry
      if (padBottomH != pH) {
        oH -= 1;
        pH = ((oH - 1) * sH + eKH - iH) / 2;
      }
      if (padBottomW != pW) {
        oW -= 1;
        pW = ((oW - 1) * sW + eKW - iW) / 2;
      }
    } else {  // causal
      pH = (kH - 1) * dH;
      pW = (kW - 1) * dW;
    }
  }

  static inline void calcPadding3D(LongType& pD, LongType& pH, LongType& pW,  LongType oD,  LongType oH,  LongType oW, const LongType iD,
                                   const LongType iH, const LongType iW, const LongType kD, const LongType kH, const LongType kW, const LongType sD,
                                   const LongType sH, const LongType sW, const LongType dD, const LongType dH, const LongType dW,
                                   const int paddingMode = 1 /* default is same mode*/) {
    if (paddingMode == 0) {  // valid
      pD = 0;
      pH = 0;
      pW = 0;
    } else if (paddingMode == 1) {  // same
      const int eKD = (kD - 1) * dD + 1;
      const int eKH = (kH - 1) * dH + 1;
      const int eKW = (kW - 1) * dW + 1;

      pD = ((oD - 1) * sD + eKD - iD) / 2;
      pH = ((oH - 1) * sH + eKH - iH) / 2;
      pW = ((oW - 1) * sW + eKW - iW) / 2;

      // Handle odd padding cases
      int padBackD = (oD - 1) * sD + eKD - iD - pD;
      int padBottomH = (oH - 1) * sH + eKH - iH - pH;
      int padBottomW = (oW - 1) * sW + eKW - iW - pW;

      // Adjust padding to ensure symmetry
      if (padBackD != pD) {
        oD -= 1;
        pD = ((oD - 1) * sD + eKD - iD) / 2;
      }
      if (padBottomH != pH) {
        oH -= 1;
        pH = ((oH - 1) * sH + eKH - iH) / 2;
      }
      if (padBottomW != pW) {
        oW -= 1;
        pW = ((oW - 1) * sW + eKW - iW) / 2;
      }
    } else {  // causal
      pD = (kD - 1) * dD;
      pH = (kH - 1) * dH;
      pW = (kW - 1) * dW;
    }
  }

  // calculation of output height and width in 2D deconvolution procedure
  static inline LongType calcOutDimDeconv(const LongType inputDim, const LongType kernelDim, const LongType stride,
                                          const LongType padding, const LongType dilation, const int paddingMode) {
    LongType outputDim;
    const LongType dilatedKernelDim = (kernelDim - 1) * dilation + 1;

    if (paddingMode == 0) {  // valid
      outputDim = stride * (inputDim - 1) + dilatedKernelDim - 2 * padding;
    } else if (paddingMode == 1) {  // same
      outputDim = stride * inputDim;
    } else {  // causal
      const LongType causalPadding = (kernelDim - 1) * dilation;
      outputDim = stride * (inputDim - 1) + dilatedKernelDim - 2 * causalPadding;
    }

    return outputDim;
  }

  static inline void calcOutSizeDeconv2D(LongType& oH, LongType& oW, const LongType kH, const LongType kW,
                                         const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                                         const LongType dH, const LongType dW, const LongType iH, const LongType iW,
                                         const int paddingMode) {
    oH = calcOutDimDeconv(iH, kH, sH, pH, dH, paddingMode);
    oW = calcOutDimDeconv(iW, kW, sW, pW, dW, paddingMode);
  }

  // calculation of output height and width in 3D deconvolution procedure
  static inline void calcOutSizeDeconv3D(LongType& oD, LongType& oH, LongType& oW, const LongType kD, const LongType kH, const LongType kW,
                                         const LongType sD, const LongType sH, const LongType sW,  LongType pD,  LongType pH,
                                         LongType pW, const LongType dD, const LongType dH, const LongType dW, const LongType iD,
                                         const LongType iH, const LongType iW, const int paddingMode) {
    if (paddingMode == 1) {  // same
      oD = sD * (iD - 1) + dD * (kD - 1) + 1 - 2 * pD;
      oH = sH * (iH - 1) + dH * (kH - 1) + 1 - 2 * pH;
      oW = sW * (iW - 1) + dW * (kW - 1) + 1 - 2 * pW;
    } else if (paddingMode == 2) {  // causal
      oD = sD * (iD - 1) + dD * (kD - 1) + 1 - pD;
      oH = sH * (iH - 1) + dH * (kH - 1) + 1 - pH;
      oW = sW * (iW - 1) + dW * (kW - 1) + 1 - pW;
    } else {  // valid
      oD = sD * (iD - 1) + dD * (kD - 1) + 1;
      oH = sH * (iH - 1) + dH * (kH - 1) + 1;
      oW = sW * (iW - 1) + dW * (kW - 1) + 1;
    }
  }

  // evaluates sizes values and indexes using input and output arrays depending on data format
  static inline void getSizesAndIndexesConv2d(const bool isNCHW, const int wFormat, const NDArray& input,
                                              const NDArray& output, LongType& bS, LongType& iC, LongType& iH, LongType& iW, LongType& oC,
                                              LongType& oH, LongType& oW, LongType& indIOioC, LongType& indIiH, LongType& indWiC, LongType& indWoC,
                                              LongType& indWkH, LongType& indOoH) {
    getSizesAndIndexesConv2d(isNCHW, wFormat, input.shapeInfo(), output.shapeInfo(), bS, iC, iH, iW, oC, oH, oW,
                             indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
  }

  static inline void getSizesAndIndexesConv2d(const bool isNCHW, const int wFormat, const LongType* inShapeInfo,
                                              const LongType* outShapeInfo, LongType& bS, LongType& iC, LongType& iH, LongType& iW,
                                              LongType& oC, LongType& oH, LongType& oW, LongType& indIOioC, LongType& indIiH, LongType& indWiC,
                                              LongType& indWoC, LongType& indWkH, LongType& indOoH) {
    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (wFormat = 0), [oC, iC, kH, kW] (wFormat = 1), [oC, kH, kW, iC] (wFormat = 2)
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    if (0 == wFormat) {
      indWkH = 0;
      indWiC = 2;
      indWoC = 3;
    } else if (1 == wFormat) {
      indWkH = 2;
      indWiC = 1;
      indWoC = 0;
    } else {
      indWkH = 1;
      indWiC = 3;
      indWoC = 0;
    }

    if (!isNCHW) {
      indIOioC = 3;
      indIiH = 1;
      indOoH = 1;
    } else {
      indIOioC = 1;
      indIiH = 2;
      indOoH = 2;
    }

    bS = inShapeInfo[1];              // batch size
    iC = inShapeInfo[indIOioC + 1];   // input channels
    iH = inShapeInfo[indIiH + 1];     // input height
    iW = inShapeInfo[indIiH + 2];     // input width
    oC = outShapeInfo[indIOioC + 1];  // output channels
    oH = outShapeInfo[indOoH + 1];    // output height
    oW = outShapeInfo[indOoH + 2];    // output width
  }

  // evaluates sizes values and indexes using input and output arrays depending on data format
  static inline void getSizesAndIndexesConv3d(const bool isNCDHW, const int wFormat, const NDArray& input,
                                              const NDArray& output, LongType& bS, LongType& iC, LongType& iD, LongType& iH, LongType& iW,
                                              LongType& oC, LongType& oD, LongType& oH, LongType& oW, LongType& indIOioC, LongType& indIOioD,
                                              LongType& indWiC, LongType& indWoC, LongType& indWkD) {
    // input   [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    // weights [kD, kH, kW, iC, oC] (wFormat = 0), [oC, iC, kD, kH, kW] (wFormat = 1), [oC, kD, kH, kW, iC] (wFormat =
    // 2) output  [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    if (0 == wFormat) {
      indWkD = 0;
      indWiC = 3;
      indWoC = 4;
    } else if (1 == wFormat) {
      indWkD = 2;
      indWiC = 1;
      indWoC = 0;
    } else {
      indWkD = 1;
      indWiC = 4;
      indWoC = 0;
    }

    if (!isNCDHW) {
      indIOioC = 4;
      indIOioD = 1;
    } else {
      indIOioC = 1;
      indIOioD = 2;
    }

    bS = input.sizeAt(0);              // batch size
    iC = input.sizeAt(indIOioC);       // input channels
    iD = input.sizeAt(indIOioD);       // input depth
    iH = input.sizeAt(indIOioD + 1);   // input height
    iW = input.sizeAt(indIOioD + 2);   // input width
    oC = output.sizeAt(indIOioC);      // output channels
    oD = output.sizeAt(indIOioD);      // output depth
    oH = output.sizeAt(indIOioD + 1);  // output height
    oW = output.sizeAt(indIOioD + 2);  // output width
  }
  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  static std::vector<LongType> expectGrad0Shape(int isNCHW,LongType batchSize, LongType oH, LongType oW, LongType oC) {
    if (isNCHW) {
      return std::vector<LongType>({batchSize, oC, oH, oW});
    } else {
      return std::vector<LongType>({batchSize, oH, oW, oC});
    }

  }

  static std::vector<LongType> expectWeightsShape(const int wFormat, const LongType kH, const LongType kW, const LongType iC,
                                                  const LongType oC) {

    if (0 == wFormat) return std::vector<LongType>({kH, kW, iC, oC});

    if (1 == wFormat) return std::vector<LongType>({oC, iC, kH, kW});

    return std::vector<LongType>({oC, kH, kW, iC});
  }

  static std::vector<LongType> expectWeightsShape(const int wFormat, const LongType kD, const LongType kH, const LongType kW,
                                                  const LongType iC, const LongType oC) {
    if (0 == wFormat) return std::vector<LongType>({kH, kW, iC, oC});

    if (1 == wFormat) return std::vector<LongType>({oC, iC, kH, kW});

    return std::vector<LongType>({oC, kH, kW, iC});
  }

  static void conv2d(sd::graph::Context& block, NDArray* input, NDArray* weights, NDArray* bias,
                     NDArray* output, const LongType kH, const LongType kW, const LongType sH, const LongType sW, LongType pH, LongType pW,
                     const LongType dH, const LongType dW, const int paddingMode, const int isNCHW, const int wFormat);



  static void conv2dBP(sd::graph::Context& block, NDArray* input, NDArray* weights, NDArray* bias,
                       NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const LongType kH, const LongType kW,
                       const LongType sH, const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                       const int isNCHW, const int wFormat);

  static void depthwiseConv2d(sd::graph::Context& block, NDArray* input, NDArray* weights,
                              NDArray* bias, NDArray* output, const LongType kH, const LongType kW, const LongType sH,
                              const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                              const int isNCHW, const int wFormat);

  static void depthwiseConv2dBP(sd::graph::Context& block, NDArray* input, NDArray* weights,
                                NDArray* bias, NDArray* gradO, NDArray* gradI, NDArray* gradW,
                                NDArray* gradB, const LongType kH, const LongType kW, const LongType sH, const LongType sW, LongType pH, LongType pW,
                                const LongType dH, const LongType dW, const int paddingMode, const int isNCHW, const int wFormat);

  static void sconv2d(sd::graph::Context& block, NDArray* input, NDArray* weightsDepth,
                      NDArray* weightsPoint, NDArray* bias, NDArray* output, const LongType kH, const LongType kW,
                      const LongType sH, const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                      const int isNCHW, const int wFormat);

  static void vol2col(graph::Context& block, NDArray* vol, NDArray* col, const LongType sD, const LongType sH,
                      const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD,
                      const LongType dH, const LongType dW);

  static void col2vol(graph::Context& block, const NDArray& col, NDArray& vol, const LongType sD, const LongType sH,
                      const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW);

  static void upsampling2d(graph::Context& block, const NDArray& input, NDArray& output, const LongType factorH,
                           const LongType factorW, const bool isNCHW);

  static void upsampling3d(graph::Context& block, const NDArray& input, NDArray& output, const LongType factorD,
                           const LongType factorH, const LongType factorW, const bool isNCDHW);

  static void upsampling2dBP(graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW);

  static void upsampling3dBP(graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCDHW);

  static void pooling2d(graph::Context& block, const NDArray& input, NDArray& output, const LongType kH, const LongType kW,
                        const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                        const PoolingType poolingMode, const int extraParam0);

  static void pooling3d(graph::Context& block, const NDArray& input, NDArray& output, const LongType kD, const LongType kH,
                        const LongType kW, const LongType sD, const LongType sH, const LongType sW, const LongType pD, const LongType pH,
                        const LongType pW, const LongType dD, const LongType dH, const LongType dW, const int poolingMode,
                        const int extraParam0);

  static void pooling2dBP(graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI,
                          const LongType kH, const LongType kW, const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                          const LongType dH, const LongType dW, const int poolingMode, const int extraParam0);

  static void pooling3dBP(graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI,
                          const LongType kD, const LongType kH, const LongType kW, const LongType sD, const LongType sH, const LongType sW,
                          const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW,
                          const int poolingMode, const int extraParam0);
};

}  // namespace ops
}  // namespace sd
#endif  // LIBND4J_CONVOLUTIONS_H
