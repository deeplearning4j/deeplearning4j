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

namespace nd4j {
    namespace ops {

        class ConvolutionUtils {
        public:
            static Nd4jLong convsize(Nd4jLong x, Nd4jLong k, Nd4jLong s, const char* vf);

            template <typename T>
            static Nd4jStatus conv3D(T* output_data, T alpha, T* ptr_input, Nd4jLong nInputDepth, Nd4jLong nInputRows, Nd4jLong nInputCols, T* ptr_weight, Nd4jLong nKernelDepth, Nd4jLong nKernelRows, Nd4jLong nKernelCols, Nd4jLong sdepth, Nd4jLong srow, Nd4jLong scol, const char *vf, const char *xc);

            static Nd4jStatus conv3Dmv(NDArray* r_, double beta, double alpha, NDArray* t_, NDArray* k_, Nd4jLong sdepth, Nd4jLong srow, Nd4jLong scol, const char *vf, const char *xc);

            template <typename T>
            static void fullXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc);

            template <typename T>
            static void fullConv3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc);

            template <typename T>
            static void validXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc);

            template <typename T>
            static void validConv3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc);

            template <typename T>
            static void _dilatedMaxPool3D(T *input_p, T *output_p, T *indz_p, Nd4jLong nslices, Nd4jLong itime, Nd4jLong iwidth, Nd4jLong iheight, Nd4jLong otime, Nd4jLong owidth, Nd4jLong oheight, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH);

            template <typename T>
            static void _dilatedMaxPool3D_bp(T *gradInput_p, T *gradOutput_p, T *indz_p, Nd4jLong nslices, Nd4jLong  itime, Nd4jLong  iwidth, Nd4jLong  iheight, Nd4jLong otime, Nd4jLong owidth, Nd4jLong oheight, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH);

            static void avgPool3D(NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad);

            static void avgPool3DBP(NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad);
            
            static void calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode);

            static void calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode);

            static void calcPadding2D(int& pH, int& pW, int oH, int oW, int inH, int inW, int kH, int kW, int sH, int sW, int dH, int dW);

            static void calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW);

            // calculation of output height and width in 2D deconvolution procedure
            static void calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode);

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static void getSizesAndIndexesConv2d(const bool isNCHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH);
            static void getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH);

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static void getSizesAndIndexesConv3d(const bool isNCDHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iD, int& iH, int& iW, int& oC, int& oD, int& oH, int& oW, int& indIOioC, int& indIOioD, int& indWiC, int& indWoC, int& indWkD);

            static void conv2d(const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs);

            static void conv2dBP(const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const std::vector<int>& intArgs);

            static void depthwiseConv2d(const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs);

            static void depthwiseConv2dBP(const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const std::vector<int>& intArgs);

            static void sconv2d(const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs);

            static void vol2col(NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void col2vol(NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void upsampling2d(const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW);

            static void upsampling3d(const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW);

            static void upsampling2dBP(const NDArray& gradO, NDArray& gradI, const bool isNCHW);

            static void upsampling3dBP(const NDArray& gradO, NDArray& gradI, const bool isNCDHW);

            static void maxPool2d(NDArray* input, NDArray* output, const std::vector<int>& params, NDArray* indices);

            static void pooling3d(NDArray& input, NDArray& output, const void* extraParams);

            static void pooling2d(NDArray& input, NDArray& output, const void* extraParams);

            static void pooling2dBP(NDArray& input, NDArray& gradO, NDArray& gradI, const void* extraParams);

            static void pooling3dBP(NDArray& input, NDArray& gradO, NDArray& gradI, const void* extraParams);

    };

}
}
#endif //LIBND4J_CONVOLUTIONS_H