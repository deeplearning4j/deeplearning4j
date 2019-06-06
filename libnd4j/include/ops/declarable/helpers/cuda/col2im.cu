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
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/col2im.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

//////////////////////////////////////////////////////////////////////////
// [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
template<typename T>
__global__ static void col2imCuda(const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, const int strideY, const int strideX, const int padHeight, const int padWidth, const int imgHeight, const int imgWidth, const int dY, const int dX) {

          const auto dx = reinterpret_cast<const T*>(in);
          auto result = reinterpret_cast<T*>(out);

          auto inShape = shape::shapeOf(const_cast<Nd4jLong *>(inShapeInfo));
          auto inStride = shape::stride(const_cast<Nd4jLong *>(inShapeInfo));

          int strideex = inStride[0];
          int stridech = inStride[1];
          int stridekrow = inStride[2];
          int stridekcol = inStride[3];
          int striderow = inStride[4];
          int stridecol = inStride[5];

          int kernelHeight = inShape[2];
          int kernelWidth = inShape[3];

          auto outShape = shape::shapeOf(const_cast<Nd4jLong *>(outShapeInfo));
          auto resultOrder = shape::order(const_cast<Nd4jLong *>(outShapeInfo));
          auto outStride = shape::stride(const_cast<Nd4jLong *>(outShapeInfo));

          int samples = outShape[0];
          int depth = outShape[1];
          int imgH = outShape[2];
          int imgW = outShape[3];

          int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
          int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

          int n = samples * depth * imgHeight * imgWidth;

          //Effective kernel size, accounting for dilation
          int kEffectiveW = kernelWidth + (kernelWidth - 1) * (dX - 1);
          int kEffectiveH = kernelHeight + (kernelHeight - 1) * (dY - 1);

          for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
              T val = 0;
              int w_im = i % imgWidth + padWidth;
              int h_im = (i / imgWidth) % imgHeight + padHeight;
              int c_im = i / (imgWidth * imgHeight);

              int num_im = c_im / depth;
              int depth_im = c_im % depth;

              // compute the start and end of the output
              // These are the indexes for dimensions ??? in the 6d col matrix
              int w_col_start = (w_im < kEffectiveW) ? 0 : (w_im - kEffectiveW) / strideX + 1;
              int w_col_end = nd4j::math::nd4j_min<int>(w_im / strideX + 1, width_col);

              int h_col_start = (h_im < kEffectiveH) ? 0 : (h_im - kEffectiveH) / strideY + 1;
              int h_col_end = nd4j::math::nd4j_min<int>(h_im / strideY + 1, height_col);


              //Iterate over col entries in the 6d array... these are added up
              for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
                  for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                      int h_k = (h_im - h_col * strideY);
                      int w_k = (w_im - w_col * strideX);

                      if(h_k % dY == 0 && w_k % dX == 0){
                          h_k /= dY;
                          w_k /= dX;

                          int data_col_index = num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;
                          val += dx[data_col_index];
                      }
                  }
              }

              int i_f = 0;
              int i_c = i;
              for (int dim = 3; dim >= 0; dim--) {
                  i_f += (i_c % outShape[dim])  * outStride[dim];
                  i_c = i_c / outShape[dim];
              }

              result[i_f] = val;
          }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void col2imCudaLauncher(nd4j::LaunchContext  &context, const void *x, void *z, const Nd4jLong *xShapeInfo, const Nd4jLong *zShapeInfo, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {
    col2imCuda<T><<<512, 512, 1024, *context.getCudaStream()>>>(x, z, xShapeInfo, zShapeInfo, sH, sW, pH, pW, iH, iW, dH, dW);
}

//////////////////////////////////////////////////////////////////////////
void col2im(nd4j::LaunchContext & context, const NDArray& input, NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {
    
    NDArray::prepareSpecialUse({&output}, {&input});

    BUILD_SINGLE_SELECTOR(output.dataType(), col2imCudaLauncher, (context, input.getSpecialBuffer(), output.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialShapeInfo(), sH, sW, pH, pW, iH, iW, dH, dW), FLOAT_TYPES);

    NDArray::registerSpecialUse({&output}, {&input});
}



BUILD_SINGLE_TEMPLATE(template void col2imCudaLauncher, (nd4j::LaunchContext  &context, const void *x, void *z, const Nd4jLong *xShapeInfo, const Nd4jLong *zShapeInfo, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW), FLOAT_TYPES);

}
}
}