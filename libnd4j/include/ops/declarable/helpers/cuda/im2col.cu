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

#include <ops/declarable/helpers/im2col.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            void _CUDA_G device_im2col(T *result, T *dx, int *resultShapeBuffer, int *xShapeBuffer, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal) {
                int kSize = kX * kY;

                if (threadIdx.x == 0) {
                    printf("WOW!\n");
                }
                __syncthreads();

                int *outShape = shape::shapeOf(resultShapeBuffer);
                char resultOrder = shape::order(resultShapeBuffer);
                int *outStride = shape::stride(resultShapeBuffer);

                int *inShape = shape::shapeOf(xShapeBuffer);
                int *inStride = shape::stride(xShapeBuffer);

                int samples = inShape[0];
                int depth = inShape[1];
                int height = inShape[2];
                int width = inShape[3];

                int strideex = inStride[0];
                int stridech = inStride[1];
                int strideh = inStride[2];
                int stridew = inStride[3];

                int height_col = outShape[4];
                int width_col = outShape[5];

                int n = samples * depth * height_col * width_col;

                int index = blockIdx.x * blockDim.x + threadIdx.x;
                for (; index < n; index += blockDim.x*gridDim.x) {
                    int h_index = index / width_col;
                    int h_col = h_index % height_col;
                    int w_col = index % width_col;

                    int c_im = h_index / height_col;
                    int c_col = c_im * kSize;

                    int depth_im = c_im % depth;
                    int num_im = c_im / depth;
                    int h_offset = h_col * sY - pY;
                    int w_offset = w_col * sX - pX;

                    T* data_col_ptr = result;

                    int i_c = (c_col * height_col + h_col) * width_col + w_col;
                    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                    T* data_im_ptr = dx;

                    data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

                    for (int i = 0; i < kY; ++i) {
                        for (int j = 0; j < kX; ++j) {
                            int h_im = h_offset + i * dY;
                            int w_im = w_offset + j * dX;
                            int i_f = 0;
                            int i_c_temp = i_c;
                            for (int dim = 5; dim >= 0; dim--) {
                                i_f += (i_c_temp % outShape[dim])  * outStride[dim];
                                i_c_temp = i_c_temp / outShape[dim];
                            }
                            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width){
                                result[i_f] = data_im_ptr[i * dY * strideh + j * dX * stridew];
                            } else result[i_f] = zeroPadVal;

                            //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                            data_col_ptr += height_col * width_col;
                            i_c += height_col * width_col;
                        }
                    }
                }
            }

            /*
            extern "C" _CUDA_G void im2colDouble(double *dst, double *src, int *outShape, int *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode) {
                device_im2col<double>(dst, src, outShape, inShape, kY, kX, sY, sX, pY, pX, dY, dX, isSameMode);
            }

            extern "C" _CUDA_G void im2colFloat(float *dst, float *src, int *outShape, int *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode) {
                device_im2col<float>(dst, src, outShape, inShape, kY, kX, sY, sX, pY, pX, dY, dX, isSameMode);
            }

            extern "C" _CUDA_G void im2colHalf(float16 *dst, float16 *src, int *outShape, int *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode) {
                device_im2col<float16>(dst, src, outShape, inShape, kY, kX, sY, sX, pY, pX, dY, dX, isSameMode);
            }
            */


            template <typename T>
            _CUDA_H
            void _im2col(nd4j::graph::LaunchContext& context, T *dst, T *src, int *outShape, int *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode) {
                device_im2col<T><<<512, 512>>>(dst, src, outShape, inShape, kY, kX, sY, sX, pY, pX, dY, dX, isSameMode);
            }


            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *result, float *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *result, float16 *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *result, double *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal);
        }
    }
}