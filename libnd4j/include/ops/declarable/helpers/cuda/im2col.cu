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
            __global__ void global_im2col(void *dst, void *src, Nd4jLong *zShapeBuffer, Nd4jLong *xShapeBuffer, int kernelHeight, int kernelWidth, int strideY, int strideX, int padHeight, int padWidth, int dY, int dX, double zeroPadValD) {
                int kSize = kernelWidth * kernelHeight;
                T zeroPadVal = static_cast<T>(zeroPadValD);	//Value to use when value is padding. Usually 0 but not always

                auto outShape = shape::shapeOf(zShapeBuffer);
                auto resultOrder = shape::order(zShapeBuffer);
                auto outStride = shape::stride(zShapeBuffer);

                auto inShape = shape::shapeOf(xShapeBuffer);
                auto inStride = shape::stride(xShapeBuffer);

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

                auto result = reinterpret_cast<T*>(dst);

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
                    int h_offset = h_col * strideY - padHeight;
                    int w_offset = w_col * strideX - padWidth;

                    auto data_col_ptr = result;

                    int i_c = (c_col * height_col + h_col) * width_col + w_col;
                    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                    auto data_im_ptr = reinterpret_cast<T*>(src);

                    data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

                    for (int i = 0; i < kernelHeight; ++i) {
                        for (int j = 0; j < kernelWidth; ++j) {
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

                            data_col_ptr += height_col * width_col;
                            i_c += height_col * width_col;
                        }
                    }
                }
            }

            template <typename T>
            _CUDA_H
            void _im2col(nd4j::graph::LaunchContext& context, void *dst, void *src, Nd4jLong *outShape, Nd4jLong *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, double zeroPadVal) {
                global_im2col<T><<<512, 512, 1024, *context.getCudaStream()>>>(dst, src, outShape, inShape, kY, kX, sY, sX, pY, pX, dY, dX, zeroPadVal);
            }
            BUILD_SINGLE_TEMPLATE(template void _im2col, (nd4j::graph::LaunchContext& context, void *result, void *dx, Nd4jLong *zShape, Nd4jLong *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, double zeroPadVal), FLOAT_TYPES);


            void im2col(nd4j::graph::LaunchContext& context, const NDArray& im,  NDArray& col, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const NDArray& arrZeroPadVal) {
                BUILD_SINGLE_SELECTOR(col.dataType(), _im2col, (context, col.getSpecialBuffer(), im.getSpecialBuffer(), col.getSpecialShapeInfo(), im.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, arrZeroPadVal.e<double>(0)), FLOAT_TYPES);
            }
        }
    }
}