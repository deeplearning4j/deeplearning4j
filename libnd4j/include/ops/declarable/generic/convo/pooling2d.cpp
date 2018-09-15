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
// Created by raver119 on 08.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_pooling2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(pooling2d, 1, 1, false, 0, 11) {
            // FIXME: this op should be moved to helpers, or removed
            /*
            auto x = INPUT_VARIABLE(0);
            REQUIRE_TRUE(x->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", x->rankOf());
            std::vector<int> argI = *(block.getIArguments());				// 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - pooling mode; 10 - divisor extraParam0 for pnorm case
            auto z = this->getZ(block);

            int kH = argI[0];
            int kW = argI[1];
            int sH = argI[2];
            int sW = argI[3];
            int pH = argI[4];
            int pW = argI[5];
            int dH = argI[6];			//Dilation, height dimension
            int dW = argI[7];			//Dilation, width dimension
            int poolingMode = argI[9];
            int extraParam0 = (int)argI[10];

            REQUIRE_TRUE(dH != 0 && dW != 0, 0, "POOLING2D op: dilation must not be  zero, but got instead {%i, %i}", dH, dW);

            int kSize = kW * kH;

            auto inShape = shape::shapeOf(x->getShapeInfo());
            auto inStride = shape::stride(x->getShapeInfo());

            int samples = (int) inShape[0];
            int depth = (int) inShape[1];
            int height = (int) inShape[2];
            int width = (int) inShape[3];

            int strideex = (int) inStride[0];
            int stridech = (int) inStride[1];
            int strideh = (int) inStride[2];
            int stridew = (int) inStride[3];

            int outH = (z->getShapeInfo())[3];
            int outW = (z->getShapeInfo())[4];
            auto im2colShapeInfo = new Nd4jLong[16] {6, samples, depth, kH, kW, outH, outW, depth*kH*kW*outH*outW, kH*kW*outH*outW, kW*outH*outW, outH*outW, outW, 1, 0, 1, 99};

            auto outShape = shape::shapeOf(im2colShapeInfo);
            auto outStride = shape::stride(im2colShapeInfo);

            int height_col = outShape[4];
            int width_col = outShape[5];

            int n = samples * depth * height_col * width_col;

            int _threads = omp_get_max_threads();
            int span = (n / _threads) + 1;


#pragma omp parallel num_threads(_threads) proc_bind(close)
            {
                int tid = omp_get_thread_num();
                int start = span * tid;
                int end = span * (tid + 1);
                if (end > n) end = n;
                T res;

                for (int index = start; index < end; index++) {
                    int h_index = index / width_col;
                    int h_col = h_index % height_col;
                    int w_col = index % width_col;

                    int c_im = h_index / height_col;
                    int c_col = c_im * kSize;

                    int depth_im = c_im % depth;
                    int num_im = c_im / depth;
                    int h_offset = h_col * sH - pH;
                    int w_offset = w_col * sW - pW;

                    T *data_col_ptr = z->getBuffer();

                    int i_c = (c_col * height_col + h_col) * width_col + w_col;
                    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                    T *data_im_ptr = x->getBuffer();

                    data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset * stridew;
                    res = poolingMode == 0 ? (T) -MAX_FLOAT : (T) 0.0f;

                    for (int i = 0; i < kH; ++i) {
                        for (int j = 0; j < kW; ++j) {
                            int h_im = h_offset + i * dH;
                            int w_im = w_offset + j * dW;
                            int i_f = 0;
                            int i_c_temp = i_c;
                            for (int dim = 5; dim >= 0; dim--) {
                                i_f += (i_c_temp % outShape[dim]) * outStride[dim];
                                i_c_temp = i_c_temp / outShape[dim];
                            }

                            T val;
                            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                val = data_im_ptr[i * dH * strideh + j * dW * stridew];
                            else
                                val = (T) 0.0f;

                            //kernel[i * kH + j] = val;
                            // max
                            if (poolingMode == 0) {
                                if (res < val)
                                    res = val;
                                // avg
                            } else if (poolingMode == 1) {
                                res += val;

                                // phorm
                            } else if (poolingMode == 2) {
                                res += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(val), extraParam0);
                            }

                            //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                            data_col_ptr += height_col * width_col;
                            i_c += height_col * width_col;
                        }
                    }

                    // avg final step
                    if (poolingMode == 1) {
                        res /= kSize;

                        // pnorm final step
                    } else if (poolingMode == 2) {
                        res = nd4j::math::nd4j_pow<T>(res, (T) 1.0f /  extraParam0);
                    }

                    z->putScalar(index,res);
                }
            }
            delete[] im2colShapeInfo;
             */
            return Status::OK();
        }
        DECLARE_SYN(Pooling2D, pooling2d);

        //////////////////////////////////////////////////////////////////////////
        DECLARE_SHAPE_FN(pooling2d) {
            auto inShape = inputShape->at(0);
            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - pooling mode;
            auto argI = *(block.getIArguments());
            int kH = argI[0];
            int kW = argI[1];
            int sH = argI[2];
            int sW = argI[3];
            int pH = argI[4];
            int pW = argI[5];
            int dH = argI[6];
            int dW = argI[7];
            int isSameMode = argI[8];

            REQUIRE_TRUE(dH != 0 && dW != 0, 0, "POOLING2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

            int bS = inShape[1];
            int iD = inShape[2];
            int iH = inShape[3];
            int iW = inShape[4];

            char order = shape::order(inShape); // output order must be equal to input order

            // calculate output Height/Width
            int oH, oW;
            ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
            // allocate memory for new shape
            Nd4jLong* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), 12, Nd4jLong);
            newShapeInfo[0] = 4;		// rank
            newShapeInfo[1] = bS;
            newShapeInfo[2] = iD;
            newShapeInfo[3] = oH;
            newShapeInfo[4] = oW;
            shape::updateStrides(newShapeInfo, order);

            return SHAPELIST(newShapeInfo);
        }
    }
}

#endif