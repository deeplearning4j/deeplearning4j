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
#include <exceptions/cuda_exception.h>
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

        template <typename T, typename Z>
        static __global__ void avgPooling2dCuda(void *vx, Nd4jLong *xShapeBuffer, void *vz, Nd4jLong *zShapeBuffer, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
            auto dx = reinterpret_cast<T*>(vx);
            auto result = reinterpret_cast<Z*>(vz);

            __shared__ int batchSize;
            __shared__ int inChannels;
            __shared__ int outH;
            __shared__ int outW;
            __shared__ int inH;
            __shared__ int inW;

            __shared__ int strideB;
            __shared__ int strideC;
            __shared__ int strideY;
            __shared__ int strideX;

            __shared__ int strideOB;
            __shared__ int strideOC;
            __shared__ int strideOY;
            __shared__ int strideOX;

            __shared__ int length;
            __shared__ int kHEff;
            __shared__ int kWEff;
            __shared__ bool fOrder;


            if (threadIdx.x == 0) {
                batchSize = shape::sizeAt(xShapeBuffer, 0);
                inChannels = shape::sizeAt(xShapeBuffer, 1);
                outH = shape::sizeAt(zShapeBuffer, 2);
                outW = shape::sizeAt(zShapeBuffer, 3);
                inH = shape::sizeAt(xShapeBuffer, 2);
                inW = shape::sizeAt(xShapeBuffer, 3);

                strideB = shape::stride(xShapeBuffer)[0];
                strideC = shape::stride(xShapeBuffer)[1];
                strideY = shape::stride(xShapeBuffer)[2];
                strideX = shape::stride(xShapeBuffer)[3];

                strideOB = shape::stride(zShapeBuffer)[0];
                strideOC = shape::stride(zShapeBuffer)[1];
                strideOY = shape::stride(zShapeBuffer)[2];
                strideOX = shape::stride(zShapeBuffer)[3];

                length = shape::length(zShapeBuffer);

                //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
                kHEff = kH + (kH-1)*(dH-1);
                kWEff = kW + (kW-1)*(dW-1);

                fOrder = shape::order(zShapeBuffer) == 'f';
            }
            __syncthreads();

            int tid = blockIdx.x * gridDim.x + threadIdx.x;

            for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
                const int pw = index % outW;
                const int ph = (index / outW) % outH;
                const int c = (index / outW / outH) % inChannels;
                const int n = index / outW / outH / inChannels;
                int hstart = sH * ph - pH;
                int wstart = sW * pw - pW;
                int hend = hstart + kHEff;
                int wend = wstart + kWEff;

                if(hstart < 0){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
                    hstart += f * dH;
                }
                if(wstart < 0){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
                    wstart += f * dW;
                }
                if(hend > inH){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) (hend-inH) / (Z) dH);
                    hend -= f * dH;
                }
                if(wend > inW){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) (wend-inW) / (Z) dW);
                    wend -= f * dW;
                }
                //Accounts for dilation
                int pool_size = nd4j::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * nd4j::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

                Z sum(0.0f);

                T *input_slice = dx + (n * strideB + c * strideC);

                for (int h = hstart; h < hend; h += dH) {
                    for (int w = wstart; w < wend; w += dW) {
                        sum += static_cast<Z>(input_slice[h * strideY + w * strideX]);
                    }
                }

                int divide_factor = pool_size;  //Case 0: exclude padding
                if (extraParam0 == 1)     //Case 1: include padding
                    divide_factor = kH * kW;

                result[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = sum / static_cast<Z>(divide_factor);
            }
        }

        template <typename T, typename Z>
        static __global__ void pnormPooling2dCuda(void *vx, Nd4jLong *xShapeBuffer, void *vz, Nd4jLong *zShapeBuffer, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
            auto dx = reinterpret_cast<T*>(vx);
            auto result = reinterpret_cast<Z*>(vz);

            __shared__ int batchSize;
            __shared__ int inChannels;
            __shared__ int outH;
            __shared__ int outW;
            __shared__ int inH;
            __shared__ int inW;

            __shared__ int strideB;
            __shared__ int strideC;
            __shared__ int strideY;
            __shared__ int strideX;

            __shared__ int strideOB;
            __shared__ int strideOC;
            __shared__ int strideOY;
            __shared__ int strideOX;

            __shared__ int length;
            __shared__ int kHEff;
            __shared__ int kWEff;
            __shared__ bool fOrder;


            if (threadIdx.x == 0) {
                batchSize = shape::sizeAt(xShapeBuffer, 0);
                inChannels = shape::sizeAt(xShapeBuffer, 1);
                outH = shape::sizeAt(zShapeBuffer, 2);
                outW = shape::sizeAt(zShapeBuffer, 3);
                inH = shape::sizeAt(xShapeBuffer, 2);
                inW = shape::sizeAt(xShapeBuffer, 3);

                strideB = shape::stride(xShapeBuffer)[0];
                strideC = shape::stride(xShapeBuffer)[1];
                strideY = shape::stride(xShapeBuffer)[2];
                strideX = shape::stride(xShapeBuffer)[3];

                strideOB = shape::stride(zShapeBuffer)[0];
                strideOC = shape::stride(zShapeBuffer)[1];
                strideOY = shape::stride(zShapeBuffer)[2];
                strideOX = shape::stride(zShapeBuffer)[3];

                length = shape::length(zShapeBuffer);

                //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
                kHEff = kH + (kH-1)*(dH-1);
                kWEff = kW + (kW-1)*(dW-1);

                fOrder = shape::order(zShapeBuffer) == 'f';
            }
            __syncthreads();

            int tid = blockIdx.x * gridDim.x + threadIdx.x;

            for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
                const int pw = index % outW;
                const int ph = (index / outW) % outH;
                const int c = (index / outW / outH) % inChannels;
                const int n = index / outW / outH / inChannels;
                int hstart = sH * ph - pH;
                int wstart = sW * pw - pW;
                int hend = hstart + kHEff;
                int wend = wstart + kWEff;

                if (hstart < 0) {
                    int f = nd4j::math::nd4j_ceil<Z, int>((Z) -hstart / (Z) dH);
                    hstart += f * dH;
                }
                if (wstart < 0) {
                    int f = nd4j::math::nd4j_ceil<Z, int>((Z) -wstart / (Z) dW);
                    wstart += f * dW;
                }
                if (hend > inH) {
                    int f = nd4j::math::nd4j_ceil<Z, int>((Z) (hend - inH) / (Z) dH);
                    hend -= f * dH;
                }
                if (wend > inW) {
                    int f = nd4j::math::nd4j_ceil<Z, int>((Z) (wend - inW) / (Z) dW);
                    wend -= f * dW;
                }
                //Accounts for dilation
                int pool_size = nd4j::math::nd4j_ceil<double, int>((double) (hend - hstart) / (double) dH) *
                                nd4j::math::nd4j_ceil<double, int>((double) (wend - wstart) / (double) dW);

                Z sum(0.0f);

                T *input_slice = dx + (n * strideB + c * strideC);

                for (int h = hstart; h < hend; h += dH) {
                    for (int w = wstart; w < wend; w += dW) {
                        sum += nd4j::math::nd4j_pow<Z, Z, Z>(static_cast<Z>(nd4j::math::nd4j_abs<T>(input_slice[h * strideY + w * strideX])), extraParam0);
                    }
                }

                result[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = nd4j::math::nd4j_pow<Z, Z, Z>(sum, (Z) 1.0f / extraParam0);
            }
        }

        template <typename T, typename Z>
        static __global__ void globalMaxPooling2d(void *vx, Nd4jLong *xShapeBuffer, void *vz, Nd4jLong *zShapeBuffer, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
            auto dx = reinterpret_cast<T*>(vx);
            auto result = reinterpret_cast<Z*>(vz);

            __shared__ int batchSize;
            __shared__ int inChannels;
            __shared__ int outH;
            __shared__ int outW;
            __shared__ int inH;
            __shared__ int inW;

            __shared__ int strideB;
            __shared__ int strideC;
            __shared__ int strideY;
            __shared__ int strideX;

            __shared__ int strideOB;
            __shared__ int strideOC;
            __shared__ int strideOY;
            __shared__ int strideOX;

            __shared__ int length;
            __shared__ int kHEff;
            __shared__ int kWEff;
            __shared__ bool fOrder;


            if (threadIdx.x == 0) {
                batchSize = shape::sizeAt(xShapeBuffer, 0);
                inChannels = shape::sizeAt(xShapeBuffer, 1);
                outH = shape::sizeAt(zShapeBuffer, 2);
                outW = shape::sizeAt(zShapeBuffer, 3);
                inH = shape::sizeAt(xShapeBuffer, 2);
                inW = shape::sizeAt(xShapeBuffer, 3);

                strideB = shape::stride(xShapeBuffer)[0];
                strideC = shape::stride(xShapeBuffer)[1];
                strideY = shape::stride(xShapeBuffer)[2];
                strideX = shape::stride(xShapeBuffer)[3];

                strideOB = shape::stride(zShapeBuffer)[0];
                strideOC = shape::stride(zShapeBuffer)[1];
                strideOY = shape::stride(zShapeBuffer)[2];
                strideOX = shape::stride(zShapeBuffer)[3];

                length = shape::length(zShapeBuffer);

                //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
                kHEff = kH + (kH-1)*(dH-1);
                kWEff = kW + (kW-1)*(dW-1);

                fOrder = shape::order(zShapeBuffer) == 'f';
            }
            __syncthreads();

            int tid = blockIdx.x * gridDim.x + threadIdx.x;

            for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
                const int pw = index % outW;
                const int ph = (index / outW) % outH;
                const int c = (index / outW / outH) % inChannels;
                const int n = index / outW / outH / inChannels;
                int hstart = sH * ph - pH;
                int wstart = sW * pw - pW;
                int hend = hstart + kHEff;
                int wend = wstart + kWEff;

                if(hstart < 0){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
                    hstart += f * dH;
                }
                if(wstart < 0){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
                    wstart += f * dW;
                }
                if(hend > inH){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) (hend-inH) / (Z) dH);
                    hend -= f * dH;
                }
                if(wend > inW){
                    int f = nd4j::math::nd4j_ceil<Z,int>((Z) (wend-inW) / (Z) dW);
                    wend -= f * dW;
                }
                //Accounts for dilation
                int pool_size = nd4j::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * nd4j::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

                Z max = -nd4j::DataTypeUtils::max<Z>();

                T *input_slice = dx + (n * strideB + c * strideC);

                for (int h = hstart; h < hend; h += dH) {
                    for (int w = wstart; w < wend; w += dW) {
                        Z v = static_cast<Z>(input_slice[h * strideY + w * strideX]);
                        if (v > max)
                            max = v;
                    }
                }

                result[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = max;
            }
        }

        template <typename T, typename Z>
        static void _max_pooling2d(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
            globalMaxPooling2d<T,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
        }
        BUILD_DOUBLE_TEMPLATE(template void _max_pooling2d, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);

        template <typename T, typename Z>
        static void _pnorm_pooling2d(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
            pnormPooling2dCuda<T,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
        }
        BUILD_DOUBLE_TEMPLATE(template void _pnorm_pooling2d, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);

        template <typename T, typename Z>
        static void _avg_pooling2d(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
            avgPooling2dCuda<T,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
        }
        BUILD_DOUBLE_TEMPLATE(template void _avg_pooling2d, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);

        void ConvolutionUtils::pooling2d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {
            switch (poolingMode) {
                case MAX_POOL: {
                        BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), _max_pooling2d, (block, input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
                    }
                    break;
                case AVG_POOL: {
                        BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), _avg_pooling2d, (block, input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
                    }
                    break;
                case PNORM_POOL: {
                        BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), _pnorm_pooling2d, (block, input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
                    }
                    break;
                default:
                    throw std::runtime_error("Pooling2D: Unknown PoolingType used");
            }

            output.tickWriteDevice();
            auto result = cudaStreamSynchronize(*block.getCudaStream());
            if (result != 0)
                throw cuda_exception::build("Pooling2D failed", result);
        }

        void ConvolutionUtils::pooling3d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }

        void ConvolutionUtils::pooling2dBP(nd4j::graph::LaunchContext& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }

        void ConvolutionUtils::pooling3dBP(nd4j::graph::LaunchContext &block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }
    }
}