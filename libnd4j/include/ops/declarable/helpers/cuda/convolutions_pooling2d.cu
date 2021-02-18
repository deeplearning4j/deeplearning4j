/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <math/templatemath.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static __global__ void avgPooling2dCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY, strideOX, length, kHEff, kWEff;

    if (threadIdx.x == 0) {
        bS = shape::sizeAt(xShapeInfo, 0);
        iC = shape::sizeAt(xShapeInfo, 1);
        oH = shape::sizeAt(zShapeInfo, 2);
        oW = shape::sizeAt(zShapeInfo, 3);
        iH = shape::sizeAt(xShapeInfo, 2);
        iW = shape::sizeAt(xShapeInfo, 3);

        strideB = shape::stride(xShapeInfo)[0];
        strideC = shape::stride(xShapeInfo)[1];
        strideY = shape::stride(xShapeInfo)[2];
        strideX = shape::stride(xShapeInfo)[3];

        strideOB = shape::stride(zShapeInfo)[0];
        strideOC = shape::stride(zShapeInfo)[1];
        strideOY = shape::stride(zShapeInfo)[2];
        strideOX = shape::stride(zShapeInfo)[3];

        length = shape::length(zShapeInfo);

        //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
        kHEff = kH + (kH-1)*(dH-1);
        kWEff = kW + (kW-1)*(dW-1);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = tid; index < length; index += blockDim.x * gridDim.x) {

        const int pw = index % oW;
        const int ph = (index / oW) % oH;
        const int c = (index / oW / oH) % iC;
        const int n = index / oW / oH / iC;

        int hstart = sH * ph - pH;
        int wstart = sW * pw - pW;
        int hend = hstart + kHEff;
        int wend = wstart + kWEff;

        if(hstart < 0){
            int f = sd::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
            hstart += f * dH;
        }
        if(wstart < 0){
            int f = sd::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if(hend > iH){
            int f = sd::math::nd4j_ceil<Z,int>((Z) (hend-iH) / (Z) dH);
            hend -= f * dH;
        }
        if(wend > iW){
            int f = sd::math::nd4j_ceil<Z,int>((Z) (wend-iW) / (Z) dW);
            wend -= f * dW;
        }

        //Accounts for dilation
        int pool_size = sd::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * sd::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

        Z sum = 0.0f;

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH)
            for (int w = wstart; w < wend; w += dW)
                sum += static_cast<Z>(inSlice[h * strideY + w * strideX]);

        int divide_factor = pool_size;  //Case 0: exclude padding
        if (extraParam0 == 1)     //Case 1: include padding
            divide_factor = kH * kW;

        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = sum / static_cast<Z>(divide_factor);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void avgPooling2dCudaLauncher(sd::LaunchContext & block, const void *vx, const Nd4jLong *vxShapeInfo, void *vz, const Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    avgPooling2dCuda<X, Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static __global__ void pnormPooling2dCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY, strideOX, length, kHEff, kWEff;
    __shared__ bool fOrder;

    if (threadIdx.x == 0) {
        bS = shape::sizeAt(xShapeInfo, 0);
        iC = shape::sizeAt(xShapeInfo, 1);
        oH = shape::sizeAt(zShapeInfo, 2);
        oW = shape::sizeAt(zShapeInfo, 3);
        iH = shape::sizeAt(xShapeInfo, 2);
        iW = shape::sizeAt(xShapeInfo, 3);

        strideB = shape::stride(xShapeInfo)[0];
        strideC = shape::stride(xShapeInfo)[1];
        strideY = shape::stride(xShapeInfo)[2];
        strideX = shape::stride(xShapeInfo)[3];

        strideOB = shape::stride(zShapeInfo)[0];
        strideOC = shape::stride(zShapeInfo)[1];
        strideOY = shape::stride(zShapeInfo)[2];
        strideOX = shape::stride(zShapeInfo)[3];

        length = shape::length(zShapeInfo);

        //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
        kHEff = kH + (kH-1)*(dH-1);
        kWEff = kW + (kW-1)*(dW-1);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = tid; index < length; index += blockDim.x * gridDim.x) {

        const int pw = index % oW;
        const int ph = (index / oW) % oH;
        const int c = (index / oW / oH) % iC;
        const int n = index / oW / oH / iC;

        int hstart = sH * ph - pH;
        int wstart = sW * pw - pW;
        int hend = hstart + kHEff;
        int wend = wstart + kWEff;

        if (hstart < 0) {
            int f = sd::math::nd4j_ceil<Z, int>((Z) -hstart / (Z) dH);
            hstart += f * dH;
        }
        if (wstart < 0) {
            int f = sd::math::nd4j_ceil<Z, int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if (hend > iH) {
            int f = sd::math::nd4j_ceil<Z, int>((Z) (hend - iH) / (Z) dH);
            hend -= f * dH;
        }
        if (wend > iW) {
            int f = sd::math::nd4j_ceil<Z, int>((Z) (wend - iW) / (Z) dW);
            wend -= f * dW;
        }
        //Accounts for dilation
        int pool_size = sd::math::nd4j_ceil<double, int>((double) (hend - hstart) / (double) dH) *
                        sd::math::nd4j_ceil<double, int>((double) (wend - wstart) / (double) dW);

        Z sum = 0.f;

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH)
            for (int w = wstart; w < wend; w += dW)
                sum += sd::math::nd4j_pow<Z, Z, Z>(static_cast<Z>(sd::math::nd4j_abs<X>(inSlice[h * strideY + w * strideX])), extraParam0);

        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = sd::math::nd4j_pow<Z, Z, Z>(sum, (Z) 1.0f / extraParam0);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void pnormPooling2dCudaLauncher(sd::LaunchContext & block, const void *vx, const Nd4jLong *vxShapeInfo, void *vz, const Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    pnormPooling2dCuda<X, Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static __global__ void maxPooling2dCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY, strideOX, length, kHEff, kWEff;
    __shared__ bool fOrder;

    if (threadIdx.x == 0) {
        bS = shape::sizeAt(xShapeInfo, 0);
        iC = shape::sizeAt(xShapeInfo, 1);
        oH = shape::sizeAt(zShapeInfo, 2);
        oW = shape::sizeAt(zShapeInfo, 3);
        iH = shape::sizeAt(xShapeInfo, 2);
        iW = shape::sizeAt(xShapeInfo, 3);

        strideB = shape::stride(xShapeInfo)[0];
        strideC = shape::stride(xShapeInfo)[1];
        strideY = shape::stride(xShapeInfo)[2];
        strideX = shape::stride(xShapeInfo)[3];

        strideOB = shape::stride(zShapeInfo)[0];
        strideOC = shape::stride(zShapeInfo)[1];
        strideOY = shape::stride(zShapeInfo)[2];
        strideOX = shape::stride(zShapeInfo)[3];

        length = shape::length(zShapeInfo);

        //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
        kHEff = kH + (kH-1)*(dH-1);
        kWEff = kW + (kW-1)*(dW-1);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = tid; index < length; index += blockDim.x * gridDim.x) {

        const int pw = index % oW;
        const int ph = (index / oW) % oH;
        const int c = (index / oW / oH) % iC;
        const int n = index / oW / oH / iC;

        int hstart = sH * ph - pH;
        int wstart = sW * pw - pW;
        int hend = hstart + kHEff;
        int wend = wstart + kWEff;

        if(hstart < 0){
            int f = sd::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
            hstart += f * dH;
        }
        if(wstart < 0){
            int f = sd::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if(hend > iH){
            int f = sd::math::nd4j_ceil<Z,int>((Z) (hend-iH) / (Z) dH);
            hend -= f * dH;
        }
        if(wend > iW){
            int f = sd::math::nd4j_ceil<Z,int>((Z) (wend-iW) / (Z) dW);
            wend -= f * dW;
        }
        //Accounts for dilation
        int pool_size = sd::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * sd::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

        Z max = -sd::DataTypeUtils::max<Z>();

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH) {
            for (int w = wstart; w < wend; w += dW) {
                Z v = static_cast<Z>(inSlice[h * strideY + w * strideX]);
                if (v > max)
                    max = v;
            }
        }

        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = max;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void maxPooling2dCudaLauncher(sd::LaunchContext & block, const void *vx, const Nd4jLong *vxShapeInfo, void *vz, const Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    maxPooling2dCuda<X,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling2d(sd::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {

    if(!input.isActualOnDeviceSide()) input.syncToDevice();

    switch (poolingMode) {

        case MAX_POOL: {
                BUILD_SINGLE_SELECTOR_TWICE(input.dataType(), maxPooling2dCudaLauncher, (*block.launchContext(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), NUMERIC_TYPES);
            }
            break;
        case AVG_POOL: {
                BUILD_SINGLE_SELECTOR_TWICE(input.dataType(), avgPooling2dCudaLauncher, (*block.launchContext(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), NUMERIC_TYPES);
            }
            break;
        case PNORM_POOL: {
                BUILD_SINGLE_SELECTOR_TWICE(input.dataType(), pnormPooling2dCudaLauncher, (*block.launchContext(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), FLOAT_TYPES);
            }
            break;
        default:
            throw std::runtime_error("Pooling2D: Unknown PoolingType used");
    }

    output.tickWriteDevice();
    input.tickReadDevice();

    auto result = cudaStreamSynchronize(*block.launchContext()->getCudaStream());
    if (result != 0)
        throw cuda_exception::build("Pooling2D failed", result);
}

}
}
