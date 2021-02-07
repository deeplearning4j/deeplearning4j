/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>
#include <helpers/PointersManager.h>
#include <math/templatemath.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void pooling2dBPCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // x: input [bS, iC, iH, iW]
    // y: gradO [bS, iC, oH, oW]
    // z: gradI [bS, iC, iH, iW] -> gradI is output in this function

    const T* x = reinterpret_cast<const T*>(vx);
    const T* y = reinterpret_cast<const T*>(vy);
          T* z = reinterpret_cast<T*>(vz);

    Nd4jLong coord2, coord3;
    __shared__ int rank, kHeff, kWeff, iH, iW, kProd;
    __shared__ Nd4jLong yLen, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        yLen = shape::length(yShapeInfo);
        rank = 4;

        kHeff = kH + (kH - 1) * (dH - 1);
        kWeff = kW + (kW - 1) * (dW - 1);

        iH = xShapeInfo[3];
        iW = xShapeInfo[4];

        kProd = kH * kW;
    }
    __syncthreads();

    const auto yInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(yInd >= yLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(yInd, yShapeInfo, coords);

    const auto yOffset = shape::getOffset(yShapeInfo, coords);

    int hstart = coords[2] * sH - pH;
    int wstart = coords[3] * sW - pW;
    int hend = hstart + kHeff;
    int wend = wstart + kWeff;
    if(hstart < 0)
        hstart += dH * ((-hstart + dH - 1) / dH);
    if(wstart < 0)
        wstart += dW * ((-wstart + dW - 1) / dW);
    if(hend > iH)
        hend -= dH * ((hend - iH + dH - 1) / dH);
    if(wend > iW)
        wend -= dW * ((wend - iW + dW - 1) / dW);


    switch (poolingMode) {

        /*** max ***/
        case 0: {
            coord2 = hstart;
            coord3 = wstart;

            T max = -DataTypeUtils::max<T>();
            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH) {
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW){
                    T val = x[shape::getOffset(xShapeInfo, coords)];
                    if (val > max) {
                        max = val;
                        coord2 = coords[2];
                        coord3 = coords[3];
                    }
                }
            }
            coords[2] = coord2;
            coords[3] = coord3;
            auto zOffset = shape::getOffset(zShapeInfo, coords);
            sd::math::atomics::nd4j_atomicAdd<T>(&z[zOffset], y[yOffset]);
            //z[zOffset] += y[yOffset];
        }
        break;

        /*** avg ***/
        case 1: {

            T val = y[yOffset];

            if (extraParam0 == 0)         //Exclude padding
                val /= sd::math::nd4j_ceil<double,T>(static_cast<double>(hend - hstart) / static_cast<double>(dH)) * sd::math::nd4j_ceil<double,T>(static_cast<double>(wend - wstart) / static_cast<double>(dW));   //Accounts for dilation
            else if (extraParam0 == 1)    //Include padding
                val /= kProd;

            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH)
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW)
                    sd::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(zShapeInfo, coords)], val);
        }
        break;

        /*** pnorm ***/
        case 2: {

            T sum = static_cast<T>(0.);
            T val = y[yOffset];

            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH)
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW)
                    sum += sd::math::nd4j_pow<T,T,T>(sd::math::nd4j_abs<T>(x[shape::getOffset(xShapeInfo, coords)]), extraParam0);

            val *= sd::math::nd4j_pow<T,T,T>(sum, ((T)1.f - extraParam0) / extraParam0);

            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH) {
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW) {
                    const auto xOffset = shape::getOffset(xShapeInfo, coords);
                    const auto zOffset = shape::getOffset(zShapeInfo, coords);
                    sd::math::atomics::nd4j_atomicAdd<T>(&z[zOffset], val * sd::math::nd4j_pow<T,T,T>(sd::math::nd4j_abs<T>(x[xOffset]), extraParam0 - 1.f) * sd::math::nd4j_sgn<T,T>(x[xOffset]));
                }
            }
        }
        break;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling2dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                    const void* vx, const Nd4jLong* xShapeInfo,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo,
                                    const int kH, const int kW,
                                    const int sH, const int sW,
                                    const int pH, const int pW,
                                    const int dH, const int dW,
                                    const int poolingMode, const int extraParam0) {

    pooling2dBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling2dBP(sd::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // initial zeroing of gradI
    gradI.nullify();

    PointersManager manager(block.launchContext(), "pooling2dBP");

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (gradO.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = gradO.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&gradI}, {&input, &gradO});
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling2dBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), gradO.specialBuffer(), gradO.specialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), NUMERIC_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&input, &gradO});

    manager.synchronize();
}

}
}