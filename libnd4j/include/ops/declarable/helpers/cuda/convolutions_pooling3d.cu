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
#include <helpers/PointersManager.h>
#include <math/templatemath.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void pooling3dCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // x input  is [bS, iC, iD, iH, iW]
    // z output is [bS, iC, oD, oH, oW]

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, kDeff, kHeff, kWeff, iD, iH, iW, kProd;
    __shared__ Nd4jLong zLen, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        zLen = shape::length(zShapeInfo);
        rank = 5;

        kDeff = kD + (kD - 1) * (dD - 1);
        kHeff = kH + (kH - 1) * (dH - 1);
        kWeff = kW + (kW - 1) * (dW - 1);

        iD = xShapeInfo[3];
        iH = xShapeInfo[4];
        iW = xShapeInfo[5];

        kProd = kD * kH * kW;
    }
    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(zInd, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    int dstart = coords[2] * sD - pD;
    int hstart = coords[3] * sH - pH;
    int wstart = coords[4] * sW - pW;
    int dend = dstart + kDeff;
    int hend = hstart + kHeff;
    int wend = wstart + kWeff;

    if(dstart < 0)
        dstart += dD * ((-dstart + dD - 1) / dD);
    if(hstart < 0)
        hstart += dH * ((-hstart + dH - 1) / dH);
    if(wstart < 0)
        wstart += dW * ((-wstart + dW - 1) / dW);
    if(dend > iD)
        dend -= dD * ((dend - iD + dD - 1) / dD);
    if(hend > iH)
        hend -= dH * ((hend - iH + dH - 1) / dH);
    if(wend > iW)
        wend -= dW * ((wend - iW + dW - 1) / dW);


    switch (poolingMode) {

        /*** max ***/
        case 0: {
            T max = -DataTypeUtils::max<T>();
            for (coords[2] = dstart; coords[2] < dend; coords[2] += dD) {
                for (coords[3] = hstart; coords[3] < hend; coords[3] += dH){
                    for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
                        T val = x[shape::getOffset(xShapeInfo, coords)];
                        if (val > max)
                            max = val;
                    }
                }
            }
            z[zOffset] = max;
        }
        break;

        /*** avg ***/
        case 1: {
            T sum = static_cast<T>(0.);
            for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
                for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
                    for (coords[4] = wstart; coords[4] < wend; coords[4] += dW)
                        sum += x[shape::getOffset(xShapeInfo, coords)];

            if (extraParam0 == 0) {         //Exclude padding
                uint a = (dend - dstart) / dD + ((dend - dstart) % dD == 0 ? 0 : 1);
                uint b = (hend - hstart) / dH + ((hend - hstart) % dH == 0 ? 0 : 1);
                uint c = (wend - wstart) / dW + ((wend - wstart) % dW == 0 ? 0 : 1);
                sum /=  static_cast<T>(a * b * c);                                       //  /= sd::math::nd4j_ceil<double,T>(static_cast<double>(dend - dstart) / static_cast<double>(dD)) * sd::math::nd4j_ceil<double,T>(static_cast<double>(hend - hstart) / static_cast<double>(dH)) * sd::math::nd4j_ceil<double,T>(static_cast<double>(wend - wstart) / static_cast<double>(dW));   //Accounts for dilation
            }
            else if (extraParam0 == 1)    //Include padding
                sum /= kProd;

            z[zOffset] = sum;
        }
        break;

        /*** pnorm ***/
        case 2: {
            T sum = static_cast<T>(0.);
            for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
                for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
                    for (coords[4] = wstart; coords[4] < wend; coords[4] += dW)
                        sum += sd::math::nd4j_pow<T,T,T>(sd::math::nd4j_abs<T>(x[shape::getOffset(xShapeInfo, coords)]), extraParam0);

            sum = sd::math::nd4j_pow<T,T,T>(sum, (T) 1.f / extraParam0);

            z[zOffset] = sum;
        }
        break;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling3dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* vx, const Nd4jLong* xShapeInfo,
                                      void* vz, const Nd4jLong* zShapeInfo,
                                const int kD, const int kH, const int kW,
                                const int sD, const int sH, const int sW,
                                const int pD, const int pH, const int pW,
                                const int dD, const int dH, const int dW,
                                const int poolingMode, const int extraParam0) {

    pooling3dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling3d(sd::graph::Context& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    PointersManager manager(block.launchContext(), "pooling3d");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = output.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling3dCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}


}
}
