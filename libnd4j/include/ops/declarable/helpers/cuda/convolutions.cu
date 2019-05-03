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
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
template <typename T>
static __global__ void vol2colCuda(const void* volume, const Nd4jLong* volShapeInfo, void* column, const Nd4jLong* colShapeInfo,  const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const T* vol = reinterpret_cast<const T*>(volume);
          T* col = reinterpret_cast<T*>(column);

    const int volRank = 5;
    const int colRank = 8;

    __shared__ Nd4jLong colLen, bS, iC, iD, iH, iW, kD, kH, kW, oD, oH, oW, colStride0, colStride1, colStride2, colStride3, colStride4, colStride5, colStride6, colStride7, volStride0, volStride1, volStride2, volStride3, volStride4;

    if (threadIdx.x == 0) {

        colLen = shape::length(colShapeInfo);

        bS = volShapeInfo[1];
        iC = volShapeInfo[2];
        iD = volShapeInfo[3];
        iH = volShapeInfo[4];
        iW = volShapeInfo[5];
        kD = colShapeInfo[3];
        kH = colShapeInfo[4];
        kW = colShapeInfo[5];
        oD = colShapeInfo[6];
        oH = colShapeInfo[7];
        oW = colShapeInfo[8];

        volStride0 = volShapeInfo[volRank + 1];
        volStride1 = volShapeInfo[volRank + 2];
        volStride2 = volShapeInfo[volRank + 3];
        volStride3 = volShapeInfo[volRank + 4];
        volStride4 = volShapeInfo[volRank + 5];
        colStride0 = colShapeInfo[colRank + 1];
        colStride1 = colShapeInfo[colRank + 2];
        colStride2 = colShapeInfo[colRank + 3];
        colStride3 = colShapeInfo[colRank + 4];
        colStride4 = colShapeInfo[colRank + 5];
        colStride5 = colShapeInfo[colRank + 6];
        colStride6 = colShapeInfo[colRank + 7];
        colStride7 = colShapeInfo[colRank + 8];
    }
    
    __syncthreads();
               
    const int ind = blockDim.x * blockIdx.x + threadIdx.x;
    if(ind >= colLen) return;

    int temp = ind;

    // const int colW = temp % oW; temp /= oW;
    // const int colH = temp % oH; temp /= oH;
    // const int colD = temp % oD; temp /= oD;
    // const int kCol = temp % kW; temp /= kW;
    // const int kRow = temp % kH; temp /= kH;
    // const int kDep = temp % kD; temp /= kD;
    // const int c    = temp % iC; temp /= iC;
    // const int b    = temp;

    Nd4jLong coord[colRank];
    shape::index2coords(volRank, volShapeInfo+1, ind, colLen, coord);

    const int colW = coord[7];
    const int colH = coord[6];
    const int colD = coord[5];
    const int kCol = coord[4];
    const int kRow = coord[3];
    const int kDep = coord[2];
    const int c    = coord[1];
    const int b    = coord[0];

    const int volDep = (-pD + kDep * dD) + colD * sD;
    const int volRow = (-pH + kRow * dH) + colH * sH;
    const int volCol = (-pW + kCol * dW) + colW * sW;
    
    const T* pVol = vol + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
          T* pCol = col + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                                    
    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
        *pCol = 0.f;
    else 
        *pCol = *pVol;    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void vol2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* volume, const Nd4jLong* volShapeInfo, void* column, const Nd4jLong* colShapeInfo, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
    
    vol2colCuda<T><<<blocksPerGrid, threadsPerBlock, 4192, *stream>>>(volume, volShapeInfo, column, colShapeInfo,  sD, sH, sW, pD, pH, pW, dD, dH, dW);
}  

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::vol2col(nd4j::graph::LaunchContext& context, const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
    
    if(!vol.isActualOnDeviceSide()) vol.syncToDevice();

    const int threadsPerBlock = MAX_NUM_THREADS;
    const int blocksPerGrid = (col.lengthOf() + threadsPerBlock - 1) / threadsPerBlock; // ceil

    BUILD_SINGLE_SELECTOR(vol.dataType(), vol2colCudaLauncher, (blocksPerGrid, threadsPerBlock, context.getCudaStream(), vol.getSpecialBuffer(), vol.getSpecialShapeInfo(), col.getSpecialBuffer(), col.getSpecialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);

    vol.tickReadDevice();
    col.tickWriteDevice();
}


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

    int tid = blockIdx.x * gridDim.x + threadIdx.x;

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
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
            hstart += f * dH;
        }
        if(wstart < 0){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if(hend > iH){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (hend-iH) / (Z) dH);
            hend -= f * dH;
        }
        if(wend > iW){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (wend-iW) / (Z) dW);
            wend -= f * dW;
        }
    
        //Accounts for dilation
        int pool_size = nd4j::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * nd4j::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

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
static void avgPooling2dCudaLauncher(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
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

    int tid = blockIdx.x * gridDim.x + threadIdx.x;

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
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) -hstart / (Z) dH);
            hstart += f * dH;
        }
        if (wstart < 0) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if (hend > iH) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) (hend - iH) / (Z) dH);
            hend -= f * dH;
        }
        if (wend > iW) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) (wend - iW) / (Z) dW);
            wend -= f * dW;
        }
        //Accounts for dilation
        int pool_size = nd4j::math::nd4j_ceil<double, int>((double) (hend - hstart) / (double) dH) *
                        nd4j::math::nd4j_ceil<double, int>((double) (wend - wstart) / (double) dW);

        Z sum = 0.f;

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH) 
            for (int w = wstart; w < wend; w += dW) 
                sum += nd4j::math::nd4j_pow<Z, Z, Z>(static_cast<Z>(nd4j::math::nd4j_abs<X>(inSlice[h * strideY + w * strideX])), extraParam0);
            
        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = nd4j::math::nd4j_pow<Z, Z, Z>(sum, (Z) 1.0f / extraParam0);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void pnormPooling2dCudaLauncher(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
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

    int tid = blockIdx.x * gridDim.x + threadIdx.x;

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
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
            hstart += f * dH;
        }
        if(wstart < 0){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if(hend > iH){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (hend-iH) / (Z) dH);
            hend -= f * dH;
        }
        if(wend > iW){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (wend-iW) / (Z) dW);
            wend -= f * dW;
        }
        //Accounts for dilation
        int pool_size = nd4j::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * nd4j::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

        Z max = -nd4j::DataTypeUtils::max<Z>();

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
static void maxPooling2dCudaLauncher(nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    maxPooling2dCuda<X,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}
        
//////////////////////////////////////////////////////////////////////////        
void ConvolutionUtils::pooling2d(nd4j::graph::LaunchContext& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {
 
    if(!input.isActualOnDeviceSide()) input.syncToDevice();

    switch (poolingMode) {
        
        case MAX_POOL: {
                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), maxPooling2dCudaLauncher, (block, input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
            }
            break;
        case AVG_POOL: {
                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), avgPooling2dCudaLauncher, (block, input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
            }
            break;
        case PNORM_POOL: {
                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), pnormPooling2dCudaLauncher, (block, input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
            }
            break;
        default:
            throw std::runtime_error("Pooling2D: Unknown PoolingType used");
    }

    output.tickWriteDevice();
    input.tickReadDevice();
    
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



BUILD_DOUBLE_TEMPLATE(template void maxPooling2dCudaLauncher, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void pnormPooling2dCudaLauncher, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void avgPooling2dCudaLauncher, (nd4j::graph::LaunchContext& block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void vol2colCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void *vol, const Nd4jLong *volShapeInfo, void *col, const Nd4jLong *colShapeInfo, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW), FLOAT_TYPES);


}
}