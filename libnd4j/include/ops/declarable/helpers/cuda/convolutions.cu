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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <exceptions/cuda_exception.h>
#include <NDArrayFactory.h>
#include <MmulHelper.h>
#include <PointersManager.h>
#include <templatemath.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// vol [bS, iC, iD, iH, iW] is convoluted to col [bS, iC, kD, kH, kW, oD, oH, oW]
template <typename T>
static __global__ void vol2colCuda(const void* volume, const Nd4jLong* volShapeInfo, void* columns, const Nd4jLong* colShapeInfo,  const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const T* vol = reinterpret_cast<const T*>(volume);
          T* col = reinterpret_cast<T*>(columns);

    __shared__ int colRank, volRank;
    __shared__ Nd4jLong colLen, iD, iH, iW, *sharedMem;
    __shared__ char colOrder;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        volRank = 5;
        colRank = 8;

        colLen   = shape::length(colShapeInfo);
        colOrder = shape::order(colShapeInfo);

        iD = volShapeInfo[3];
        iH = volShapeInfo[4];
        iW = volShapeInfo[5];
    }

    __syncthreads();

    const auto colInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(colInd >= colLen)
        return;

    auto coords = sharedMem + threadIdx.x * colRank;

    shape::index2coords(colRank, colShapeInfo + 1, colInd, colLen, coords, colOrder);

    // const auto colW = coords[7];
    // const auto colH = coords[6];
    // const auto colD = coords[5];
    // const auto kCol = coords[4];
    // const auto kRow = coords[3];
    // const auto kDep = coords[2];
    // const auto c    = coords[1];
    // const auto b    = coords[0];

    const auto colOffset = shape::getOffset(0, colShapeInfo + 1, colShapeInfo + colRank + 1, coords, colRank);

    coords[2] = -pD + coords[2] * dD + coords[5] * sD;     // const auto volDep = (-pD + kDep * dD) + colD * sD;
    coords[3] = -pH + coords[3] * dH + coords[6] * sH;     // const auto volRow = (-pH + kRow * dH) + colH * sH;
    coords[4] = -pW + coords[4] * dW + coords[7] * sW;     // const auto volCol = (-pW + kCol * dW) + colW * sW;

    if (static_cast<unsigned>(coords[2]) >= static_cast<unsigned>(iD) || static_cast<unsigned>(coords[3]) >= static_cast<unsigned>(iH) || static_cast<unsigned>(coords[4]) >= static_cast<unsigned>(iW))
        col[colOffset] = static_cast<T>(0.);
    else
        col[colOffset] = vol[shape::getOffset(0, volShapeInfo + 1, volShapeInfo + volRank + 1, coords, volRank)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void vol2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* volume, const Nd4jLong* volShapeInfo,
                                      void* columns, const Nd4jLong* colShapeInfo,
                                const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    vol2colCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(volume, volShapeInfo, columns, colShapeInfo,  sD, sH, sW, pD, pH, pW, dD, dH, dW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::vol2col(nd4j::graph::Context& block, const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    PointersManager manager(block.launchContext(), "vol2col");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (col.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = col.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&col}, {&vol});
    BUILD_SINGLE_SELECTOR(vol.dataType(), vol2colCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), vol.getSpecialBuffer(), vol.getSpecialShapeInfo(), col.specialBuffer(), col.specialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&col}, {&vol});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
// columns [bS, iC, kD, kH, kW, oD, oH, oW] to be de-convoluted to volume [bS, iC, iD, iH, iW]
template <typename T>
static __global__ void col2volCuda(const void* columns, const Nd4jLong* colShapeInfo, void* volume, const Nd4jLong* volShapeInfo,  const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const T* col = reinterpret_cast<const T*>(columns);
          T* vol = reinterpret_cast<T*>(volume);

    __shared__ int colRank, volRank, kDeff, kHeff, kWeff, oD, oH, oW;
    __shared__ Nd4jLong *sharedMem, volLen;
    __shared__ char volOrder;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        oD = colShapeInfo[6];
        oH = colShapeInfo[7];
        oW = colShapeInfo[8];

        kDeff = colShapeInfo[3] + (colShapeInfo[3] - 1) * (dD - 1);
        kHeff = colShapeInfo[4] + (colShapeInfo[4] - 1) * (dH - 1);
        kWeff = colShapeInfo[5] + (colShapeInfo[5] - 1) * (dW - 1);

        volRank = 5;
        colRank = 8;

        volLen   = shape::length(volShapeInfo);
        volOrder = shape::order(volShapeInfo);
    }

    __syncthreads();

    const auto volInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(volInd >= volLen)
        return;

    auto coords = sharedMem + threadIdx.x * colRank;

    shape::index2coords(volRank, volShapeInfo + 1, volInd, volLen, coords, volOrder);

    const auto volOffset = shape::getOffset(0, volShapeInfo + 1, volShapeInfo + volRank + 1, coords, volRank);

    const int imD = coords[2] + pD;
    const int imH = coords[3] + pH;
    const int imW = coords[4] + pW;

    const int colDstart = (imD < kDeff) ? 0 : (imD - kDeff) / sD + 1;
    const int colHstart = (imH < kHeff) ? 0 : (imH - kHeff) / sH + 1;
    const int colWstart = (imW < kWeff) ? 0 : (imW - kWeff) / sW + 1;

    const int colDend = nd4j::math::nd4j_min<uint>(imD / sD + 1, oD);
    const int colHend = nd4j::math::nd4j_min<uint>(imH / sH + 1, oH);
    const int colWend = nd4j::math::nd4j_min<uint>(imW / sW + 1, oW);

    T val = 0;

    for(coords[5] = colDstart; coords[5] < colDend; ++coords[5]) {
        coords[2] = imD - coords[5] * sD;

        for(coords[6] = colHstart; coords[6] < colHend; ++coords[6]) {
            coords[3] = imH - coords[6] * sH;

            for(coords[7] = colWstart; coords[7] < colWend; ++coords[7]) {
                coords[4] = imW - coords[7] * sW;

                if(coords[2] % dD == 0 && coords[3] % dH == 0 && coords[4] % dW == 0) {
                    coords[2] /= dD;
                    coords[3] /= dH;
                    coords[4] /= dW;

                    val += col[shape::getOffset(0, colShapeInfo + 1, colShapeInfo + colRank + 1, coords, colRank)];
                }
            }
        }
    }

    vol[volOffset] = val;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void col2volCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* columns, const Nd4jLong* colShapeInfo,
                                      void* volume, const Nd4jLong* volShapeInfo,
                                const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    col2volCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(columns, colShapeInfo, volume, volShapeInfo, sD, sH, sW, pD, pH, pW, dD, dH, dW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::col2vol(nd4j::graph::Context& block, const NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    PointersManager manager(block.launchContext(), "col2vol");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (vol.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = col.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&vol}, {&col});
    BUILD_SINGLE_SELECTOR(vol.dataType(), col2volCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), col.getSpecialBuffer(), col.getSpecialShapeInfo(), vol.specialBuffer(), vol.specialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&vol}, {&col});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2d_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] always
    // bias    [oC]
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    // kH  filter(kernel) height
    // kW  filter(kernel) width
    // sH  strides height
    // sW  strides width
    // pH  paddings height
    // pW  paddings width
    // dH  dilations height
    // dW  dilations width
    // isSameMode 0-VALID, 1-SAME
    // isNCHW     1-NCHW,  0-NHWC

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    std::vector<int> permutForOutput;

    if(isNCHW)
        permutForOutput = {0, 3, 1, 2};                                             // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    else
        input = new NDArray(input->permute({0, 3, 1, 2}));                         // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC

    NDArray col('c', {bS, oH, oW, kH, kW, iC}, input->dataType(), input->getContext());
    NDArray colP = col.permute({0, 5, 3, 4, 1, 2});            // {bS, iC, kH, kW, oH, oW}
    NDArray mmulResult('f', {bS*oH*oW, oC}, output->dataType(), output->getContext());

    //----- calculation of output -----//
    auto ctx = block.launchContext();
    helpers::im2col(*ctx, *input, colP, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    MmulHelper::tensorDot(&col, weights, &mmulResult, {3,4,5}, {0,1,2}, {}); // [bS, oH, oW, kH, kW, iC] x [kH, kW, iC, oC] = [bS, oH, oW, oC]

    //----- assign outTemp to output  -----//
    if(isNCHW) {
        mmulResult.reshapei({bS, oH, oW, oC});
        mmulResult.permutei(permutForOutput);
    }
    output->assign(mmulResult);

    //----- add biases if required -----//
    if(bias)
        output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
        // helpers::addBias(*output, *bias, isNCHW);

    if(!isNCHW)
        delete input;

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::conv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), conv2d_, (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2d_(const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input     [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights   [kH, kW, iC, mC] always
    // bias      [oC] = iC*mC
    // output    [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

    // kH           filter(kernel) height
    // kW           filter(kernel) width
    // sH           strides height
    // sW           strides width
    // pH           paddings height
    // pW           paddings width
    // dH           dilations height
    // dW           dilations width
    // isSameMode   0-VALID, 1-SAME
    // isNCHW       0-NCHW,  1-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
    std::vector<std::vector<Nd4jLong>> modifOutput;
    std::vector<Nd4jLong> outReShape;

    if(!isNCHW) {
        outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        input = new NDArray(input->permute({0, 3, 1, 2}));                             // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
    }
    else {
        outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
    NDArray outputReshaped = output->reshape(output->ordering(), outReShape);

    helpers::im2col(*output->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    MmulHelper::tensorDot(&columns, weights, &outputReshaped, modifColumns, {{2,0,1,3},{iC,kH*kW,mC}}, modifOutput);              // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]

    if(bias)
        output->applyBroadcast(broadcast::Add, {indIOioC}, bias);

    if(!isNCHW)
        delete input;
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::depthwiseConv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), depthwiseConv2d_, (input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void sconv2d_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input         [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    // weightsDepth  [kH, kW, iC, mC]  always
    // weightsPoint  [1, 1, iC*mC, oC] always
    // bias          [oC], oC = iC*mC if weightsPoint=nullptr
    // output is     [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)

    //  kH         filter(kernel) height
    //  kW         filter(kernel) width
    //  sH         strides height
    //  sW         strides width
    //  pH         paddings height
    //  pW         paddings width
    //  dH         dilations height
    //  dW         dilations width
    //  isSameMode 0-VALID, 1-SAME
    //  isNCHW     1-NCHW,  0-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    NDArray* outputDepth = output;
    if(weightsPoint)                        // if pointwise convolution is expected
        outputDepth = new NDArray(output->ordering(), !isNCHW ? std::vector<Nd4jLong>({bS, oH, oW, iC*mC}) : std::vector<Nd4jLong>({bS, iC*mC, oH, oW}), input->dataType(), input->getContext());

    // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //
    ConvolutionUtils::depthwiseConv2d(block, input, weightsDepth, weightsPoint ? nullptr : bias, outputDepth, kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, isNCHW);

    // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
    if (weightsPoint) {
        ConvolutionUtils::conv2d(block, outputDepth, weightsPoint, bias, output, 1,1, 1,1, 0,0, 1,1, isSameMode, isNCHW);             // in this case oH=iH, oW=iW
        delete outputDepth;
    }
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::sconv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), sconv2d_, (block, input, weightsDepth, weightsPoint, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), FLOAT_TYPES);
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
static void avgPooling2dCudaLauncher(nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
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
static void pnormPooling2dCudaLauncher(nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
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
static void maxPooling2dCudaLauncher(nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    maxPooling2dCuda<X,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling2d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {

    if(!input.isActualOnDeviceSide()) input.syncToDevice();

    switch (poolingMode) {

        case MAX_POOL: {
                BUILD_SINGLE_SELECTOR_TWICE(input.dataType(), maxPooling2dCudaLauncher, (*block.launchContext(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), FLOAT_TYPES);
            }
            break;
        case AVG_POOL: {
                BUILD_SINGLE_SELECTOR_TWICE(input.dataType(), avgPooling2dCudaLauncher, (*block.launchContext(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), FLOAT_TYPES);
            }
            break;
        case PNORM_POOL: {
                BUILD_SINGLE_SELECTOR_TWICE(input.dataType(), pnormPooling2dCudaLauncher, (*block.launchContext(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), FLOAT_TYPES);
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

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void pooling3dCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // x input  is [bS, iC, iD, iH, iW]
    // z output is [bS, iC, oD, oH, oW]

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, kDeff, kHeff, kWeff, iD, iH, iW, kProd;
    __shared__ Nd4jLong *sharedMem, zLen;
    __shared__ char zOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        zLen   = shape::length(zShapeInfo);
        zOrder = shape::order(zShapeInfo);
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

    shape::index2coords(rank, zShapeInfo + 1, zInd, zLen, coords, zOrder);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

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
                        T val = x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];
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
                        sum += x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];

            if (extraParam0 == 0) {         //Exclude padding
                uint a = (dend - dstart) / dD + ((dend - dstart) % dD == 0 ? 0 : 1);
                uint b = (hend - hstart) / dH + ((hend - hstart) % dH == 0 ? 0 : 1);
                uint c = (wend - wstart) / dW + ((wend - wstart) % dW == 0 ? 0 : 1);
                sum /=  static_cast<T>(a * b * c);                                       //  /= nd4j::math::nd4j_ceil<double,T>(static_cast<double>(dend - dstart) / static_cast<double>(dD)) * nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend - hstart) / static_cast<double>(dH)) * nd4j::math::nd4j_ceil<double,T>(static_cast<double>(wend - wstart) / static_cast<double>(dW));   //Accounts for dilation
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
                        sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)]), extraParam0);

            sum = nd4j::math::nd4j_pow<T,T,T>(sum, (T) 1.f / extraParam0);

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
void ConvolutionUtils::pooling3d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    PointersManager manager(block.launchContext(), "pooling3d");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = output.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling3dCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}

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
    __shared__ Nd4jLong *sharedMem, yLen;
    __shared__ char yOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        yLen   = shape::length(yShapeInfo);
        yOrder = shape::order(yShapeInfo);
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

    shape::index2coords(rank, yShapeInfo + 1, yInd, yLen, coords, yOrder);

    const auto yOffset = shape::getOffset(0, yShapeInfo + 1, yShapeInfo + rank + 1, coords, rank);

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

            T max = -DataTypeUtils::max<T>();
            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH) {
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW){
                    T val = x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];
                    if (val > max) {
                        max = val;
                        coord2 = coords[2];
                        coord3 = coords[3];
                    }
                }
            }
            coords[2] = coord2;
            coords[3] = coord3;
            nd4j::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank)], y[yOffset]);

        }
        break;

        /*** avg ***/
        case 1: {

            T val = y[yOffset];

            if (extraParam0 == 0)         //Exclude padding
                val /= nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend - hstart) / static_cast<double>(dH)) * nd4j::math::nd4j_ceil<double,T>(static_cast<double>(wend - wstart) / static_cast<double>(dW));   //Accounts for dilation
            else if (extraParam0 == 1)    //Include padding
                val /= kProd;

            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH)
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW)
                    nd4j::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank)], val);
        }
        break;

        /*** pnorm ***/
        case 2: {

            T sum = static_cast<T>(0.);
            T val = y[yOffset];

            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH)
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW)
                    sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)]), extraParam0);

            val *= nd4j::math::nd4j_pow<T,T,T>(sum, ((T)1.f - extraParam0) / extraParam0);

            for (coords[2] = hstart; coords[2] < hend; coords[2] += dH)
                for (coords[3] = wstart; coords[3] < wend; coords[3] += dW)
                    nd4j::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank)], val * nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)]), extraParam0 - 1.f));
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
void ConvolutionUtils::pooling2dBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // initial zeroing of gradI
    gradI.nullify();

    PointersManager manager(block.launchContext(), "pooling2dBP");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (gradO.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = gradO.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&gradI}, {&input, &gradO});
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling2dBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&input, &gradO});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void pooling3dBPCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // x: input [bS, iC, iD, iH, iW]
    // y: gradO [bS, iC, oD, oH, oW]
    // z: gradI [bS, iC, iD, iH, iW] -> gradI is output in this function


    const T* x = reinterpret_cast<const T*>(vx);
    const T* y = reinterpret_cast<const T*>(vy);
          T* z = reinterpret_cast<T*>(vz);

    Nd4jLong coord2, coord3, coord4;
    __shared__ int rank, kDeff, kHeff, kWeff, iD, iH, iW, kProd;
    __shared__ Nd4jLong *sharedMem, yLen;
    __shared__ char yOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        yLen   = shape::length(yShapeInfo);
        yOrder = shape::order(yShapeInfo);
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

    const auto yInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(yInd >= yLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(rank, yShapeInfo + 1, yInd, yLen, coords, yOrder);

    const auto yOffset = shape::getOffset(0, yShapeInfo + 1, yShapeInfo + rank + 1, coords, rank);

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
                        T val = x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];
                        if (val > max) {
                            max = val;
                            coord2 = coords[2];
                            coord3 = coords[3];
                            coord4 = coords[4];
                        }
                    }
                }
            }
            coords[2] = coord2;
            coords[3] = coord3;
            coords[4] = coord4;
            nd4j::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank)], y[yOffset]);
        }
        break;

        /*** avg ***/
        case 1: {

            T val = y[yOffset];

            if (extraParam0 == 0)         //Exclude padding
                val /= nd4j::math::nd4j_ceil<double,T>(static_cast<double>(dend - dstart) / static_cast<double>(dD))  * nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend - hstart) / static_cast<double>(dH))     * nd4j::math::nd4j_ceil<double,T>(static_cast<double>(wend - wstart)    / static_cast<double>(dW));   //Accounts for dilation
            else if (extraParam0 == 1)    //Include padding
                val /= kProd;

            for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
                for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
                    for (coords[4] = wstart; coords[4] < wend; coords[4] += dW)
                        nd4j::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank)], val);
        }
        break;

        /*** pnorm ***/
        case 2: {

            T sum = static_cast<T>(0.);
            T val = y[yOffset];

            for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
                for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
                    for (coords[4] = wstart; coords[4] < wend; coords[4] += dW)
                        sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)]), extraParam0);

            val *= nd4j::math::nd4j_pow<T,T,T>(sum, ((T)1.f - extraParam0) / extraParam0);

            for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
                for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
                    for (coords[4] = wstart; coords[4] < wend; coords[4] += dW)
                        nd4j::math::atomics::nd4j_atomicAdd<T>(&z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank)], val * nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)]), extraParam0 - 1.f));
        }
        break;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling3dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                    const void* vx, const Nd4jLong* xShapeInfo,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo,
                                    const int kD, const int kH, const int kW,
                                    const int sD, const int sH, const int sW,
                                    const int pD, const int pH, const int pW,
                                    const int dD, const int dH, const int dW,
                                    const int poolingMode, const int extraParam0) {

    pooling3dBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling3dBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

    // initial zeroing of gradI
    gradI.nullify();

    PointersManager manager(block.launchContext(), "pooling3dBP");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (gradO.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = gradO.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&gradI}, {&input, &gradO});
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling3dBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&input, &gradO});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2dBP_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] always
    // bias    [oC]
    // gradO   [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

    // gradI    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    // gradW    [kH, kW, iC, oC] always
    // gradB    [oC]

    // kH         filter(kernel) height
    // kW         filter(kernel) width
    // sH         strides height
    // sW         strides width
    // pH         paddings height
    // pW         paddings width
    // dH         dilations height
    // dW         dilations width
    // isSameMode 0-VALID, 1-SAME
    // isNCHW     0-NHWC, 1-NCHW

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    std::vector<int> gradOaxesForDot;

    if(!isNCHW) {
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW
        input = new NDArray(input->permute({0, 3, 1, 2}));                      // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
        gradI = new NDArray(gradI->permute({0, 3, 1, 2}));                      // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    } else {
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
    }

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());

    // ----- calculation of gradW ----- //
    if(gradW) {
        auto ctx = block.launchContext();
        helpers::im2col(*ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));   // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
        nd4j::MmulHelper::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, {2, 0, 1, 3});       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]
    }

    // ----- calculation of gradB ----- //
    if(gradB) {
        NDArray* gradBR = gradB;
        if(gradB->rankOf() == 2)
            gradBR = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}));
        gradO->reduceAlongDimension(reduce::Sum, gradBR, gradOaxesForDot);                          // sum over bS, oH, oW
        if(gradBR != gradB)
            delete gradBR;
    }

    //----- calculation of gradI -----//
    nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, {2, 3, 1, 0, 4, 5});  // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]

    helpers::col2im(*block.launchContext(), columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                          // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::conv2dBP(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), conv2dBP_, (block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2dBP_(const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    // weights  [kH, kW, iC, mC] always
    // bias     [oC] = [iC*mC]
    // gradO    [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    // gradI    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    // gradW    [kH, kW, iC, mC] always
    // gradB    [oC]

    //  kH          filter(kernel) height
    //  kW          filter(kernel) width
    //  sH          strides height
    //  sW          strides width
    //  pH          paddings height
    //  pW          paddings width
    //  dH          dilations height
    //  dW          dilations width
    //  isSameMode  0-VALID, 1-SAME
    //  isNCHW      0-NHWC, 1-NCHW

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,2,3,0,4,5}, {iC, kH*kW, bS*oH*oW}};      // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
    std::vector<std::vector<Nd4jLong>> modifGradO1, modifGradO2;
    std::vector<Nd4jLong> gradOreShape;

    if(!isNCHW) {
        gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        input = new NDArray(input->permute({0, 3, 1, 2}));                             // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
        gradI = new NDArray(gradI->permute({0, 3, 1, 2}));                             // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
    }
    else {
        gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
    NDArray gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);

    // ----- calculation of gradW and gradB ----- //

    helpers::im2col(*input->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    nd4j::MmulHelper::tensorDot(&columns, &gradOreshaped, gradW, modifColumns, modifGradO1, {{2,0,1,3},{iC,kH*kW,mC}});  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    // ----- calculation of gradB ----- //
    if(gradB) {
        NDArray* gradBR = gradB;
        if(gradB->rankOf() == 2)
            gradBR = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}));
        gradO->reduceAlongDimension(reduce::Sum, gradBR, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW
        if(gradBR != gradB)
            delete gradBR;
    }

    //----- calculation of gradI -----//
    nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {{2,0,1,3},{iC,kH*kW,mC}}, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]
    helpers::col2im(*input->getContext(), columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                                       // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::depthwiseConv2dBP(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), depthwiseConv2dBP_, (input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), FLOAT_TYPES);
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void upsampling2dCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int factorH, const int factorW, const bool isNCHW) {

    // x has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)
    // z has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, dimIH;
    __shared__ Nd4jLong *sharedMem, zLen;
    __shared__ char zOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        dimIH  = isNCHW ? 2 : 1;
        zLen   = shape::length(zShapeInfo);
        zOrder = shape::order(zShapeInfo);
        rank   = 4;
    }

    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(rank, zShapeInfo + 1, zInd, zLen, coords, zOrder);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

    coords[dimIH]     /= factorH;
    coords[dimIH + 1] /= factorW;

    const auto xOffset = shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank);

    z[zOffset] = x[xOffset];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                     const void* vx, const Nd4jLong* xShapeInfo,
                                           void* vz, const Nd4jLong* zShapeInfo,
                                     const int factorH, const int factorW, const bool isNCHW) {

    upsampling2dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, factorH, factorW, isNCHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling2d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {

    PointersManager manager(block.launchContext(), "upsampling2d");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = output.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), upsampling2dCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), factorH, factorW, isNCHW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void upsampling3dCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

    // x has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
    // z has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, dimID;
    __shared__ Nd4jLong *sharedMem, zLen;
    __shared__ char zOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        dimID  = isNCDHW ? 2 : 1;
        zLen   = shape::length(zShapeInfo);
        zOrder = shape::order(zShapeInfo);
        rank   = 5;
    }

    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(rank, zShapeInfo + 1, zInd, zLen, coords, zOrder);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

    coords[dimID]     /= factorD;
    coords[dimID + 1] /= factorH;
    coords[dimID + 2] /= factorW;

    const auto xOffset = shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank);

    z[zOffset] = x[xOffset];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                     const void* vx, const Nd4jLong* xShapeInfo,
                                           void* vz, const Nd4jLong* zShapeInfo,
                                     const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

    upsampling3dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, factorD, factorH, factorW, isNCDHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling3d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

    PointersManager manager(block.launchContext(), "upsampling3d");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = output.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), upsampling3dCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), factorD, factorH, factorW, isNCDHW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void upsampling2dBPCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const bool isNCHW) {

    // x (gradO) has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    // z (gradI) has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, dimIH;
    __shared__ uint factorH, factorW;
    __shared__ Nd4jLong *sharedMem, zLen;
    __shared__ char zOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        dimIH  = isNCHW ? 2 : 1;
        zLen   = shape::length(zShapeInfo);
        zOrder = shape::order(zShapeInfo);
        rank   = 4;

        factorH = xShapeInfo[dimIH + 1] / zShapeInfo[dimIH + 1];
        factorW = xShapeInfo[dimIH + 2] / zShapeInfo[dimIH + 2];
    }

    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(rank, zShapeInfo + 1, zInd, zLen, coords, zOrder);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

    const Nd4jLong zCoord2 = coords[dimIH];
    const Nd4jLong zCoord3 = coords[dimIH + 1];

    for(coords[dimIH] = zCoord2; coords[dimIH] < zCoord2 + factorH; ++coords[dimIH])
        for(coords[dimIH + 1] = zCoord3; coords[dimIH + 1] < zCoord3 + factorW; ++coords[dimIH + 1])
            z[zOffset] += x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                       const void* vx, const Nd4jLong* xShapeInfo,
                                             void* vz, const Nd4jLong* zShapeInfo,
                                       const bool isNCHW) {

    upsampling2dBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, isNCHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling2dBP(nd4j::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {

    PointersManager manager(block.launchContext(), "upsampling2d_bp");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (gradI.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = gradI.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&gradI}, {&gradO});
    BUILD_SINGLE_SELECTOR(gradI.dataType(), upsampling2dBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), isNCHW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&gradO});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void upsampling3dBPCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const bool isNCDHW) {

    // x (gradO) has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
    // z (gradI) has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, dimID;
    __shared__ uint factorD, factorH, factorW;
    __shared__ Nd4jLong *sharedMem, zLen;
    __shared__ char zOrder;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        dimID  = isNCDHW ? 2 : 1;
        zLen   = shape::length(zShapeInfo);
        zOrder = shape::order(zShapeInfo);
        rank   = 5;

        factorD = xShapeInfo[dimID + 1] / zShapeInfo[dimID + 1];
        factorH = xShapeInfo[dimID + 2] / zShapeInfo[dimID + 2];
        factorW = xShapeInfo[dimID + 3] / zShapeInfo[dimID + 3];
    }

    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(rank, zShapeInfo + 1, zInd, zLen, coords, zOrder);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

    const Nd4jLong zCoord2 = coords[dimID];
    const Nd4jLong zCoord3 = coords[dimID + 1];
    const Nd4jLong zCoord4 = coords[dimID + 2];

    for(coords[dimID] = zCoord2; coords[dimID] < zCoord2 + factorD; ++coords[dimID])
        for(coords[dimID + 1] = zCoord3; coords[dimID + 1] < zCoord3 + factorH; ++coords[dimID + 1])
            for(coords[dimID + 2] = zCoord4; coords[dimID + 2] < zCoord4 + factorW; ++coords[dimID + 2])
                z[zOffset] += x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                       const void* vx, const Nd4jLong* xShapeInfo,
                                             void* vz, const Nd4jLong* zShapeInfo,
                                       const bool isNCDHW) {

    upsampling3dBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, isNCDHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling3dBP(nd4j::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {

    PointersManager manager(block.launchContext(), "upsampling3d_bp");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (gradI.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = gradI.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&gradI}, {&gradO});
    BUILD_SINGLE_SELECTOR(gradI.dataType(), upsampling3dBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), isNCDHW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&gradO});

    manager.synchronize();
}









}
}