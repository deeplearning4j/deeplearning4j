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


//////////////////////////////////////////////////////////////////////////
// input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]
template <typename T>
__global__ static void im2colCuda(const void *in, void *out, 
                                  const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, 
                                  const int kH, const int kW, 
                                  const int sH, const int sW, 
                                  const int pH, const int pW, 
                                  const int dH, const int dW, 
                                  const double zeroPadValD) {

    T zeroPadVal = static_cast<T>(zeroPadValD); //Value to use when value is padding. Usually 0 but not always
    const auto im  = reinterpret_cast<const T*>(in);
          auto col = reinterpret_cast<T*>(out);    

    __shared__ Nd4jLong colLen, *colStrides, *imStrides, *colShape, *colIndices;
    __shared__ int iH, iW, colRank;
    
    if (threadIdx.x == 0) {
        
        extern __shared__ unsigned char shmem[];
        colIndices = reinterpret_cast<Nd4jLong*>(shmem);

        colRank = shape::rank(outShapeInfo);
        colLen = shape::length(outShapeInfo);        
        colShape = shape::shapeOf(const_cast<Nd4jLong*>(outShapeInfo));
        colStrides = shape::stride(outShapeInfo);
        imStrides = shape::stride(inShapeInfo);
        iH = inShapeInfo[3];
        iW = inShapeInfo[4];
    }

    __syncthreads();  
    
    const auto colInd  = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(colInd >= colLen) return;
    
    const auto indexes = colIndices + threadIdx.x * colRank;    

    shape::index2coords(colRank, colShape, colInd, colLen, indexes);

    const auto imh = (-pH + indexes[2] * dH) + indexes[4] * sH;
    const auto imw = (-pW + indexes[3] * dW) + indexes[5] * sW;
                                                                        
    const auto colBuff = col + indexes[0]*colStrides[0] + indexes[1]*colStrides[1] + indexes[2]*colStrides[2] + indexes[3]*colStrides[3] + indexes[4]*colStrides[4] + indexes[5]*colStrides[5];
    const auto imBuff  = im  + indexes[0]*imStrides[0]  + indexes[1]*imStrides[1]  + imh*imStrides[2] + imw*imStrides[3]; 

    if (static_cast<unsigned>(imh) >= static_cast<unsigned>(iH) || static_cast<unsigned>(imw) >= static_cast<unsigned>(iW))
        *colBuff = zeroPadVal;
    else 
        *colBuff = *imBuff;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>            
static void im2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, nd4j::graph::LaunchContext& context, const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, int kY, int kX, int sH, int sW, int pH, int pW, int dH, int dW, double zeroPadVal) {
    im2colCuda<T><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Nd4jLong) * 6 /* rank of out = 6 */, *context.getCudaStream()>>>(in, out, inShapeInfo, outShapeInfo, kY, kX, sH, sW, pH, pW, dH, dW, zeroPadVal);
}

//////////////////////////////////////////////////////////////////////////
void im2col(nd4j::graph::LaunchContext& context, const NDArray& in, NDArray& out, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const NDArray& arrZeroPadVal) {

    if(!in.isActualOnDeviceSide()) in.syncToDevice();
    
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (out.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    BUILD_SINGLE_SELECTOR(out.dataType(), im2colCudaLauncher, (blocksPerGrid, threadsPerBlock, context, in.getSpecialBuffer(), out.getSpecialBuffer(), in.getSpecialShapeInfo(), out.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, arrZeroPadVal.e<double>(0)), FLOAT_TYPES);

    in.tickReadDevice();
    out.tickWriteDevice();
}




BUILD_SINGLE_TEMPLATE(template void im2colCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, nd4j::graph::LaunchContext& context, const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, const int kY, const int kX, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const double zeroPadVal), FLOAT_TYPES);

}
}
}