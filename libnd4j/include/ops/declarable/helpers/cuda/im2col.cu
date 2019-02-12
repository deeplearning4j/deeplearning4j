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
__global__ static void im2colCuda(const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const double zeroPadValD) {
        
    T zeroPadVal = static_cast<T>(zeroPadValD); //Value to use when value is padding. Usually 0 but not always
    const auto x = reinterpret_cast<const T*>(in);
          auto z = reinterpret_cast<T*>(out);

    __shared__ Nd4jLong *outShape, *inShape, *outStride, *inStride;
    __shared__ int kSize, iC, iH, iW, oH, oW, n, stride0, stride1, stride2, stride3;

    if (threadIdx.x == 0) {
        
        outShape  = shape::shapeOf(const_cast<Nd4jLong*>(outShapeInfo));
        inShape   = shape::shapeOf(const_cast<Nd4jLong*>(inShapeInfo));
        outStride = shape::stride(const_cast<Nd4jLong*>(outShapeInfo));
        inStride  = shape::stride(const_cast<Nd4jLong*>(inShapeInfo));

        iC = inShape[1];
        iH = inShape[2];
        iW = inShape[3];

        oH = outShape[4];
        oW = outShape[5];

        stride0 = inStride[0];
        stride1 = inStride[1];
        stride2 = inStride[2];
        stride3 = inStride[3];

        n = inShape[0] * iC * oH * oW;
        kSize = kW * kH;
    }
    
    __syncthreads();
    

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += blockDim.x*gridDim.x) {
    
        const int ind = i / oW;
        const int outH = ind % oH;
        const int outW = i % oW;

        const int inC = ind / oH;
        const int outC = inC * kSize;

        const int inDepth = inC % iC;
        const int inNum = inC / iC;
        const int hOffset = outH * sH - pH;
        const int wOffset = outW * sW - pW;
                    
        int ic = (outC * oH + outH) * oW + outW;
        
        const auto pX = x + inNum * stride0 + inDepth * stride1 + hOffset * stride2 + wOffset*stride3;
              auto pZ = z + (outC * oH + outH) * oW + outW;

        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                
                const int inH = hOffset + i * dH;
                const int inW = wOffset + j * dW;                
                int icTemp = ic;
                int outInd = 0;
                
                for (int dim = 5; dim >= 0; dim--) {
                    outInd += (icTemp % outShape[dim])  * outStride[dim];
                    icTemp = icTemp / outShape[dim];
                }
                
                if (inH >= 0 && inW >= 0 && inH < iH && inW < iW)
                    z[outInd] = pX[i * dH * stride2 + j * dW * stride3];
                else 
                    z[outInd] = zeroPadVal;

                pZ += oH * oW;
                ic += oH * oW;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>            
static void im2colCudaLauncher(nd4j::graph::LaunchContext& context, const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, int kY, int kX, int sH, int sW, int pH, int pW, int dH, int dW, double zeroPadVal) {
       im2colCuda<T><<<512, 512, 1024, *context.getCudaStream()>>>(in, out, inShapeInfo, outShapeInfo, kY, kX, sH, sW, pH, pW, dH, dW, zeroPadVal);
}

//////////////////////////////////////////////////////////////////////////
void im2col(nd4j::graph::LaunchContext& context, const NDArray& in, NDArray& out, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const NDArray& arrZeroPadVal) {
    BUILD_SINGLE_SELECTOR(out.dataType(), im2colCudaLauncher, (context, in.getSpecialBuffer(), out.getSpecialBuffer(), in.getSpecialShapeInfo(), out.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, arrZeroPadVal.e<double>(0)), FLOAT_TYPES);
}




BUILD_SINGLE_TEMPLATE(template void im2colCudaLauncher, (nd4j::graph::LaunchContext& context, const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *xShape, const int kY, const int kX, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const double zeroPadVal), FLOAT_TYPES);

}
}
}