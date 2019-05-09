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

#include <ops/declarable/helpers/col2im.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

//////////////////////////////////////////////////////////////////////////
// [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
template<typename T>
__global__ static void col2imCuda(const void *in, void *out, const Nd4jLong *inShapeInfo, const Nd4jLong *outShapeInfo, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {

    const auto x = reinterpret_cast<const T*>(in);
          auto z = reinterpret_cast<T*>(out);

    __shared__ Nd4jLong *outShape, *inShape, *outStride, *inStride;
    __shared__ int n, kHeff, kWeff, iC, oH, oW, kH, kW, stride0, stride1, stride2, stride3, stride4, stride5;

    if (threadIdx.x == 0) {
        
        outShape  = shape::shapeOf(const_cast<Nd4jLong*>(outShapeInfo));
        inShape   = shape::shapeOf(const_cast<Nd4jLong*>(inShapeInfo));
        outStride = shape::stride(const_cast<Nd4jLong*>(outShapeInfo));
        inStride  = shape::stride(const_cast<Nd4jLong*>(inShapeInfo));

        iC = outShape[1];        
        
        kH = inShape[2];
        kW = inShape[3];
        oH = inShape[4];    //(iH + 2 * pH - kH) / sW + 1;
        oW = inShape[5];    //(iW + 2 * pW - kW) / sH + 1;

        stride0 = inStride[0];
        stride1 = inStride[1];
        stride2 = inStride[2];
        stride3 = inStride[3];
        stride4 = inStride[4];
        stride5 = inStride[5];

        n = outShape[0] * iC * oH * oW;        
         
        //Effective kernel size, accounting for dilation
        kHeff = kH + (kH - 1) * (dH - 1);
        kWeff = kW + (kW - 1) * (dW - 1);
        
    }

    __syncthreads();

    for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        
        T val = 0.f;
        const int outW = i % iW + pW;
        const int outH = (i / iW) % iH + pH;
        const int outC = i / (iW * iH);
        const int outNum = outC / iC;
        const int outDepth = outC % iC;

        // compute the start and end of the output
        // These are the indexes for dimensions ??? in the 6d col matrix
        const int inHstart = (outH < kHeff) ? 0 : (outH - kHeff) / sH + 1;
        const int inHend = nd4j::math::nd4j_min<int>(outH / sH + 1, oH);
        const int inWstart = (outW < kWeff) ? 0 : (outW - kWeff) / sW + 1;
        const int inWend = nd4j::math::nd4j_min<int>(outW / sW + 1, oW);

        //Iterate over col entries in the 6d array... these are added up
        for (int inH = inHstart; inH < inHend; inH += 1) {
            for (int inW = inWstart; inW < inWend; inW += 1) {
                
                int hK = (outH - inH * sH);
                int wK = (outW - inW * sW);

                if(hK % dH == 0 && wK % dW == 0) {
                    hK /= dH;
                    wK /= dW;

                    int inInd = outNum * stride0 + outDepth * stride1 + hK * stride2 + wK * stride3 + inH * stride4 + inW * stride5;
                    val += x[inInd];
                }
            }
        }
        int outInd = 0;
        int ic = i;
        
        for (int dim = 3; dim >= 0; dim--) {
            outInd += (ic % outShape[dim])  * outStride[dim];
            ic = ic / outShape[dim];
        }
        z[outInd] = val;
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void col2imCudaLauncher(nd4j::LaunchContext  &context, const void *x, void *z, const Nd4jLong *xShapeInfo, const Nd4jLong *zShapeInfo, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {
    col2imCuda<T><<<512, 512, 1024, *context.getCudaStream()>>>(x, z, xShapeInfo, zShapeInfo, sH, sW, pH, pW, iH, iW, dH, dW);
}

//////////////////////////////////////////////////////////////////////////
void col2im(nd4j::LaunchContext & context, const NDArray& input, NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {
    
    if(!input.isActualOnDeviceSide()) input.syncToDevice();

    BUILD_SINGLE_SELECTOR(output.dataType(), col2imCudaLauncher, (context, input.getSpecialBuffer(), output.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialShapeInfo(), sH, sW, pH, pW, iH, iW, dH, dW), FLOAT_TYPES);

    input.tickReadDevice();
    output.tickWriteDevice();
}



BUILD_SINGLE_TEMPLATE(template void col2imCudaLauncher, (nd4j::LaunchContext  &context, const void *x, void *z, const Nd4jLong *xShapeInfo, const Nd4jLong *zShapeInfo, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW), FLOAT_TYPES);

}
}
}