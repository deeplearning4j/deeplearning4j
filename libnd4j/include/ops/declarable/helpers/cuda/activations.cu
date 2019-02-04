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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/activations.h>
#include <ShapeUtils.h>
#include <numeric>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ static void preluCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                 const void *vy, const Nd4jLong *yShapeInfo,
                                       void *vz) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<X*>(vz);    

    __shared__ Nd4jLong  len;
    
    if (threadIdx.x == 0)         
        len = shape::length(xShapeInfo);    

    __syncthreads();    

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalThreads = gridDim.x * blockDim.x;

    for (int i = tid; i < len; i += totalThreads) {
            
        const auto xzOffset = shape::getIndexOffset(i, xShapeInfo, len);
        const auto xVal     = x[xzOffset];

        if(xVal < 0)                
            z[xzOffset] = xVal * y[shape::subArrayOffset(i, xShapeInfo, yShapeInfo)];
        else
            z[xzOffset] = xVal;
    }    
}

template<typename X, typename Y>
__host__ static void preluCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz) {

    preluCuda<X, Y><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz);
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
__global__ static void preluBPCuda(const void *vIn,    const Nd4jLong *inShapeInfo,
                                   const void *vAlpha, const Nd4jLong *alphaShapeInfo,
                                   const void *vdLdO,  const Nd4jLong *dLdOShapeInfo,
                                         void *vdLdI,  const Nd4jLong *dLdIShapeInfo,
                                         void *vdLdA,  const Nd4jLong *dLdAShapeInfo) {

    const auto in    = reinterpret_cast<const X*>(vIn);
    const auto alpha = reinterpret_cast<const Y*>(vAlpha);
    const auto dLdO  = reinterpret_cast<const Z*>(vdLdO);
          auto dLdI  = reinterpret_cast<Z*>(vdLdI);
          auto dLdA  = reinterpret_cast<Z*>(vdLdA);

    __shared__ Nd4jLong alphaLen;    
    
    if (threadIdx.x == 0)         
        alphaLen = shape::length(alphaShapeInfo);        

    __syncthreads();    

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= alphaLen) return;    

    Nd4jLong inputIdxs[MAX_RANK*2];
    int numIdxs = shape::outerArrayOffsets(inputIdxs, i, inShapeInfo, alphaShapeInfo);
    Nd4jLong dLdOIdxs[MAX_RANK*2];
    shape::outerArrayOffsets(dLdOIdxs, i, dLdOShapeInfo, alphaShapeInfo);
    Nd4jLong dLdIIdxs[MAX_RANK*2];
    shape::outerArrayOffsets(dLdIIdxs, i, dLdIShapeInfo, alphaShapeInfo);
        
    const auto alphaOffset = shape::getIndexOffset(i, alphaShapeInfo, alphaLen);
    const auto dLdAOffset  = shape::getIndexOffset(i, dLdAShapeInfo, alphaLen);
        
    for(Nd4jLong j = 0; j < numIdxs; ++j) {
                
        const auto inInd   = inputIdxs[j];
        const auto dLdOInd = dLdOIdxs[j];
        const auto dLdIInd = dLdIIdxs[j];

        if(in[inInd] < 0) {                    
            dLdI[dLdIInd] = dLdO[dLdOInd] * alpha[alphaOffset];
            auto prevVal = dLdA[dLdAOffset];
            prevVal = prevVal + dLdO[dLdOInd] * in[inInd];
            dLdA[dLdAOffset] = prevVal;
        }
        else
            dLdI[dLdIInd] = dLdO[dLdOInd];
    }            
}


template<typename X, typename Y, typename Z>
__host__ static void preluBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vIn, const Nd4jLong *inShapeInfo, const void *vAlpha, const Nd4jLong *alphaShapeInfo, const void *vdLdO,  const Nd4jLong *dLdOShapeInfo, void *vdLdI,  const Nd4jLong *dLdIShapeInfo, void *vdLdA,  const Nd4jLong *dLdAShapeInfo) {

    preluBPCuda<X, Y, Z><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vIn, inShapeInfo, vAlpha, alphaShapeInfo, vdLdO, dLdOShapeInfo, vdLdI, dLdIShapeInfo, vdLdA, dLdAShapeInfo);
}

    template <typename T>
    void _softMaxForVector(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {

    }

    template <typename T>
    void _logSoftMaxForVector(void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo) {

    }

    ///////////////////////////////////////////////////////////////////
    void softMaxForVector(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _softMaxForVector, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }


    ///////////////////////////////////////////////////////////////////
    void logSoftMaxForVector(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

        if(!input.isVector() || !output.isVector())
            throw std::runtime_error("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

        auto xType = input.dataType();
        BUILD_SINGLE_SELECTOR(xType, _logSoftMaxForVector, (input.getBuffer(), input.getShapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    void softmax(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {

        const int rank = input.rankOf();

        if(input.isVector()) {
        
            if(rank == 1 || input.sizeAt(dimension) != 1)
                softMaxForVector(context, input, output);
            else
                output = 1.;
        }
        else {
            auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
            auto exponents = (input - maxAlongDim).transform(transform::Exp);
            auto sumAlongDim = exponents.reduceAlongDims(reduce::Sum, {dimension}, true);

            // FIXME: assign?
            output.assign(exponents / sumAlongDim);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void prelu(graph::LaunchContext* context, const NDArray& input, const NDArray& alpha, NDArray& output) {

        if(!input.isActualOnDeviceSide()) input.syncToDevice();
        if(!alpha.isActualOnDeviceSide()) alpha.syncToDevice();        

        const auto xType = input.dataType();
        const auto yType = alpha.dataType();

        int threadsPerBlock = MAX_NUM_THREADS;
        int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        BUILD_DOUBLE_SELECTOR(xType, yType, preluCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), alpha.getSpecialBuffer(), alpha.getSpecialShapeInfo(), output.getSpecialBuffer()), LIBND4J_TYPES, FLOAT_TYPES);
        
        input.tickReadHost();
        alpha.tickReadHost();
        output.tickWriteDevice();
    }

    //////////////////////////////////////////////////////////////////////////
    void preluBP(graph::LaunchContext* context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {

        if(!input.isActualOnDeviceSide()) input.syncToDevice();
        if(!alpha.isActualOnDeviceSide()) alpha.syncToDevice();
        if(!dLdO.isActualOnDeviceSide())  dLdO.syncToDevice();

        const auto xType = input.dataType();
        const auto yType = alpha.dataType();
        const auto zType = dLdO.dataType();

        int threadsPerBlock = MAX_NUM_THREADS;
        int blocksPerGrid = (alpha.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        BUILD_TRIPLE_SELECTOR(xType, yType, zType, preluBPCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), alpha.getSpecialBuffer(), alpha.getSpecialShapeInfo(), dLdO.getSpecialBuffer(),  dLdO.getSpecialShapeInfo(), dLdI.getSpecialBuffer(), dLdI.getSpecialShapeInfo(), dLdA.getSpecialBuffer(), dLdA.getSpecialShapeInfo()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        
        input.tickReadHost();
        alpha.tickReadHost();
        dLdO.tickReadHost();
        dLdI.tickWriteDevice();
        dLdA.tickWriteDevice();

    }


    template <typename T>
    static void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
        auto routine = LAMBDA_T(_x, threshold) {
            return _x > (T)threshold? _x: (T)0.f;
        };
        const_cast<NDArray&>(input).applyLambda<T>(routine, &output);
    }

    void thresholdRelu(graph::LaunchContext* context, NDArray const& input, double threshold, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), FLOAT_TYPES);
    }

    template <typename T>
    static void thresholdReluDerivative_(NDArray* input, double theta, NDArray* dLdO, NDArray* output) {

    }

    void thresholdReluDerivative(graph::LaunchContext* context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (input, threshold, dLdO, output), FLOAT_TYPES);
    }


BUILD_SINGLE_TEMPLATE(template void _softMaxForVector, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void _logSoftMaxForVector, (void *input, Nd4jLong *inShapeInfo, void *output, Nd4jLong *outShapeInfo), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void thresholdReluDerivative_, (NDArray* input, double threshold, NDArray* dLdO, NDArray* output), FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void preluCudaLauncher,   (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template void preluBPCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vIn, const Nd4jLong *inShapeInfo, const void *vAlpha, const Nd4jLong *alphaShapeInfo, const void *vdLdO,  const Nd4jLong *dLdOShapeInfo, void *vdLdI,  const Nd4jLong *dLdIShapeInfo, void *vdLdA,  const Nd4jLong *dLdAShapeInfo), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);


}
}
}

