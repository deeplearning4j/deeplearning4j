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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/nth_element.h>
#include <TAD.h>
#include <PointersManager.h>
#include <NativeOps.h>
#include <helpers/ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void fillUpElementKernel(void* outputBuffer, Nd4jLong* outputShapeInfo, void* inputBuffer, Nd4jLong* inputShapeInfo, Nd4jLong* pTadShape, Nd4jLong* pTadOffsets, Nd4jLong n) {
        __shared__ T *z, *x;
        __shared__ Nd4jLong bufferLength, arrLen;

        if (threadIdx.x == 0) {
            z = reinterpret_cast<T*>(outputBuffer);
            x = reinterpret_cast<T*>(inputBuffer);
            arrLen = shape::length(pTadShape);
            bufferLength = shape::length(outputShapeInfo);
        }
        __syncthreads();

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        for (int t = tid; t < bufferLength; t += step) {
            auto tX = x + pTadOffsets[t];
            z[shape::getIndexOffset(t, outputShapeInfo, bufferLength)] = tX[shape::getIndexOffset(n, pTadShape, arrLen)]; //tX];
        }
    }

    template <typename T>
    void nthElementFunctor_(nd4j::LaunchContext * context, NDArray* input, NDArray* nVal, NDArray* output, bool reverse) {
            Nd4jLong n = nVal->e<Nd4jLong>(0);
            NDArray sortedVals(*input);
            Nd4jPointer params[2];
            params[0] = context->getCudaStream();
            params[1] = *context->getCudaStream();

            if (input->isVector()) {
                NativeOps ops;
                ops.sort(params, sortedVals.buffer(), sortedVals.shapeInfo(), sortedVals.specialBuffer(), sortedVals.specialShapeInfo(), reverse);

                cudaMemcpy(reinterpret_cast<T*>(output->specialBuffer()), reinterpret_cast<T*>(sortedVals.specialBuffer()) + n, sizeof(T), cudaMemcpyDeviceToDevice);
            }
            else { // rank greater than 1
                std::vector<int> lastDims({input->rankOf() - 1});// = ShapeUtils::evalDimsToExclude(input->rankOf(), {input->rankOf() - 1});

                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(sortedVals.getShapeInfo(), lastDims);

                //PointersManager manager(context, "helpers::nth_element");
                auto pTadShape = packX.specialShapeInfo();
                auto pTadOffsets = packX.specialOffsets();
                //auto pLastDimData = (int*) manager.replicatePointer(lastDims.data(), lastDims.size() * sizeof(int));

                NativeOps ops;
                ops.sortTad(params, sortedVals.buffer(), sortedVals.shapeInfo(), sortedVals.specialBuffer(), sortedVals.specialShapeInfo(), lastDims.data(), lastDims.size(), pTadShape, pTadOffsets, reverse);
                auto stream = context->getCudaStream();
                fillUpElementKernel<T><<<32, 64, 1024, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), sortedVals.specialBuffer(), sortedVals.specialShapeInfo(), pTadShape, pTadOffsets, n);
                //manager.synchronize();
        }
    }
    void nthElementFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* n, NDArray* output, bool reverse) {
    BUILD_SINGLE_SELECTOR(input->dataType(), nthElementFunctor_, (context, input, n, output, reverse), LIBND4J_TYPES);

    }
    BUILD_SINGLE_TEMPLATE(template void nthElementFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* n, NDArray* output, bool reverse), LIBND4J_TYPES);
    
}
}
}
