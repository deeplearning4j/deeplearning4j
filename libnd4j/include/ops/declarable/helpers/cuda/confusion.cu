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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/confusion.h>
#include <cuda_exception.h>
#include <TAD.h>
#include <PointersManager.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    __global__ static void confusionFunctorKernel(Nd4jLong* labelsBuffer, Nd4jLong* predictionBuffer, Nd4jLong bufferLength, void const* weightsBuffer, void* outputBuffer, Nd4jLong* tadShape, Nd4jLong* tadOffsets) {
        __shared__ int arrIdx, blocksPerArr;
        __shared__ T *z;
        __shared__ T const* w;
        __shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLen;

        if (threadIdx.x == 0) {
            z = reinterpret_cast<T*>(outputBuffer);
            w = reinterpret_cast<T const*>(weightsBuffer);
            arrLen = shape::length(tadShape);
        }
        __syncthreads();

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        for (Nd4jLong t = tid; t < bufferLength; t += step) {
            //auto tX = reinterpret_cast<T*>(inputList[t]);
            //auto xShape = reinterpret_cast<Nd4jLong*>(inputShapeList[t]);
            auto label = labelsBuffer[t]; //->e<Nd4jLong>(j);
            auto pred = predictionBuffer[t]; //->e<Nd4jLong>(j);
            auto tZ = z + tadOffsets[label];
            T val = (weightsBuffer == nullptr ? (T)1.0f : w[t]);

            //for (int e = threadIdx.x; e < arrLen; e += blockDim.x) {

            tZ[shape::getIndexOffset(pred, tadShape, arrLen)] = val; //tX[shape::getIndexOffset(e, , arrLen)];
        }
    }

    template <typename T>
    void _confusionFunctor(graph::LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights, NDArray* output) {
//        std::unique_ptr<ResultSet> arrs(output->allTensorsAlongDimension({1}));
//
//#pragma omp parallel for if(labels->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
//        for (int j = 0; j < labels->lengthOf(); ++j){
//            auto label = labels->e<Nd4jLong>(j);
//            auto pred = predictions->e<Nd4jLong>(j);
//            T value = (weights == nullptr ? (T)1.0f : weights->e<T>(j));
//            (*arrs->at(label)).p<T>(pred, value);
//        }

        std::vector<int> axis({1}); // = ShapeUtils::evalDimsToExclude(outArr->rankOf(), {dim});
        shape::TAD tadOutput(output->shapeInfo(), axis.data(), axis.size());
        tadOutput.createTadOnlyShapeInfo();
        tadOutput.createOffsets();

        PointersManager manager(context, "helpers::confusion");
        auto pTadShape = (Nd4jLong *) manager.replicatePointer(tadOutput.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadOutput.tadOnlyShapeInfo));
        auto pTadOffsets = (Nd4jLong *) manager.replicatePointer(tadOutput.tadOffsets, tadOutput.numTads * sizeof(Nd4jLong));

        Nd4jLong* labelsLongBuffer = labels->dataType() == nd4j::DataType::INT64?(Nd4jLong*)labels->specialBuffer():nullptr;
        Nd4jLong* predictionLongBuffer = predictions->dataType() == nd4j::DataType::INT64?(Nd4jLong*)predictions->specialBuffer():nullptr;

        if (labelsLongBuffer == nullptr) {
            cudaError_t err = cudaMalloc(&labelsLongBuffer, labels->lengthOf() * sizeof(Nd4jLong));
            if (err != 0)
                throw nd4j::cuda_exception::build("Cannot allocate memory for labels long buffer", err);
            // copy with type conversion
        }

        if (predictionLongBuffer == nullptr) {
            cudaError_t err = cudaMalloc(&predictionLongBuffer, predictions->lengthOf() * sizeof(Nd4jLong));
            if (err != 0)
                throw nd4j::cuda_exception::build("Cannot allocate memory for predictions long buffer", err);
            // copy with type conversion
        }

        auto bufferLength = labels->lengthOf();
        dim3 launchDims(256, 512, 8192);
        auto stream = context->getCudaStream();
        confusionFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(labelsLongBuffer, predictionLongBuffer,
                bufferLength, weights != nullptr? weights->getSpecialBuffer():nullptr, output->specialBuffer(), pTadShape, pTadOffsets);
        manager.synchronize();
    }

    void confusionFunctor(graph::LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights, NDArray* output) {
        auto xType = output->dataType(); // weights can be null

        BUILD_SINGLE_SELECTOR(xType, _confusionFunctor, (context, labels, predictions, weights, output), NUMERIC_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _confusionFunctor, (graph::LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights, NDArray* output);, NUMERIC_TYPES);

}
}
}