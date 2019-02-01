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
// Created by GS <sgazeos@gmail.com> on 3/21/2018.
//

#include "ResultSet.h"
#include <ops/declarable/helpers/matrix_diag_part.h>
#include <Status.h>
#include <ShapeUtils.h>
#include <ShapeUtils.h>
#include <TAD.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void matrixDiagPartKernel(void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength,
                                            Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
                                            Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets) {
        int totalThreads = blockDim.x;
        for (Nd4jLong i = blockIdx.x; i < numTads; i += gridDim.x) {
            auto yOffset = tadInputOffsets[i];
            auto xOffset = tadOutputOffsets[i];
            for (Nd4jLong j = threadIdx.x; j < inputLength; j += totalThreads) {
                Nd4jLong coords[2] = {j, j};
                Nd4jLong tadOffset = shape::getOffset(0, shape::shapeOf(tadOnlyInputShapeInfo), shape::stride(tadOnlyInputShapeInfo), coords, 2);
                //shape::getIndexOffset(j, tadOnlyOutputShapeInfo, inputLength)
                *(reinterpret_cast<T*>(outputBuffer) + xOffset + shape::getIndexOffset(j, tadOnlyOutputShapeInfo, inputLength)) = *(reinterpret_cast<T const*>(inputBuffer) + yOffset + tadOffset);
            }
        }
    }

//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
    template <typename T>
    int _matrixDiagPart(graph::LaunchContext* context, const NDArray* input, NDArray* output) {
        auto stream = context->getCudaStream();
        auto listOut  = output->allTensorsAlongDimension({output->rankOf() - 1});
        auto listDiag = input->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf() - 1});

        if (listOut->size() != listDiag->size()) {
            nd4j_printf("matrix_diag_part: Input matrix has wrong shape.", "");
            return ND4J_STATUS_VALIDATION;
        }
        Nd4jLong lastDimension = nd4j::math::nd4j_min(input->sizeAt(-2), input->sizeAt(-1));

        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(output->rankOf(), {output->rankOf() - 1});
        const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(input->getShapeInfo(), dimsToExclude); //this->tensorsAlongDimension({dimension});
        //printf("Repeat delta %lld, numTads %lld\n", repeatDelta, numTads);
        //tadOnlyInputShapeInfo, tadInputOffsets, tadOnlyOutputShapeInfo, tadOutputOffsets;
        std::vector<int> outputDims({output->rankOf() - 1});
        shape::TAD tadOutput(output->getShapeInfo(), outputDims.data(), outputDims.size());
        tadOutput.createTadOnlyShapeInfo();
        tadOutput.createOffsets();
        if (!input->isActualOnDeviceSide())
            input->syncToDevice();

        std::vector<int> inputDims({input->rankOf() - 2, input->rankOf() - 1});
        shape::TAD tadInput(input->getShapeInfo(), inputDims.data(), inputDims.size());
        tadInput.createTadOnlyShapeInfo();
        tadInput.createOffsets();
        if (!input->isActualOnDeviceSide())
            input->syncToDevice();

        // prepare input arrays for prepareDataForCuda function
        std::vector<std::pair<void*,size_t>> hostData;
        hostData.emplace_back(tadInput.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadInput.tadOnlyShapeInfo));	// 1 -- xTadShapeInfo
        hostData.emplace_back(tadInput.tadOffsets, tadInput.numTads * sizeof(Nd4jLong));							// 2 -- xTadOffsets
        hostData.emplace_back(tadOutput.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadOutput.tadOnlyShapeInfo));	// 1 -- xTadShapeInfo
        hostData.emplace_back(tadOutput.tadOffsets, tadOutput.numTads * sizeof(Nd4jLong));							// 2 -- xTadOffsets
        std::vector<void*> devicePtrs(hostData.size(), nullptr);

        // create cuda stream and LaunchContext
        cudaError_t cudaResult;
        //cudaStream_t stream;
        //cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
        //cudaStream_t* stream = this->getContext()->getCudaStream();
        // allocate required amount of global device memory and copy host data to it
//    cudaResult = allocateDeviceMem(*pLc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);
        for(int i = 0; i < devicePtrs.size(); ++i) {
            cudaResult = cudaMalloc(reinterpret_cast<void **>(&devicePtrs[i]), hostData[i].second);
            if(cudaResult != 0) throw cuda_exception::build("Cannot allocate memory for tads on device", cudaResult);
            cudaResult = cudaMemcpy(devicePtrs[i], hostData[i].first, hostData[i].second, cudaMemcpyHostToDevice);
            if(cudaResult != 0) throw cuda_exception::build("Cannot copy memory block for tads on device", cudaResult);
        }

        dim3 launchDims(256, 512, 8192);
        matrixDiagPartKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(input->getSpecialBuffer(), output->getSpecialBuffer(), numTads, lastDimension, (Nd4jLong*)devicePtrs[0], (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2], (Nd4jLong*)devicePtrs[3]);
        for(int i = 0; i < devicePtrs.size(); ++i) {
            cudaResult = cudaFree(devicePtrs[i]);
            if(cudaResult != 0) throw cuda_exception::build("Cannot allocate memory for tads on device", cudaResult);
        }
        return Status::OK();
    }

    int matrixDiagPart(graph::LaunchContext* context, const NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return _matrixDiagPart, (context, input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int _matrixDiagPart, (graph::LaunchContext* context, const NDArray* input, NDArray* output), LIBND4J_TYPES);

}
}
}