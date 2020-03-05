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

#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <array/NDArrayFactory.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
namespace ops {
namespace helpers {

    // -------------------------------------------------------------------------------------------------------------- //
    // Sorted segments ops implementations

    template <typename T, typename I>
    static bool segmentIndicesValidate_(NDArray* indices, NDArray& aexpected, NDArray& aoutput) {
        return true;
    }

    bool segmentIndicesValidate(sd::LaunchContext* context , NDArray* indices, NDArray& expected, NDArray& output) {
        BUILD_DOUBLE_SELECTOR(output.dataType(), indices->dataType(), return segmentIndicesValidate_, (indices, expected, output), NUMERIC_TYPES, INDEXING_TYPES);
    }

    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted segment ops functors implementation
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename I>
    static __global__ void unsortedSegmentIndexValidateKernel(I* indices, Nd4jLong* indicesShape, I expected, I* found) {
        __shared__ bool onlyTrue;
        __shared__ Nd4jLong len;

        if (threadIdx.x == 0) {
            onlyTrue = true;
            len = shape::length(indicesShape);
        }
        __syncthreads();
        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = gridDim.x * blockDim.x;
        for (int e = start; e < len && onlyTrue; e += step) {
            sd::math::atomics::nd4j_atomicMax(found, indices[e]);
            if (expected < *found)
                onlyTrue = false;
        }
    }

    template <typename I>
    static bool unsortedSegmentIndicesValidate_(sd::LaunchContext* context , NDArray* indices, Nd4jLong expected, Nd4jLong& output) {
        output = expected;
        I found = output;
        I exp = expected;
        auto stream = context->getCudaStream();
        I* devFound;
        cudaMalloc(&devFound, sizeof(I));
        cudaMemcpy(devFound, &found, sizeof(I), cudaMemcpyHostToDevice);
        unsortedSegmentIndexValidateKernel<I><<<1, indices->lengthOf(), 128, *stream>>>(reinterpret_cast<I*>(indices->specialBuffer()), indices->specialShapeInfo(), exp, devFound);
        cudaMemcpy(&found, devFound, sizeof(I), cudaMemcpyDeviceToHost);
        cudaFree(devFound);
        output = found;
        return expected == output;
    }

    bool unsortedSegmentIndicesValidate(sd::LaunchContext* context , NDArray* indices, Nd4jLong expected, Nd4jLong& output) {
        BUILD_SINGLE_SELECTOR(indices->dataType(), return unsortedSegmentIndicesValidate_, (context, indices, expected, output), INDEXING_TYPES);
    }

    // -------------------------------------------------------------------------------------------------------------- //

    // -------------------------------------------------------------------------------------------------------------- //
    // fill up segments starts and ends - splitted ordered case
    template <typename I>
    static __global__ void fillUpSegmentsKernel(void* indices, Nd4jLong* indexShape, int numClasses, int* classesRangesStart, int* classesRangesLenghts) {
        __shared__ I* idxBuf;
        __shared__ Nd4jLong idxLen;
        __shared__ int* result;
        if (threadIdx.x == 0) {
            idxBuf = reinterpret_cast<I*>(indices);
            idxLen = shape::length(indexShape);
        }
        __syncthreads();

        auto tid = threadIdx.x + blockDim.x * blockIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (auto j = tid; j < idxLen; j += step) {
            auto pos = idxBuf[j];
            sd::math::atomics::nd4j_atomicMin<int>(&classesRangesStart[pos], (int)j);
            sd::math::atomics::nd4j_atomicAdd<int>(&classesRangesLenghts[pos], 1);
        }
    }

        // -------------------------------------------------------------------------------------------------------------- //

        template <typename I>
        static void fillUpSegments_(NDArray* indices, Nd4jLong numClasses, NDArray& classesRangesBegs, NDArray& classesRangesLens) {
            dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
            int* begins = reinterpret_cast<int*>(classesRangesBegs.getSpecialBuffer());
            int* lengths = reinterpret_cast<int*>(classesRangesLens.getSpecialBuffer());
            auto stream = classesRangesBegs.getContext()->getCudaStream();
            fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream >>>(indices->specialBuffer(), indices->specialShapeInfo(), numClasses, begins, lengths);
        }
        // -------------------------------------------------------------------------------------------------------------- //

        void fillUpSegments(NDArray* indices, Nd4jLong numClasses, NDArray& classesRangesBegs, NDArray& classesRangesLens) {
            BUILD_SINGLE_SELECTOR(indices->dataType(), fillUpSegments_, (indices, numClasses, classesRangesBegs, classesRangesLens), INDEXING_TYPES);
        }
        // -------------------------------------------------------------------------------------------------------------- //

}
}
}
// -------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------- //
