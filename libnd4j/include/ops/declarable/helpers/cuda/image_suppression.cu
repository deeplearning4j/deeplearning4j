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

#include <ops/declarable/helpers/image_suppression.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __device__ bool needToSuppressWithThreshold(T* boxes, Nd4jLong* boxesShape, int previousIndex, int nextIndex, T threshold) {
        Nd4jLong previous0[] = {previousIndex, 0};
        Nd4jLong previous1[] = {previousIndex, 1};
        Nd4jLong previous2[] = {previousIndex, 2};
        Nd4jLong previous3[] = {previousIndex, 3};
        Nd4jLong next0[] = {nextIndex, 0};
        Nd4jLong next1[] = {nextIndex, 1};
        Nd4jLong next2[] = {nextIndex, 2};
        Nd4jLong next3[] = {nextIndex, 3};
        Nd4jLong* shapeOf = shape::shapeOf(boxesShape);
        Nd4jLong* strideOf = shape::stride(boxesShape);
        T minYPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shapeOf, strideOf, previous0, 2)], boxes[shape::getOffset(0, shapeOf, strideOf, previous2, 2)]);
        T minXPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shapeOf, strideOf, previous1, 2)], boxes[shape::getOffset(0, shapeOf, strideOf, previous3, 2)]);
        T maxYPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shapeOf, strideOf, previous0, 2)], boxes[shape::getOffset(0, shapeOf, strideOf, previous2, 2)]);
        T maxXPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shapeOf, strideOf, previous1, 2)], boxes[shape::getOffset(0, shapeOf, strideOf, previous3, 2)]);
        T minYNext = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shapeOf, strideOf, next0, 2)],     boxes[shape::getOffset(0, shapeOf, strideOf, next2, 2)]);
        T minXNext = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shapeOf, strideOf, next1, 2)],     boxes[shape::getOffset(0, shapeOf, strideOf, next3, 2)]);
        T maxYNext = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shapeOf, strideOf, next0, 2)],     boxes[shape::getOffset(0, shapeOf, strideOf, next2, 2)]);
        T maxXNext = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shapeOf, strideOf, next1, 2)],     boxes[shape::getOffset(0, shapeOf, strideOf, next3, 2)]);

        T areaPrev = (maxYPrev - minYPrev) * (maxXPrev - minXPrev);
        T areaNext = (maxYNext - minYNext) * (maxXNext - minXNext);

        if (areaNext <= T(0.f) || areaPrev <= T(0.f)) return false;

        T minIntersectionY = nd4j::math::nd4j_max(minYPrev, minYNext);
        T minIntersectionX = nd4j::math::nd4j_max(minXPrev, minXNext);
        T maxIntersectionY = nd4j::math::nd4j_min(maxYPrev, maxYNext);
        T maxIntersectionX = nd4j::math::nd4j_min(maxXPrev, maxXNext);
        T intersectionArea =
                nd4j::math::nd4j_max(T(maxIntersectionY - minIntersectionY), T(0.0f)) *
                nd4j::math::nd4j_max(T(maxIntersectionX - minIntersectionX), T(0.0f));
        T intersectionValue = intersectionArea / (areaPrev + areaNext - intersectionArea);
        return intersectionValue > threshold;
    };

    template <typename T, typename I>
    static __global__ void shouldSelectKernel(T* boxesBuf, Nd4jLong* boxesShape, I* indexBuf, I* selectedIndicesData, double threshold, int numSelected, int i, bool* shouldSelect) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = gridDim.x * blockDim.x;
        __shared__ bool shouldSelectShared;
        if (threadIdx.x == 0) {
            shouldSelectShared = shouldSelect[0];
        }
        __syncthreads();
        for (int j = numSelected - 1 - tid; j >= 0; j -= step) {
            if (shouldSelectShared) {
                if (needToSuppressWithThreshold(boxesBuf, boxesShape, indexBuf[i],
                                                                  indexBuf[selectedIndicesData[j]], T(threshold)))
                    shouldSelectShared = false;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *shouldSelect = shouldSelectShared;
        }
    }
    template <typename I>

    static __global__ void copyIndices(void* indices,  void* indicesLong, Nd4jLong len) {
        __shared__ I* indexBuf;
        __shared__ Nd4jLong* srcBuf;
        if (threadIdx.x == 0) {
            indexBuf = reinterpret_cast<I*>(indices);
            srcBuf = reinterpret_cast<Nd4jLong*>(indicesLong);
        }
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for (auto i = tid; i < len; i += step)
            indexBuf[i] = (I)srcBuf[i];
    }

    template <typename T, typename I>
    static void nonMaxSuppressionV2_(nd4j::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {boxes, scales});
        std::unique_ptr<NDArray> indices(NDArrayFactory::create_<I>('c', {scales->lengthOf()})); // - 1, scales->lengthOf()); //, scales->getContext());
        indices->linspace(0);
        indices->syncToDevice(); // linspace only on CPU, so sync to Device as well

        NDArray scores(*scales);
        Nd4jPointer extras[2] = {nullptr, stream};

        sortByValue(extras, indices->buffer(), indices->shapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), scores.buffer(), scores.shapeInfo(), scores.specialBuffer(), scores.specialShapeInfo(), true);
        // TO DO: sort indices using scales as value row
        //std::sort(indices.begin(), indices.end(), [scales](int i, int j) {return scales->e<T>(i) > scales->e<T>(j);});
        I* indexBuf = reinterpret_cast<I*>(indices->specialBuffer());

        NDArray selectedIndices = NDArrayFactory::create<I>('c', {output->lengthOf()});
        int numSelected = 0;
        int numBoxes = boxes->sizeAt(0);
        T* boxesBuf = reinterpret_cast<T*>(boxes->specialBuffer());

        I* selectedIndicesData = reinterpret_cast<I*>(selectedIndices.specialBuffer());
        I* outputBuf = reinterpret_cast<I*>(output->specialBuffer());

        bool* shouldSelectD;
        auto err = cudaMalloc(&shouldSelectD, sizeof(bool));
        if (err) {
            throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot allocate memory for bool flag", err);
        }
        for (I i = 0; i < boxes->sizeAt(0); ++i) {
            bool shouldSelect = numSelected < output->lengthOf();
            if (shouldSelect) {
                err = cudaMemcpy(shouldSelectD, &shouldSelect, sizeof(bool), cudaMemcpyHostToDevice);
                if (err) {
                    throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot set up bool flag to device", err);
                }

                shouldSelectKernel<T> <<< 128, 256, 1024, *stream >>>
                                                           (boxesBuf, boxes->specialShapeInfo(), indexBuf, selectedIndicesData, threshold, numSelected, i, shouldSelectD);
                err = cudaMemcpy(&shouldSelect, shouldSelectD, sizeof(bool), cudaMemcpyDeviceToHost);
                if (err) {
                    throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot set up bool flag to host", err);
                }
            }

            if (shouldSelect) {
                cudaMemcpy(reinterpret_cast<I*>(output->specialBuffer()) + numSelected, indexBuf + i, sizeof(I), cudaMemcpyDeviceToDevice);
                cudaMemcpy(selectedIndicesData + numSelected, &i, sizeof(I), cudaMemcpyHostToDevice);
                numSelected++;
            }
        }

        err = cudaFree(shouldSelectD);
        if (err) {
            throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot deallocate memory for bool flag", err);
        }

    }

    void nonMaxSuppressionV2(nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(boxes->dataType(), output->dataType(), nonMaxSuppressionV2_, (context, boxes, scales, maxSize, threshold, output), FLOAT_TYPES, INTEGER_TYPES);
    }
    BUILD_DOUBLE_TEMPLATE(template void nonMaxSuppressionV2_, (nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);

}
}
}
