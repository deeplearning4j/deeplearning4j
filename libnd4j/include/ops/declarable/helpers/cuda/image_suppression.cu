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

        T minYPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous2, 2)]);
        T minXPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous3, 2)]);
        T maxYPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous2, 2)]);
        T maxXPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous3, 2)]);
        T minYNext = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next2, 2)]);
        T minXNext = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next3, 2)]);
        T maxYNext = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next2, 2)]);
        T maxXNext = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next3, 2)]);

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
    static __global__ void nonMaxSuppressionKernel(T* boxes, Nd4jLong* boxesShape, I* indices, int* selectedIndices, Nd4jLong numBoxes, I* output, Nd4jLong* outputShape, T threshold) {
        __shared__ Nd4jLong outputLen;

        if (threadIdx.x == 0) {
            outputLen = shape::length(outputShape);
        }
        __syncthreads();

        auto numSelected = blockIdx.x;
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;
//        for (int numSelected = blockIdx.x; numSelected < outputLen; numSelected += gridDim.x) {
        for (int i = start; i < numBoxes; i += step) {
                bool shouldSelect = true;
                for (int j = numSelected - 1; shouldSelect && j >= 0; --j) {
                    if (needToSuppressWithThreshold<T>(boxes, boxesShape, indices[i], indices[selectedIndices[j]], threshold)) {
                        shouldSelect = false;
                    }
                }

                if (shouldSelect) {
                    auto zPos = shape::getIndexOffset(numSelected, outputShape, outputLen);
                    output[zPos] = indices[i];
                    selectedIndices[numSelected] = i;
                }

        }
    }

    template <typename T, typename I>
    static __global__ void sortIndices(I* indices, Nd4jLong* indexShape, T* scores, Nd4jLong* scoreShape) {
        __shared__ Nd4jLong len;
//        __shared__ Nd4jLong* sortedPart;
//        __shared__ Nd4jLong part;
//        __shared__ Nd4jLong partSize;

        if (threadIdx.x == 0) {
//            blocksPerArr = (gridDim.x + numOfArrs - 1) / numOfArrs;     // ceil
//            part = blockIdx.x / blocksPerArr;

            len = shape::length(indexShape);
//            __shared__ Nd4jLong* shmem = shared[];
//            sortedPart = shmem;
        }

        for (int m = 0; m < len; m++) {
            if (m % 2 == 0) {
                for (int tid = threadIdx.x; tid < len; tid += blockDim.x) {
                    auto top = 2 * tid + 1;
                    if (top < len) {
                        auto t0 = shape::getIndexOffset(top - 1, indexShape, len);
                        auto t1 = shape::getIndexOffset(top, indexShape, len);
                        auto z0 = shape::getIndexOffset(top - 1, scoreShape, len);
                        auto z1 = shape::getIndexOffset(top, scoreShape, len);

                        if (scores[t0] < scores[t1]) {
                            // swap indices first
                            Nd4jLong di0 = indices[t0];
                            indices[t0] = indices[t1];
                            indices[t1] = di0;

                            //swap scores next
//                            T dz0 = scores[z0];
//                            scores[z0] = scores[z1];
//                            scores[z1] = dz0;
                        }
                    }
                }
            } else {
                for (int tid = threadIdx.x; tid < len; tid += blockDim.x) {
                    auto top = 2 * tid + 2;
                    if (top < len) {
                        auto t0 = shape::getIndexOffset(top - 1, indexShape, len);
                        auto t1 = shape::getIndexOffset(top, indexShape, len);
                        auto z0 = shape::getIndexOffset(top - 1, scoreShape, len);
                        auto z1 = shape::getIndexOffset(top, scoreShape, len);

                        if (scores[t0] < scores[t1]) {
                            // swap indices first
                            Nd4jLong di0 = indices[t0];
                            indices[t0] = indices[t1];
                            indices[t1] = di0;

                            //swap scores next
//                            T dz0 = scores[z0];
//                            scores[z0] = scores[z1];
//                            scores[z1] = dz0;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    template <typename T, typename I>
    static void nonMaxSuppressionV2_(nd4j::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {boxes, scales});
        NDArray* indices = NDArrayFactory::create_<I>('c', {scales->lengthOf()}); // - 1, scales->lengthOf()); //, scales->getContext());
        indices->linspace(0);
        NDArray scores(*scales);
        indices->syncToHost(); //linspace(0);
        I* indexBuf = reinterpret_cast<I*>(indices->specialBuffer());
        T* scoreBuf = reinterpret_cast<T*>(scores.specialBuffer());
        sortIndices<T, I><<<1, 32, 128, *stream>>>(indexBuf, indices->specialShapeInfo(), scoreBuf, scores.specialShapeInfo());
        // TO DO: sort indices using scales as value row
        //std::sort(indices.begin(), indices.end(), [scales](int i, int j) {return scales->e<T>(i) > scales->e<T>(j);});
        indices->tickWriteDevice();
        indices->syncToHost();
        indices->printIndexedBuffer("AFTERSORT OUTPUT");
        NDArray selected = NDArrayFactory::create<int>({output->lengthOf()});

        NDArray selectedIndices = NDArrayFactory::create<int>({output->lengthOf()});
        int numSelected = 0;
        int numBoxes = boxes->sizeAt(0);
        T* boxesBuf = reinterpret_cast<T*>(boxes->specialBuffer());
//        Nd4jLong* indicesData = reinterpret_cast<Nd4jLong*>(indices->specialBuffer());
//        int* selectedData = reinterpret_cast<int*>(selected.specialBuffer());
        int* selectedIndicesData = reinterpret_cast<int*>(selectedIndices.specialBuffer());
        I* outputBuf = reinterpret_cast<I*>(output->specialBuffer());
        nonMaxSuppressionKernel<T, I><<<output->lengthOf(), 512, 1024, *stream>>>(boxesBuf, boxes->specialShapeInfo(), indexBuf, selectedIndicesData, numBoxes, outputBuf, output->specialShapeInfo(), T(threshold));
        NDArray::registerSpecialUse({output}, {boxes, scales});
//        for (int i = 0; i < boxes->sizeAt(0); ++i) {
//            if (selected.size() >= output->lengthOf()) break;
//            bool shouldSelect = true;
//            // Overlapping boxes are likely to have similar scores,
//            // therefore we iterate through the selected boxes backwards.
//            for (int j = numSelected - 1; j >= 0; --j) {
//                if (needToSuppressWithThreshold(*boxes, indices[i], indices[selectedIndices[j]], T(threshold)) {
//                    shouldSelect = false;
//                    break;
//                }
//            }
//            if (shouldSelect) {
//                selected.push_back(indices[i]);
//                selectedIndices[numSelected++] = i;
//            }
//        }
//        for (size_t e = 0; e < selected.size(); ++e)
//            output->p<int>(e, selected[e]);
//
        delete indices;
    }

    void nonMaxSuppressionV2(nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(boxes->dataType(), output->dataType(), nonMaxSuppressionV2_, (context, boxes, scales, maxSize, threshold, output), FLOAT_TYPES, INTEGER_TYPES);
    }
    BUILD_DOUBLE_TEMPLATE(template void nonMaxSuppressionV2_, (nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);

}
}
}