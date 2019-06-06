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
#include <NDArrayFactory.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {

    // -------------------------------------------------------------------------------------------------------------- //
    // Segment ops linear kernels
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentMaxLinearKernel(void* input, Nd4jLong* inputShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
         __shared__ T* val;
         __shared__ Nd4jLong xLen, zLen, segment, zIndex;
         __shared__ T* x;
         __shared__ T* z;
         __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            extern __shared__ unsigned char shmem[];
            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape, zLen);
                start = starts[segment];
                finish = start + lengths[segment];
                z[zIndex] = x[shape::getIndexOffset(start, inputShape, xLen)];
                val[segment] = z[zIndex];
            }

        }
        __syncthreads();

         for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
             auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
             nd4j::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
         }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void unsortedSegmentMaxLinearKernel(void* input, Nd4jLong* inputShape, void* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = blockIdx.x;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            //start = starts[segment];
            //finish = start + lengths[segment];
            if (lengths[segment] > 0)
                z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape, xLen)];
            else
                z[zIndex] = -DataTypeUtils::max<T>();
        }
        __syncthreads();
        if (lengths[segment] > 0)
        for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
            if (y[yIndex] == segment) {
                nd4j::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentMinLinearKernel(void* input, Nd4jLong* inputShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            extern __shared__ unsigned char shmem[];
            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape, zLen);
                start = starts[segment];
                finish = start + lengths[segment];
                z[zIndex] = x[shape::getIndexOffset(start, inputShape, xLen)];
                val[segment] = z[zIndex];
            }

        }
        __syncthreads();

        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
           nd4j::math::atomics::nd4j_atomicMin(&z[zIndex], x[xIndex]);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void unsortedSegmentMinLinearKernel(void* input, Nd4jLong* inputShape, void* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = blockIdx.x;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            if (lengths[segment] > 0)
                z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape, xLen)];
            else
                z[zIndex] = DataTypeUtils::max<T>();

        }
        __syncthreads();
        if (lengths[segment] > 0)
            for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
                auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
                if (y[yIndex] == segment) {
                    nd4j::math::atomics::nd4j_atomicMin(&z[zIndex], x[xIndex]);
                }
            }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentSumLinearKernel(void* input, Nd4jLong* inputShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);

            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);


            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape, zLen);
                start = starts[segment];
                finish = start + lengths[segment];
                //val[segment] = ;
                z[zIndex] = x[shape::getIndexOffset(start, inputShape, xLen)];
            }

        }
        __syncthreads();

        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex]);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void unsortedSegmentSumLinearKernel(void* input, Nd4jLong* inputShape, void* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = blockIdx.x;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            if (lengths[segment] > 0)
                z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape, xLen)];
            else
                z[zIndex] = 0; //DataTypeUtils::max<T>();
        }
        __syncthreads();

        if (lengths[segment] > 0)
            for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
                auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
                if (y[yIndex] == segment && e != starts[segment]) {
                    nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex]);
                }
            }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentMeanLinearKernel(void* input, Nd4jLong* inputShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
//            extern __shared__ unsigned char shmem[];
//            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            //[zIndex] =
            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape, zLen);
                start = starts[segment];
                finish = start + lengths[segment];
                //val[segment] = ;
                z[zIndex] = T(x[shape::getIndexOffset(start, inputShape, xLen)] / lengths[segment]);
//                val[segment] = z[zIndex];
            }

        }
        __syncthreads();

        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            if (lengths[segment])
            nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex] / lengths[segment]));
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void unsortedSegmentMeanLinearKernel(void* input, Nd4jLong* inputShape, void* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
//            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x;// / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
//            extern __shared__ unsigned char shmem[];
//            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

//            if (segment < numOfClasses) {
            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            //start = starts[segment];
            //finish = start + lengths[segment];
            if (lengths[segment] > 0)
                z[zIndex] = T(x[shape::getIndexOffset(starts[segment], inputShape, xLen)] / T(lengths[segment]));
            else
                z[zIndex] = 0; //DataTypeUtils::max<T>();
//                val[segment] = z[zIndex];
//            }

        }
        __syncthreads();
        if (lengths[segment] > 0)
            for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
                auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
                if (y[yIndex] == segment && e != starts[segment]) {
                    nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex]/T(lengths[segment])));
                }
            }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentProdLinearKernel(void* input, Nd4jLong* inputShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            extern __shared__ unsigned char shmem[];
            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            if (segment < numOfClasses) {
                zIndex = shape::getIndexOffset(segment, outputShape, zLen);
                start = starts[segment];
                finish = start + lengths[segment];
                //val[segment] = ;
                z[zIndex] = x[shape::getIndexOffset(start, inputShape, xLen)];
                val[segment] = z[zIndex];
            }

        }
        __syncthreads();
//         auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//         auto step = blockDim.x * gridDim.x;

        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            nd4j::math::atomics::nd4j_atomicMul(&val[segment], x[xIndex]);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            z[zIndex] = val[segment];
        }

    }
    template <typename T, typename I>
    static __global__ void unsortedSegmentProdLinearKernel(void* input, Nd4jLong* inputShape, void* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
//            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x;// / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
//            extern __shared__ unsigned char shmem[];
//            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

//            if (segment < numOfClasses) {
            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            //start = starts[segment];
            //finish = start + lengths[segment];
            if (lengths[segment] > 0)
                z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape, xLen)];
            else
                z[zIndex] = 0; //DataTypeUtils::max<T>();
//                val[segment] = z[zIndex];
//            }

        }
        __syncthreads();
        if (lengths[segment] > 0)
            for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
                auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
                if (y[yIndex] == segment && e != starts[segment]) {
                    nd4j::math::atomics::nd4j_atomicMul(&z[zIndex], x[xIndex]);
                }
            }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T, typename I>
    static __global__ void unsortedSegmentSqrtNLinearKernel(void* input, Nd4jLong* inputShape, void* indices, Nd4jLong* indicesShape, int* starts, int* lengths, Nd4jLong numOfClasses, void* output, Nd4jLong* outputShape) {
        __shared__ T* val;
        __shared__ Nd4jLong xLen, zLen, segment, zIndex;
        __shared__ T* x;
        __shared__ T* z;
        __shared__ I* y; //int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
//            threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
            segment = blockIdx.x;// / threadsPerSegment;
            x = reinterpret_cast<T*>(input);
            z = reinterpret_cast<T*>(output);
            y = reinterpret_cast<I*>(indices);
//            extern __shared__ unsigned char shmem[];
//            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

//            if (segment < numOfClasses) {
            zIndex = shape::getIndexOffset(segment, outputShape, zLen);
            //start = starts[segment];
            //finish = start + lengths[segment];
            if (lengths[segment] > 0)
                z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape, xLen)] / nd4j::math::nd4j_sqrt<int, T>(lengths[segment]);
            else
                z[zIndex] = 0; //DataTypeUtils::max<T>();
//                val[segment] = z[zIndex];
//            }

        }
        __syncthreads();
        if (lengths[segment] > 0)
            for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
                auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
                auto yIndex = shape::getIndexOffset(e, indicesShape, xLen);
                if (y[yIndex] == segment && e != starts[segment]) {
                    nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex] / nd4j::math::nd4j_sqrt<int, T>(lengths[segment]));
                }
            }
    }
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
            nd4j::math::atomics::nd4j_atomicMin(&classesRangesStart[pos], (int)j);
            nd4j::math::atomics::nd4j_atomicAdd(&classesRangesLenghts[pos], 1);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    // -------------------------------------------------------------------------------------------------------------- //
    // fill up segments starts and counts - cumulative  case
    template <typename I>
    static __global__ void fillUpUnsortedSegmentsKernel(void* indices, Nd4jLong* indexShape, int numClasses, int* classes) {
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
            auto k = idxBuf[j];
            auto beginPos = 2 * k;
            auto sizePos = beginPos + 1;
            printf("%d, %d\n", beginPos, sizePos);
            nd4j::math::atomics::nd4j_atomicMin(&classes[beginPos], (int)j);
            nd4j::math::atomics::nd4j_atomicAdd(&classes[sizePos], 1);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    // -------------------------------------------------------------------------------------------------------------- //
    // segment ops multidimentional cases
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static __global__ void segmentMaxTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads,
            Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf,
            Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets, T filler = 0) {

        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);

            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);
        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = x[xIndex];
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    nd4j::math::atomics::nd4j_atomicMax(&z[zIndex], x[xIndex]);
                }
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    // SegmentMin kernel
    template <typename T, typename I>
    static __global__ void segmentMinTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);
            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);

        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = x[xIndex];
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    nd4j::math::atomics::nd4j_atomicMin(&z[zIndex], x[xIndex]);
                }
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    // SegmentSum kernel
    template <typename T, typename I>
    static __global__ void segmentSumTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);
            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);

        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = x[xIndex];
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    if (lengths[segment])
                    nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex]);
                }
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    // SegmentMean kernel
    template <typename T, typename I>
    static __global__ void segmentMeanTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);
            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);

        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = T(x[xIndex]/lengths[segment]);
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    if (lengths[segment])
                        nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], T(x[xIndex]/lengths[segment]));
                }
            }
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //

    // SegmentProd kernel
    template <typename T, typename I>
    static __global__ void segmentProdTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);
            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);

        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = x[xIndex];
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    nd4j::math::atomics::nd4j_atomicMul(&z[zIndex], x[xIndex]);
                }
            }
        }
    }
    // SegmentSqrtN kernel
    template <typename T, typename I>
    static __global__ void segmentSqrtNTadKernel(void* inputBuf, Nd4jLong* inputShape, Nd4jLong* inputTads, Nd4jLong* inputTadOffsets, I* indices, int* starts, int* lengths, Nd4jLong numOfClasses, void* outputBuf, Nd4jLong* outputShape, Nd4jLong* outputTads, Nd4jLong* outputTadOffsets) {
        __shared__ T* val;
        __shared__ Nd4jLong len, segment, zIndex, total;
        __shared__ T* z;
        __shared__ int threadsPerSegment, start, finish;

        if (threadIdx.x == 0) {
            segment = indices[blockIdx.x]; // / threadsPerSegment;
            z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
            len = shape::length(inputTads);
            start = starts[segment];
            finish = start + lengths[segment];
            total = shape::sizeAt(inputShape, 0);

        }
        __syncthreads();

        auto idx = blockIdx.x;
        if (blockIdx.x <= total) {
            auto x = reinterpret_cast<T *>(inputBuf) + inputTadOffsets[idx];
            if (blockIdx.x == start) {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    z[zIndex] = x[xIndex] / nd4j::math::nd4j_sqrt<int, T>(lengths[segment]);
                }
            }
            else {
                for (auto e = threadIdx.x; e < len; e += blockDim.x) {
                    auto xIndex = shape::getIndexOffset(e, inputTads, len);
                    auto zIndex = shape::getIndexOffset(e, outputTads, len);
                    nd4j::math::atomics::nd4j_atomicAdd(&z[zIndex], x[xIndex] / nd4j::math::nd4j_sqrt<int, T>(lengths[segment]));
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------------------------- //
    // Sorted segments ops implementations
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void segmentMaxFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        //int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        //Nd4jLong idx = indices->e<Nd4jLong>(0);
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(256, 512, 256);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numClasses, begins, lengths);

        if (input->isVector()) {
            segmentMaxLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            segmentMaxTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }
    }

    // segmen min 
    template <typename T, typename I>
    static void segmentMinFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numClasses, begins, lengths);

        if (input->isVector()) {
            segmentMinLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            segmentMinTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);

        }
    }

    // segmen mean
    template <typename T, typename I>
    static void segmentMeanFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numClasses, begins, lengths);

        if (input->isVector()) {
            segmentMeanLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            segmentMeanTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }

    template <typename T, typename I>
    static void segmentSumFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numClasses, begins, lengths);

        if (input->isVector()) {
            segmentSumLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            segmentSumTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }

    template <typename T, typename I>
    static void segmentProdFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
        auto stream = context->getCudaStream();
        Nd4jLong numClasses = indices->e<Nd4jLong>(indices->lengthOf() - 1) + 1;
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses});

        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);

        dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numClasses, begins, lengths);

        if (input->isVector()) {
            segmentProdLinearKernel<T,I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            segmentProdTadKernel<T,I><<<input->sizeAt(0), 512, 2048, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }

    template <typename T, typename I>
    static bool segmentIndicesValidate_(NDArray* indices, NDArray& aexpected, NDArray& aoutput) {
        return true;
    }

    void segmentMaxFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMaxFunctor_, (context, input, indices, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    void segmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMinFunctor_, (context, input, indices, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    void segmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentMeanFunctor_, (context, input, indices, output), FLOAT_TYPES, INTEGER_TYPES);
    }

    void segmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentSumFunctor_, (context, input, indices, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    void segmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentProdFunctor_, (context, input, indices, output), FLOAT_TYPES, INTEGER_TYPES);
    }

    bool segmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, NDArray& expected, NDArray& output) {
        BUILD_DOUBLE_SELECTOR(output.dataType(), indices->dataType(), return segmentIndicesValidate_, (indices, expected, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template bool segmentIndicesValidate_, (NDArray*, NDArray&, NDArray&), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentProdFunctor_, (nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentSumFunctor_, (nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentMeanFunctor_, (nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentMinFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentMaxFunctor_, (LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);

    // -------------------------------------------------------------------------------------------------------------- //

    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted segment ops functors implementation
    // -------------------------------------------------------------------------------------------------------------- //

    bool unsortedSegmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, Nd4jLong expected, Nd4jLong& output) {
        return true;
    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void unsortedSegmentMaxFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numOfClasses, begins, lengths);
        classesRangesBegs.syncToHost();
        classesRangesLens.syncToHost();

        if (input->isVector()) {
            unsortedSegmentMaxLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            output->assign(-DataTypeUtils::max<T>());
            segmentMaxTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void unsortedSegmentMinFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numOfClasses, begins, lengths);

        if (input->isVector()) {
            unsortedSegmentMinLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(DataTypeUtils::max<T>());
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentMinTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void unsortedSegmentMeanFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numOfClasses, begins, lengths);

        if (input->isVector()) {
            unsortedSegmentMeanLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(0);
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentMeanTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void unsortedSegmentSumFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), (numOfClasses + 1) * 64);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numOfClasses, begins, lengths);

        if (input->isVector()) {
            unsortedSegmentSumLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(0);
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentSumTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void unsortedSegmentProdFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numOfClasses, begins, lengths);

        if (input->isVector()) {
            unsortedSegmentProdLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(1);
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentProdTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }

    }
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T, typename I>
    static void unsortedSegmentSqrtNFunctor_(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        auto stream = context->getCudaStream();
//        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
        NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses});
        NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses});
//        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
//        classes.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Assign(), &row, &classes);
        classesRangesBegs.assign(indices->lengthOf());
        classesRangesLens.assign(0);
        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
//        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
        int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
        int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
        fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(), numOfClasses, begins, lengths);

        if (input->isVector()) {
            unsortedSegmentSqrtNLinearKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        }
        else {
            output->assign(0);
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimensions);
            Nd4jLong* inputTads = packX.specialShapeInfo();
            Nd4jLong* inputTadOffsets = packX.specialOffsets();
            Nd4jLong* outputTads = packZ.specialShapeInfo();
            Nd4jLong* outputTadOffsets = packZ.specialOffsets();
            dims.x = input->sizeAt(0);
            segmentSqrtNTadKernel<T,I><<<dims.x, dims.y, dims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets, reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo(), outputTads, outputTadOffsets);
        }
    }
    // -------------------------------------------------------------------------------------------------------------- //
    // -------------------------------------------------------------------------------------------------------------- //
    // unsorted ops functors
    // -------------------------------------------------------------------------------------------------------------- //

    void unsortedSegmentMaxFunctor(nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMaxFunctor_, (context, input, indices, numOfClasses, output), NUMERIC_TYPES, INTEGER_TYPES);
    }
    // -------------------------------------------------------------------------------------------------------------- //

    void unsortedSegmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMinFunctor_, (context, input, indices, numOfClasses, output),
                              NUMERIC_TYPES, INTEGER_TYPES);
    }
    // -------------------------------------------------------------------------------------------------------------- //

    void unsortedSegmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMeanFunctor_, (context, input, indices, numOfClasses, output),
                              FLOAT_TYPES, INTEGER_TYPES);
    }
    // -------------------------------------------------------------------------------------------------------------- //

    void unsortedSegmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSumFunctor_, (context, input, indices, numOfClasses, output),
                              NUMERIC_TYPES, INTEGER_TYPES);

    }
    // -------------------------------------------------------------------------------------------------------------- //

    void unsortedSegmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentProdFunctor_, (context, input, indices, numOfClasses, output),
                              FLOAT_TYPES, INTEGER_TYPES);
    }
    // -------------------------------------------------------------------------------------------------------------- //

    void unsortedSegmentSqrtNFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSqrtNFunctor_, (context, input, indices, numOfClasses, output),
                              FLOAT_TYPES, INTEGER_TYPES);
    }
    // -------------------------------------------------------------------------------------------------------------- //

    BUILD_DOUBLE_TEMPLATE(template void unsortedSegmentMaxFunctor_, (nd4j::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void unsortedSegmentMinFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void unsortedSegmentMeanFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void unsortedSegmentSumFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void unsortedSegmentProdFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void unsortedSegmentSqrtNFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), FLOAT_TYPES, INTEGER_TYPES);
    // -------------------------------------------------------------------------------------------------------------- //
    // -------------------------------------------------------------------------------------------------------------- //
    // Backpropagate ops helpers
    // -------------------------------------------------------------------------------------------------------------- //
    // Sorted backpropagate ops
    //

    // segment max
    template <typename T>
    int segmentMaxFunctorBP_(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {

        return Status::OK();
    }

    int segmentMaxFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), return segmentMaxFunctorBP_, (input, indices, gradOut, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int segmentMaxFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output), NUMERIC_TYPES);

    // segmen min
    int segmentMinFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    // segmen mean
    int segmentMeanFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    int segmentSumFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted backpropagate segment ops
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T>
    static int unsortedSegmentMaxFunctorBP_(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentMaxFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMaxFunctorBP_, (input, indices, gradOut, numOfClasses, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int unsortedSegmentMaxFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    template <typename T>
    static int unsortedSegmentMinFunctorBP_(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentMinFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMinFunctorBP_, (input, indices, gradOut, numOfClasses, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int unsortedSegmentMinFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    int unsortedSegmentMeanFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentSumFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentProdFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

//    template <typename T>
    int unsortedSegmentSqrtNFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

//    int unsortedSegmentSqrtNFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
//        BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentSqrtNFunctorBP_, (input, indices, gradOut, numOfClasses, output), FLOAT_TYPES);
//    }
//    BUILD_SINGLE_TEMPLATE(template int unsortedSegmentSqrtNFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), FLOAT_TYPES);
}
}
}