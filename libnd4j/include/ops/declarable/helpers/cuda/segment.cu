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

namespace nd4j {
namespace ops {
namespace helpers {

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

            //[zIndex] =
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
             //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
             if (val[segment] < x[xIndex])
                 val[segment] = x[xIndex];

         }
        __syncthreads();
        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
            if (val[segment] < x[xIndex])
                val[segment] = x[xIndex];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            z[zIndex] = val[segment];
        }

    }

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

            //[zIndex] =
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
            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
            if (val[segment] > x[xIndex])
                val[segment] = x[xIndex];

        }
        __syncthreads();
        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
            if (val[segment] > x[xIndex])
                val[segment] = x[xIndex];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            z[zIndex] = val[segment];
        }

    }
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
            extern __shared__ unsigned char shmem[];
            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            //[zIndex] =
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
            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
            nd4j::math::atomics::nd4j_atomicAdd(&val[segment], x[xIndex]);
            //atomicAdd(&val[segment], x[xIndex]);

        }
//        __syncthreads();
//        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
//            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
//            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
//            val[segment] += x[xIndex];
//        }
        __syncthreads();

        if (threadIdx.x == 0) {
            z[zIndex] = val[segment];
        }

    }
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
            extern __shared__ unsigned char shmem[];
            val = reinterpret_cast<T*>(shmem);
            xLen = shape::length(inputShape);
            zLen = shape::length(outputShape);

            //[zIndex] =
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
            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
            nd4j::math::atomics::nd4j_atomicAdd(&val[segment], x[xIndex]);
        }
//        __syncthreads();
//        for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
//            auto xIndex = shape::getIndexOffset(e, inputShape, xLen);
//            //val[segment] = nd4j::math::nd4j_max<T>(x[xIndex], val[segment]);
//            val[segment] += x[xIndex];
//        }
        __syncthreads();

        if (threadIdx.x == 0) {
            z[zIndex] = val[segment] / lengths[segment];
        }

    }

    template <typename I>
    static __global__ void fillUpSegmentsKernel(void* indices, Nd4jLong* indexShape, int numClasses, int* classesRangesStart, int* classesRangesLenghts) {
        __shared__ I* idxBuf;
        __shared__ Nd4jLong idxLen;
        __shared__ int* result;
        if (threadIdx.x == 0) {
            idxBuf = reinterpret_cast<I*>(indices);
            idxLen = shape::length(indexShape);
            extern __shared__ unsigned char shmem[];
            result = reinterpret_cast<int*>(shmem);
            result[0] = 0; //idxBuf[0];
        }
        __syncthreads();

        auto tid = threadIdx.x + blockDim.x * blockIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (auto j = tid; j < idxLen; j += step) {
            auto pos = idxBuf[j];
             if (classesRangesStart[pos] == idxLen)
                 classesRangesStart[pos] = j;
            result[pos] = nd4j::math::nd4j_min<int>(classesRangesStart[pos], j);

            classesRangesStart[pos] = nd4j::math::nd4j_min<int>(classesRangesStart[pos], result[pos]);
            nd4j::math::atomics::nd4j_atomicAdd(&classesRangesLenghts[pos], 1);
        }
    }
    // segment max
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
//            std::vector<int> restDims(input->rankOf() - 1);
//            Nd4jLong loop_size = input->rankOf();
//            PRAGMA_OMP_PARALLEL_FOR
//            for (Nd4jLong e = 1; e < loop_size; e++)
//                restDims[e - 1] = e;
//
//            auto listOfTensors = input->allTensorsAlongDimension(restDims);
//            auto listOfOutTensors = output->allTensorsAlongDimension(restDims);
//
//            auto numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
//            auto maxT = listOfOutTensors->at(idx);
//
//            //int pos = 0;
//            maxT->assign(listOfTensors->at(0));
//
//            for (Nd4jLong i = 1; i < indices->lengthOf(); i++) {
//                if (indices->e<int>(i) == idx) {
//
//                    for (Nd4jLong e = 0; e < maxT->lengthOf(); e++) {
//                        maxT->t<T>(e) = nd4j::math::nd4j_max(maxT->t<T>(e), listOfTensors->at(i)->t<T>(e));
//                    }
//                }
//                else {
//                    idx = indices->e<Nd4jLong>(i);
//                    maxT = listOfOutTensors->at(idx);
//                    maxT->assign(listOfTensors->at(i));
//                }
//
//            }
//            delete listOfTensors;
//            delete listOfOutTensors;
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

        }

    }

    template <typename T>
    static void segmentProdFunctor_(NDArray* input, NDArray* indices, NDArray* output) {

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
        BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentMeanFunctor_, (context, input, indices, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    void segmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentSumFunctor_, (context, input, indices, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    void segmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentProdFunctor_, (input, indices, output), NUMERIC_TYPES);
    }

    bool segmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, NDArray& expected, NDArray& output) {
        BUILD_DOUBLE_SELECTOR(output.dataType(), indices->dataType(), return segmentIndicesValidate_, (indices, expected, output), NUMERIC_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template bool segmentIndicesValidate_, (NDArray*, NDArray&, NDArray&), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentProdFunctor_, (NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentSumFunctor_, (nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentMeanFunctor_, (nd4j::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentMinFunctor_, (nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void segmentMaxFunctor_, (LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output), NUMERIC_TYPES, INTEGER_TYPES);
    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted segment ops
    // -------------------------------------------------------------------------------------------------------------- //

    bool unsortedSegmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, Nd4jLong expected, Nd4jLong& output) {
        return true;
    }

    template <typename T>
    static void unsortedSegmentMaxFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentMaxFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMaxFunctor_, (input, indices, numOfClasses, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void unsortedSegmentMaxFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    template <typename T>
    static void unsortedSegmentMinFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMinFunctor_, (input, indices, numOfClasses, output),
                              NUMERIC_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void unsortedSegmentMinFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    void unsortedSegmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
      //  BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentProdFunctor_, (input, indices, numOfClasses, output), NUMERIC_TYPES);
    }
    //BUILD_SINGLE_TEMPLATE(template void unsortedSegmentProdFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    void unsortedSegmentSqrtNFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

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