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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void concatCuda(const int numOfArrs, void* pVx,  void* pxShapeInfo, void* pVz, void* pzShapeInfo) {
    
    __shared__ int arrIdx, blocksPerArr;
    __shared__ T *x, *z;
    __shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLen, arrLenPerBlock, start, end;
    
    if (threadIdx.x == 0) {
            
        blocksPerArr = (gridDim.x + numOfArrs - 1) / numOfArrs;     // ceil 
        arrIdx = blockIdx.x / blocksPerArr;
    
        x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[arrIdx]);
        z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[arrIdx]);        
        xShapeInfo = reinterpret_cast<Nd4jLong**>(pxShapeInfo)[arrIdx];
        zShapeInfo = reinterpret_cast<Nd4jLong**>(pzShapeInfo)[arrIdx];
        arrLen = shape::length(xShapeInfo);

        arrLenPerBlock = (arrLen + blocksPerArr - 1) / blocksPerArr;  // ceil

        start = (blockIdx.x % blocksPerArr) * arrLenPerBlock;
        end   = (start + arrLenPerBlock) > arrLen ? arrLen : (start + arrLenPerBlock);        
    }

    __syncthreads();    

    for (Nd4jLong i = start + threadIdx.x; i < end; i += blockDim.x)
        z[shape::getIndexOffset(i, zShapeInfo, arrLen)] = x[shape::getIndexOffset(i, xShapeInfo, arrLen)];
}

template<typename T>
__host__ static void concatCudaLauncher(const int numOfArrs, const cudaStream_t *stream,  void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

    concatCuda<T><<<512, 256, 1024, *stream>>>(numOfArrs, pVx, pxShapeInfo, pVz, pzShapeInfo);
}

    //////////////////////////////////////////////////////////////////////////
    void triu(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int diagonal) {

    }


    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void triuBP_(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    }

    void triuBP(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {
        BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (context, input, gradO, gradI, diagonal), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void triuBP_, (graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void trace_(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

    }

    void trace(graph::LaunchContext* context, const NDArray& input, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), trace_, (context, input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void trace_, (graph::LaunchContext* context, const NDArray& input, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void randomShuffle_(graph::LaunchContext* context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

    }

    void randomShuffle(graph::LaunchContext* context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (context, input, output, rng, isInplace), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void randomShuffle_, (graph::LaunchContext* context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    // Pad kernels
    //
    static __global__ void padFillIndicesKernel(Nd4jLong* outIndices, void* paddingBuffer, Nd4jLong* paddingShape, Nd4jLong* inputShape, bool shortType, int rankBorder) {
        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
//        if (threadIdx.x == 0) {
//            outIndices[2 * rankBorder + 1] = shortType?reinterpret_cast<int*>(paddingBuffer)[rankBorder]:reinterpret_cast<Nd4jLong*>(paddingBuffer)[pos];
//            outIndices[2 * rankBorder] = rightBorder;
//        }
        __syncthreads();

        for(int i = tid; i < rankBorder + 1; i += step) {
            Nd4jLong coords[2] = {i, 0};
            auto pos = shape::getOffset(0, shape::shapeOf(paddingShape), shape::stride(paddingShape), coords, rankBorder + 1);
            outIndices[2 * i] = shortType?reinterpret_cast<int*>(paddingBuffer)[pos]:reinterpret_cast<Nd4jLong*>(paddingBuffer)[pos];
            outIndices[2 * i + 1] = outIndices[2 * i] + shape::sizeAt(inputShape, i);
        }
    }

    template <typename T>
    static __global__ void padFillValues(void* outputBuffer, Nd4jLong* outputShape, Nd4jLong* paddingBound,
            void* inputBuffer, Nd4jLong* inputShape,
            Nd4jLong* inputTadShape, Nd4jLong* inputTadOffsets, Nd4jLong* outputTadShape, Nd4jLong* outputTadOffsets, const int mode, void* value) {

            __shared__ T* z;
            __shared__ T* x;
            __shared__ T* val;
            __shared__ Nd4jLong inputLen;
            __shared__ Nd4jLong outputLen;
            __shared__ Nd4jLong rank;
            __shared__ Nd4jLong lastInDimSize;
            __shared__ Nd4jLong outTadCount;
            __shared__ Nd4jLong inTadCount;
            if (threadIdx.x == 0) {
                z = reinterpret_cast<T*>(outputBuffer);
                x = reinterpret_cast<T*>(inputBuffer);
                inputLen = shape::length(inputShape);
                outputLen = shape::length(outputShape);
                if (value && mode == 0) // only for CONSTANT mode
                    val = reinterpret_cast<T*>(value);
                else
                    val = nullptr;
                rank = shape::rank(outputShape);
                Nd4jLong lastInDimSize  = shape::sizeAt(inputShape, rank - 1);

                outTadCount = outputLen / shape::length(outputTadShape);
                inTadCount = inputLen / shape::length(inputTadShape);
                printf("%lld, %lld\n", inTadCount, outTadCount);
            }
            __syncthreads();

            const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
            const auto step = gridDim.x * blockDim.x;
            const auto stepY = gridDim.y * blockDim.y;
            Nd4jLong k = rank - 1;
            //for (Nd4jLong k = blockIdx.y * gridDim.y + threadIdx.y; k < rank; k += stepY) {
                for (Nd4jLong i = tid; i < outputLen; i += step) {
                    if (i >= paddingBound[2 * k] && i < paddingBound[2 * k + 1]) {
                        if (k == rank - 1)
                            z[i] = x[i - paddingBound[2 * k]];
                    }
                    else if (val)
                        z[i] = val[0];
                    else if (mode != 0){
                        Nd4jLong startL = mode == 1 ? 1 : 0;                            // REFLECT or SYMMETRIC
                        Nd4jLong startR = mode == 1 ? lastInDimSize - 2 : lastInDimSize - 1;        // REFLECT or SYMMETRIC
                        if (i < paddingBound[2 * k]) {
                            z[i] = x[paddingBound[2 * k] - i - startL];
                        }
                        else {
                            z[i] = x[i - startR - paddingBound[2 * k + 1] + paddingBound[2 * k] + 1];
                        }
                    }
                }
            //}
    }

    template<typename T>
    void pad_(graph::LaunchContext* context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {
        const int rank = output.rankOf();
        const int rankBorder = rank - 1;
        std::vector<int> dimsToExclude({rankBorder});
        std::iota(dimsToExclude.begin(), dimsToExclude.end(), 0);             // fill with 0, 1, ... rank-1
        //dimsToExclude.pop_back();

//        Nd4jLong numLeft    = paddings.e<Nd4jLong>(rankBorder, 0);
//        Nd4jLong numRight   = paddings.e<Nd4jLong>(rankBorder, 1);
        Nd4jLong inDimSize  = input.sizeAt(rankBorder);
        Nd4jLong outDimSize = output.sizeAt(rankBorder);
        Nd4jLong* outIdx = nullptr;
        cudaError_t err = cudaMalloc(&outIdx, 2 * rank * sizeof(Nd4jLong));
        if (0 != err) {
            throw cuda_exception::build("Cannot allocate memory for pad indices", err);
        }
        err = cudaMemset(outIdx, 0, 2 * rank * sizeof(Nd4jLong));
        if (0 != err) {
            throw cuda_exception::build("Cannot initialize memory for pad indices", err);
        }

       // dim3 launcDim(16, 32, 512);
        auto stream = context->getCudaStream();
        bool shortedType = (paddings.dataType() == DataType::INT32);
        padFillIndicesKernel<<<16, 32, 512, *stream>>>(outIdx, paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), input.getSpecialShapeInfo(), shortedType, rankBorder);

        Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);

        //NDArray outSubArr0 = output(outIdx, true);
        dim3 launchDim(128, 256, 2048);
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimsToExclude);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), dimsToExclude);

        padFillValues<T><<<launchDim.x, launchDim.y, launchDim.z, *stream>>>(output.specialBuffer(),
                output.specialShapeInfo(), outIdx, input.getSpecialBuffer(), input.getSpecialShapeInfo(),
                packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(), mode, padValue.getSpecialBuffer());

//        err = cudaFree(outIdx);
//        if (0 != err) {
//            throw cuda_exception::build("Cannot release memory for pad indices", err);
//        }

//#pragma omp parallel for schedule(guided)
//        for(Nd4jLong j = 0; j < numOfSubArrs; ++j) {
//
//            NDArray outSubArr1   = outSubArr0(j, dimsToExclude);
//            NDArray inSubArr     = input(j, dimsToExclude);
//            NDArray outSubArrMid = outSubArr1(outIdx[1]);
//
//            outSubArrMid.assign(inSubArr);      // assign middle
//
//            if(mode == 0)  { // CONSTANT
//                if(numLeft != 0) {
//                    NDArray temp = outSubArr1(outIdx[2]);
//                    temp.assign(padValue);                        // assign left
//                }
//                if(numRight != 0) {
//                    NDArray temp = outSubArr1(outIdx[3]);
//                    temp.assign(padValue);                        // assign right
//                }
//            }
//            else {                                                              // REFLECT or SYMMETRIC
//
//#pragma omp parallel for schedule(guided)
//                for(Nd4jLong k = numLeft-1, e = startL; k >= 0; --k, ++e)     // fill left side
//                    outSubArr1.t<T>(k) = inSubArr.t<T>(e);
//
//#pragma omp parallel for schedule(guided)
//                for(Nd4jLong k = numLeft + inDimSize, e = startR; k < outDimSize; ++k, --e)     // fill right side
//                    outSubArr1.t<T>(k) = inSubArr.t<T>(e);
//            }
//        }
    }

    void pad(graph::LaunchContext* context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {
        BUILD_SINGLE_SELECTOR(input.dataType(), pad_, (context, mode, input, paddings, output, padValue), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void pad_, (graph::LaunchContext* context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    void invertPermutation(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void gatherND_(graph::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output) {

    }

    void gatherND(graph::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), gatherND_, (context, input, indices, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void gatherND_, (graph::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output), LIBND4J_TYPES);



    //////////////////////////////////////////////////////////////////////////
    void eye(graph::LaunchContext* context, NDArray& output) {

    }

    //////////////////////////////////////////////////////////////////////////
    void scatterUpdate(graph::LaunchContext* context, NDArray& operand, NDArray& updates, const std::vector<int>* intArgs) {

    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T, typename Z>
    static __global__ void global_mergeMaxIndex_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<Z*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T mVal = -DataTypeUtils::max<T>();
            Z mIdx(0);

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);
                auto val = x[shape::getIndexOffset(e, xShape, length)];;
                if (mVal < val)
                    mIdx = static_cast<Z>(e);
            }
            __syncthreads();

            output[shape::getIndexOffset(e, outputShape, length)] = mIdx;
        }
    }

    template <typename T, typename Z>
    static void mergeMaxIndex_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeMaxIndex");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeMaxIndex_<T,Z><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }

    void mergeMaxIndex(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (context, inArrs, output), LIBND4J_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void mergeMaxIndex_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES, INTEGER_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void global_mergeMax_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<T*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T mVal = -DataTypeUtils::max<T>();

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);
                auto val = x[shape::getIndexOffset(e, xShape, length)];;
                if (mVal < val)
                    mVal = val;
            }
            __syncthreads();

            output[shape::getIndexOffset(e, outputShape, length)] = mVal;
        }
    }

    template<typename T>
    static void mergeMax_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeMax");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeMax_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void mergeMax_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeMax(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (context, inArrs, output), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void global_mergeAvg_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<T*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T sum(0.0f);

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);

                sum += x[shape::getIndexOffset(e, xShape, length)];
            }

            output[shape::getIndexOffset(e, outputShape, length)] = sum / numArrays;
        }
    }

    template<typename T>
    static void mergeAvg_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeAvg");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeAvg_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void mergeAvg_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeAvg(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (context, inArrs, output), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void global_mergeAdd_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<T*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T sum(0.0f);

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);

                sum += x[shape::getIndexOffset(e, xShape, length)];
            }

            output[shape::getIndexOffset(e, outputShape, length)] = sum;
        }
    }

    template<typename T>
    static void mergeAdd_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeAdd");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeAdd_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeAdd(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNorm_(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {

    }

    void clipByNorm(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNorm_, (graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

    template <typename T>
    static void clipByGlobalNorm_(graph::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {

    }

    void clipByGlobalNorm(graph::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (context, inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (graph::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNormBP_(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {

    }

    void clipByNormBP(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBP_, (context, input, gradO, gradI, dimensions, clipNorm), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNormBP_, (graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm), FLOAT_TYPES);


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByAveraged_(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        auto cn = clipNorm.e<T>(0);
        if (dimensions.size() == 0) {
            // all-reduce
            T n2 = input.reduceNumber(reduce::Norm2).e<T>(0) / input.lengthOf();
            if (n2 <= cn) {
                if (!isInplace)
                    output.assign(input);
            }
            else {
                const T factor = cn / n2;
                //auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
                //input.applyLambda<T>(lambda, &output);
                output.assign(input * factor);
            }
        }
        else {
            // along dimension
            auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions, false);
            if (!isInplace)
                output.assign(input);
            auto tads = output.allTensorsAlongDimension(dimensions);
            auto outTads = output.allTensorsAlongDimension(dimensions);
            // TODO: make this CUDA-compliant somehow
            for (int e = 0; e < tads->size(); e++) {
                T n2 = norm2.e<T>(e) / tads->at(e)->lengthOf();
                const T factor = cn / n2;
                if (n2 > cn) {
                    //auto lambda = LAMBDA_T(_x, factor) {return _x * factor;};
                    tads->at(e)->applyScalar(scalar::Multiply, factor, outTads->at(e));//applyLambda<T>(lambda, &output);
                }
            }
            delete tads;
            delete outTads;
        }
    }

    void clipByAveraged(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByAveraged_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByAveraged_, (graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

/*
    if (d1 > params[1])
    return params[1];
    else if (d1 < params[0])
    return params[0];
    else return d1;
*/
    template <typename T>
    static void __global__ clipByValueKernel(void* input, Nd4jLong* inputShape, void* output, Nd4jLong* outputShape, double leftBound, double rightBound) {
        __shared__ T* outputBuf;
        __shared__ T* inputBuf;
        __shared__ Nd4jLong length;
        __shared__ bool linearBuffers;
        if (threadIdx.x == 0) {
            outputBuf = reinterpret_cast<T *>(output);
            inputBuf = reinterpret_cast<T *>(input);
            length = shape::length(inputShape);
            linearBuffers = shape::elementWiseStride(inputShape) == shape::elementWiseStride(outputShape) && shape::elementWiseStride(inputShape) == 1;
        }
        __syncthreads();
        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            if (linearBuffers) {
                if (inputBuf[e] > rightBound) outputBuf[e] = (T) rightBound;
                else if (inputBuf[e] < leftBound) outputBuf[e] = (T) leftBound;
                else outputBuf[e] = inputBuf[e];
            }
            else {
                auto inputOffset = shape::getIndexOffset(e, inputShape, length);
                auto outputOffset = shape::getIndexOffset(e, outputShape, length);
                if (inputBuf[inputOffset] > rightBound) outputBuf[outputOffset] = (T) rightBound;
                else if (inputBuf[inputOffset] < leftBound) outputBuf[outputOffset] = (T) leftBound;
                else outputBuf[outputOffset] = inputBuf[outputOffset];
            }
        }
    }

    template <typename T>
    static void clipByValue_(graph::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        auto stream = context->getCudaStream();
        if (!input.isActualOnDeviceSide())
            input.syncToDevice();
        NDArray::prepareSpecialUse({&output}, {&input});
        clipByValueKernel<T><<<256, 512, 8192, *stream>>>(input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftBound, rightBound);
        NDArray::registerSpecialUse({&output}, {&input});
    }

    void clipByValue(graph::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (context, input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (graph::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void mirrorPad_(graph::LaunchContext* context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {

    }

    void mirrorPad(graph::LaunchContext* context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_SINGLE_SELECTOR(input.dataType(), mirrorPad_, (context, input, paddings, output, mode), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mirrorPad_, (graph::LaunchContext* context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void concat(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

    const int numOfArrs = inArrs.size();
    for(int i = 0; i < numOfArrs; ++i)
        if(!inArrs[i]->isActualOnDeviceSide()) inArrs[i]->syncToDevice();

    const int rank  = inArrs[0]->rankOf();
    const int rank2 = 2*rank;
    std::vector<std::vector<Nd4jLong>> indices(numOfArrs, std::vector<Nd4jLong>(rank2,0));

    // take into account indices for first array
    indices[0][2 * axis + 1] = inArrs[0]->sizeAt(axis);

    // loop through the rest of input arrays
    for(int i = 1; i < numOfArrs; ++i) {
        indices[i][2 * axis]     = indices[i-1][2 * axis + 1];                                // index start from
        indices[i][2 * axis + 1] = indices[i-1][2 * axis + 1] + inArrs[i]->sizeAt(axis);      // index end with (excluding)
    }

    std::vector<NDArray*> outSubArrs(numOfArrs);
    for(int i = 0; i < numOfArrs; ++i) 
        outSubArrs[i] = new NDArray(output(indices[i], true));

    // prepare arrays of pointers on buffers and shapes
    std::vector<void*>     hOutBuffers(numOfArrs), hInBuffers(numOfArrs);
    std::vector<Nd4jLong*> hOutShapeInfo(numOfArrs), hInShapeInfo(numOfArrs);
    for(int i = 0; i < numOfArrs; ++i) {        
        hOutBuffers[i]   = outSubArrs[i]->getSpecialBuffer();
        hInBuffers[i]    =     inArrs[i]->getSpecialBuffer();
        hOutShapeInfo[i] = outSubArrs[i]->getSpecialShapeInfo();
        hInShapeInfo[i]  =     inArrs[i]->getSpecialShapeInfo();
    }

    // allocate and copy all buffers and shapes arrays to global memory    
    PointersManager manager(context, "helpers::concat");
    void* dOutBuffers	= manager.replicatePointer(hOutBuffers.data(),   hOutBuffers.size() * sizeof(void*));
    void* dInBuffers	= manager.replicatePointer(hInBuffers.data(),    hInBuffers.size() * sizeof(void*));
    void* dInShapeInfo  = manager.replicatePointer(hInShapeInfo.data(),  hInShapeInfo.size() * sizeof(Nd4jLong*));
    void* dOutShapeInfo = manager.replicatePointer(hOutShapeInfo.data(), hOutShapeInfo.size() * sizeof(Nd4jLong*));

    BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher, (numOfArrs, context->getCudaStream(), dInBuffers, dInShapeInfo, dOutBuffers, dOutShapeInfo), LIBND4J_TYPES);

    manager.synchronize();
    
    for(int i = 0; i < numOfArrs; ++i)
        delete outSubArrs[i];

    for(int i = 0; i < numOfArrs; ++i)
        inArrs[i]->tickReadHost();

    output.tickWriteDevice();
}




    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void tileBP_(graph::LaunchContext* context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    }

    void tileBP(graph::LaunchContext* context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (context, gradO, gradI, reps), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void tileBP_, (graph::LaunchContext* context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps), FLOAT_TYPES);

    void scatterSimple(const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {

    }


BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher, (const int numOfArrs, const cudaStream_t *stream, void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo), LIBND4J_TYPES);


}
}
}
