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

///////////////////////////////////////////////////////////////////
// x - input, y - paddings, z - output
template<typename X, typename Y>
__global__ static void padCuda(const int mode,
                               const void *vx, const Nd4jLong *xShapeInfo,
                               const void *vy, const Nd4jLong *yShapeInfo,
                                     void *vz, const Nd4jLong *zShapeInfo,
                               const void *vPadVal) {

    const X padVal = *reinterpret_cast<const X*>(vPadVal);

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<X*>(vz);

    __shared__ int rank, rankMinusOne;
    __shared__ Nd4jLong zLen, yLen, totalThreads, *coord, *xShape, *zShape, *xStride, *zStride, shift1, shift2, yStride0;
    
    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        coord    = reinterpret_cast<Nd4jLong*>(shmem);
        zLen     = shape::length(zShapeInfo);
        xShape   = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
        zShape   = shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo));
        xStride  = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));
        zStride  = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));
        yStride0 = shape::stride(const_cast<Nd4jLong*>(yShapeInfo))[0];
        rank     = shape::rank(xShapeInfo);
        zLen     = shape::length(zShapeInfo);
        yLen     = 2 * rank;
        rankMinusOne = rank - 1;
        totalThreads = gridDim.x * blockDim.x;
        shift1 = mode == 1 ? 0 : 1;         // REFLECT : SYMMETRIC
        shift2 = mode == 1 ? 2 : 1;         // REFLECT : SYMMETRIC        
    }

    __syncthreads();

    auto xzCoord = coord + threadIdx.x * rank;       // we use xzCoord storage both for x and z arrays    

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(mode == 0) { // CONSTANT case
        
        for (Nd4jLong i = tid; i < zLen; i += totalThreads) {
        
            shape::index2coords(rank, zShape, i, zLen, xzCoord);            
            const auto zOffset = shape::getOffset(0, zShape, zStride, xzCoord, rank);
    
            bool within = true;
            for(int j = rankMinusOne; j >= 0; --j) {
                if(xShape[j] == zShape[j]) continue;
                const auto left = y[shape::getIndexOffset(yStride0 * j, yShapeInfo, yLen)];
                if(xzCoord[j] < left || xzCoord[j] >= left + xShape[j]) {within = false; break;}
                else                                                    {xzCoord[j] = xzCoord[j] - left;}
            }                            

            if(within)
                z[zOffset] = x[shape::getOffset(0, xShape, xStride, xzCoord, rank)];
            else 
                z[zOffset] = padVal;
        }
    }
    else {  // REFLECT and SYMMETRIC cases
                        
        for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

            shape::index2coords(rank, zShape, i, zLen, xzCoord);
            const auto zOffset = shape::getOffset(0, zShape, zStride, xzCoord, rank);

            for(int j = rankMinusOne; j >= 0; --j) {

                if(xShape[j] == zShape[j]) continue;
                xzCoord[j] = xzCoord[j] - y[shape::getIndexOffset(yStride0 * j, yShapeInfo, yLen)];    // are ready to fill middle (within input dimension range)
                if(xzCoord[j] < 0)               xzCoord[j] = -xzCoord[j] - shift1;                // means fill from left                    
                else if(xzCoord[j] >= xShape[j]) xzCoord[j] = 2 * xShape[j] - xzCoord[j] - shift2; // means fill from right
            }
    
            const auto xOffset = shape::getOffset(0, xShape, xStride, xzCoord, rank);
            z[zOffset] = x[xOffset];
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void padCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, 
                                const int mode,
                                const void *vx, const Nd4jLong *xShapeInfo, 
                                const void *vy, const Nd4jLong *yShapeInfo, 
                                      void *vz, const Nd4jLong *zShapeInfo,
                                const void* padVal) {
        
    padCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(mode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, padVal);
}

///////////////////////////////////////////////////////////////////
void pad(nd4j::LaunchContext * context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, const NDArray& padValue) {

    PointersManager manager(context, "pad");

    NDArray::prepareSpecialUse({&output}, {&input, &paddings, &padValue});

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = 8 * threadsPerBlock * output.rankOf() + 128;

    const auto xType = input.dataType();
    const auto yType = paddings.dataType();

    BUILD_DOUBLE_SELECTOR(xType, yType, padCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), mode, input.getSpecialBuffer(), input.getSpecialShapeInfo(), paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), padValue.getSpecialBuffer()), LIBND4J_TYPES, INTEGER_TYPES);

    NDArray::registerSpecialUse({&output}, {&input, &paddings, &padValue});    
    manager.synchronize();      
}




    //////////////////////////////////////////////////////////////////////////
    void triu(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int diagonal) {

    }


    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void triuBP_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    }

    void triuBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {
        BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (context, input, gradO, gradI, diagonal), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void triuBP_, (nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void trace_(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {

    }

    void trace(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), trace_, (context, input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void trace_, (nd4j::LaunchContext * context, const NDArray& input, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void randomShuffle_(nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

    }

    void randomShuffle(nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (context, input, output, rng, isInplace), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void randomShuffle_, (nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    void invertPermutation(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {

    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void gatherND_(nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output) {

    }

    void gatherND(nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), gatherND_, (context, input, indices, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void gatherND_, (nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output), LIBND4J_TYPES);



    //////////////////////////////////////////////////////////////////////////
    void eye(nd4j::LaunchContext * context, NDArray& output) {

    }

    //////////////////////////////////////////////////////////////////////////
    void scatterUpdate(nd4j::LaunchContext * context, NDArray& operand, NDArray& updates, const std::vector<int>* intArgs) {

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
    static void mergeMaxIndex_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
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

    void mergeMaxIndex(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (context, inArrs, output), LIBND4J_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void mergeMaxIndex_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES, INTEGER_TYPES);

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
    static void mergeMax_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
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
    BUILD_SINGLE_TEMPLATE(template void mergeMax_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeMax(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
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
    static void mergeAvg_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
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
    BUILD_SINGLE_TEMPLATE(template void mergeAvg_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeAvg(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
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
    static void mergeAdd_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
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
    BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeAdd(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), LIBND4J_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void clipByNormInplaceKernel(Nd4jLong numOfSubArrs, T* inputBuffer, Nd4jLong* shape, Nd4jLong* inputOffsets, T* norm2Buf, Nd4jLong* norm2shape, T clipNorm) {
        for (int arr = blockIdx.x; arr < numOfSubArrs; arr += gridDim.x) {
            __shared__ T* z;
            __shared__ Nd4jLong len;
            if (threadIdx.x == 0) {
                len = shape::length(shape);
                z = inputBuffer + inputOffsets[arr];
            }
            __syncthreads();
            for (int j = threadIdx.x; j < len; j+= blockDim.x) {
                auto xIndex = shape::getIndexOffset(j, shape, len);

                if(norm2Buf[arr] > clipNorm)
                z[xIndex] *= clipNorm / norm2Buf[arr]; // case with ews = 1 and ordering is 'c'
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void clipByNormKernel(Nd4jLong numOfSubArrs, T* inputBuffer, Nd4jLong* shape, Nd4jLong* inputOffsets, T* outputBuffer, Nd4jLong* outputShape, Nd4jLong* outputOffsets, T* norm2Buf, Nd4jLong* norm2shape, T clipNorm) {
        for (Nd4jLong arr = blockIdx.x; arr < numOfSubArrs; arr += gridDim.x) {
            __shared__ T* x, *z;
            __shared__ Nd4jLong lenX, lenZ;
            __shared__ T norm2;

            if (threadIdx.x == 0) {
                lenX = shape::length(shape);
                x = inputBuffer + inputOffsets[arr];
                z = outputBuffer + outputOffsets[arr];
                lenZ = shape::length(outputShape);
                norm2 = norm2Buf[shape::getIndexOffset(arr, norm2shape, numOfSubArrs)];
                //printf("%d: %lf (vs %lf) %lld %lld\n", arr, norm2, clipNorm, lenX, lenZ);
            }
            __syncthreads();
            for (Nd4jLong j = threadIdx.x; j < lenZ; j+= blockDim.x) {
                auto xIndex = shape::getIndexOffset(j, shape, lenX);
                auto zIndex = shape::getIndexOffset(j, outputShape, lenZ);
                if(norm2 > clipNorm) {
                    z[zIndex] = x[xIndex] * clipNorm / norm2; // case with ews = 1 and ordering is 'c'
                } else {
                    z[zIndex] = x[xIndex];
                }
                //printf("%lld: %lf %lf\n", j, z[zIndex], x[xIndex]);
            }
            __syncthreads();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNorm_(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, NDArray const& clipNormA, const bool isInplace) {
        const int rank = input.rankOf();
        auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions);
        clipNormA.syncToHost();
        //norm2.printBuffer("Norm2");
        T const clipNorm = clipNormA.e<T>(0);
        //clipNormA.printBuffer("ClipNorm");
        auto stream = context->getCudaStream();
        if (isInplace) {
            if(norm2.lengthOf() == 1) {
                norm2.syncToHost();
                T norm2Val = norm2.e<T>(0);
                if(norm2Val > clipNorm)
                    input *= clipNorm / norm2Val;
            }
            else {

                std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
                const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimensions);
                //auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), dimsToExclude);
                T* inputBuffer = reinterpret_cast<T*>(input.specialBuffer());
                T* norm2buf = reinterpret_cast<T*>(norm2.specialBuffer());

                clipByNormInplaceKernel<T><<<256, 512, 1024, *stream>>>(numOfSubArrs, inputBuffer, packX.specialShapeInfo(), packX.specialOffsets(), norm2buf, norm2.specialShapeInfo(), clipNorm);
            }
        }
        else {

            if(norm2.lengthOf() == 1) {
                norm2.syncToHost();
                T norm2Val = norm2.e<T>(0);

                if(norm2Val > clipNorm)
                    output.assign( input * (clipNorm / norm2Val));
                else
                    output.assign( input );
            }
            else {

                std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
                const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimensions);
                auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), dimensions);
                T* inputBuffer = reinterpret_cast<T*>(input.specialBuffer());
                T* norm2buf = reinterpret_cast<T*>(norm2.specialBuffer());
                T* outputBuffer = reinterpret_cast<T*>(output.specialBuffer());

                clipByNormKernel<T><<<256, 512, 1024, *stream>>>(numOfSubArrs, inputBuffer, packX.specialShapeInfo(), packX.specialOffsets(), outputBuffer, packZ.specialShapeInfo(), packZ.specialOffsets(), norm2buf, norm2.specialShapeInfo(), clipNorm);
            }
        }
    }

    void clipByNorm(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNorm_, (nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

    template <typename T>
    static void clipByGlobalNorm_(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {

    }

    void clipByGlobalNorm(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (context, inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNormBP_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {

    }

    void clipByNormBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBP_, (context, input, gradO, gradI, dimensions, clipNorm), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNormBP_, (nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm), FLOAT_TYPES);


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByAveraged_(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
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

    void clipByAveraged(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByAveraged_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByAveraged_, (nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

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
    static void clipByValue_(nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        auto stream = context->getCudaStream();
        if (!input.isActualOnDeviceSide())
            input.syncToDevice();
        NDArray::prepareSpecialUse({&output}, {&input});
        clipByValueKernel<T><<<256, 512, 8192, *stream>>>(input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftBound, rightBound);
        NDArray::registerSpecialUse({&output}, {&input});
    }

    void clipByValue(nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (context, input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void mirrorPadLinearKernel(void const* vx, Nd4jLong* xShape, void* vz, Nd4jLong* zShape, Nd4jLong leftSide, Nd4jLong leftSideCorrected, Nd4jLong xLen, Nd4jLong len, Nd4jLong zLen) {

        __shared__ T const* x;
        __shared__ T* z;
        if (threadIdx.x == 0) {
            x = reinterpret_cast<T const*>(vx);
            z = reinterpret_cast<T*>(vz);
        }
        __syncthreads();
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for(int i = start; i < zLen; i+= step) {
            auto zIndex = shape::getIndexOffset(i, zShape, zLen);
            auto xIndex = shape::getIndexOffset(len - i, xShape, xLen);

            if (i < leftSide)                                   // left side
                xIndex = shape::getIndexOffset(leftSideCorrected - i, xShape, xLen);

            else if(i >= leftSide && i < leftSide + xLen)       // middle
                xIndex = shape::getIndexOffset(i - leftSide, xShape, xLen);

//            else                                                // right side
//                z[i] = x[len - i];
            z[zIndex] = x[xIndex];
        }

    }

    template <typename F, typename I>
    static __global__ void mirrorPadKernel(void const* vx, Nd4jLong* xShape, void* vz, Nd4jLong* zShape, Nd4jLong outLen, void const* paddings, Nd4jLong* paddingShape, int reflBorder) {

        __shared__ F const* x;
        __shared__ I const* pads;
        __shared__ F* z;
        __shared__ Nd4jLong zRank, rank;
        __shared__ Nd4jLong* xShapeOf, *xStrideOf, *padsShapeOf, *padsStrideOf;
        __shared__ Nd4jLong* zShapeOf, *zStrideOf;
        __shared__ Nd4jLong* xIdx;
        if (threadIdx.x == 0) {
            extern __shared__ unsigned char shmem[];
            xIdx    = reinterpret_cast<Nd4jLong*>(shmem);
            rank = shape::rank(xShape);

            x = reinterpret_cast<F const*>(vx);//
            pads = reinterpret_cast<I const*>(paddings);
            z = reinterpret_cast<F*>(vz);
            xShapeOf = shape::shapeOf(xShape);
            xStrideOf = shape::stride(xShape);
            zShapeOf = shape::shapeOf(zShape);
            zRank = shape::rank(zShape);
            zStrideOf = shape::stride(zShape);
            padsShapeOf = shape::shapeOf(paddingShape);
            padsStrideOf = shape::stride(paddingShape);
        }
        __syncthreads();
        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

            for(Nd4jLong i = start; i < outLen; i+= step) {
                auto xzCoord = xIdx + threadIdx.x * rank;
                //auto zxCoord = xIdx + (threadIdx.x + threadIdx.x % 2 + 1) * rank;

                shape::index2coords(rank, zShapeOf, i, xzCoord);
                auto outOffset = shape::getOffset(0, zShapeOf, zStrideOf, xzCoord, rank);
//                auto intStep = blockDim.y * gridDim.y;
                for(int j = 0; j < rank; j++) {

                    const Nd4jLong inLen         = shape::sizeAt(xShape, j);
                    Nd4jLong coords[2] = {j, 0};
                    auto padOffset = shape::getOffset(0, padsShapeOf, padsStrideOf, coords, 2); // padding already has rank 2
                    const auto leftSide          = pads[padOffset];
                    const auto leftSideCorrected = leftSide - reflBorder;
                    const Nd4jLong len           = 2 * (inLen - 1) + leftSide + reflBorder;

                    if(xzCoord[j] < leftSide)                                        // left side
                        xzCoord[j] = leftSideCorrected - xzCoord[j];

                    else if(xzCoord[j] >= leftSide && xzCoord[j] < leftSide + inLen)  // middle
                        xzCoord[j] = xzCoord[j] - leftSide;

                    else if (len > xzCoord[j])                                                           // right side
                        xzCoord[j] = len - xzCoord[j];
                    else
                        xzCoord[j] = xzCoord[j] - len;
                }

                auto inOffset  = shape::getOffset(0, xShapeOf, xStrideOf,  xzCoord,  rank);

                z[outOffset] = x[inOffset];
            }
    }

    template<typename F, typename I>
    static void mirrorPad_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        // mode:  0 - REFLECT, else - SYMMETRIC
        const int reflBorder = (bool)mode ? 1 : 0;
        const int rank        = input.rankOf();
        const Nd4jLong outLen = output.lengthOf();
        auto stream = context->getCudaStream();

        if(rank <= 1) {

            const Nd4jLong inLen         = input.lengthOf();
            const auto leftSide          = paddings.e<Nd4jLong>(0);
            const auto leftSideCorrected = leftSide - reflBorder;
            const Nd4jLong len           = 2*(inLen-1) + leftSide + reflBorder;

            mirrorPadLinearKernel<F><<<256, 512, 256, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftSide, leftSideCorrected, inLen, len, outLen);
            nd4j::DebugHelper::checkErrorCode(stream, "helpers::mirrorPadLinearKernel(...) failed");
        }
        else {
            mirrorPadKernel<F, I><<<256, 256, 8192, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), outLen, paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), reflBorder);
            nd4j::DebugHelper::checkErrorCode(stream, "helpers::mirrorPadKernel(...) failed");
        }
        output.tickWriteDevice();
    }

    void mirrorPad(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_DOUBLE_SELECTOR(input.dataType(), paddings.dataType(), mirrorPad_, (context, input, paddings, output, mode), LIBND4J_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void mirrorPad_, (nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES, INTEGER_TYPES);

//////////////////////////////////////////////////////////////////////////
void concat(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

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
    static void tileBP_(nd4j::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    }

    void tileBP(nd4j::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (context, gradO, gradI, reps), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void tileBP_, (nd4j::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps), FLOAT_TYPES);

    void scatterSimple(const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {

    }


BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher,  (const int numOfArrs, const cudaStream_t *stream, void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo), LIBND4J_TYPES);
BUILD_DOUBLE_TEMPLATE(template void padCudaLauncher,     (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const int mode, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const void* vPadVal), LIBND4J_TYPES, INTEGER_TYPES);

}
}
}
