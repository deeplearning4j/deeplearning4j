/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
// implemented algorithm is GPU adaptation of algorithm described in following article:
// "MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm", https://arxiv.org/abs/1508.03167
//

#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <numeric>
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <helpers/PointersManager.h>

namespace sd    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static __global__ void fisherYatesCuda(sd::graph::RandomGenerator* rng, void* vx, const Nd4jLong ews, const Nd4jLong len, const int power) {

    T* x = reinterpret_cast<T*>(vx);

    __shared__ T* shmem, temp;
    __shared__ Nd4jLong ind, blockOffset, lenPerBlock;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char sharedMemory[];
        shmem = reinterpret_cast<T*>(sharedMemory);

        blockOffset = (len * blockIdx.x) >> power;
        lenPerBlock = ((len * (blockIdx.x + 1)) >> power) - blockOffset;
        ind = blockOffset;
    }
    __syncthreads();

    // copy from global memory to shared memory
    if(threadIdx.x < lenPerBlock)
        shmem[threadIdx.x] = x[(blockOffset + threadIdx.x) * ews];
    __syncthreads();

    // *** apply Fisher-Yates shuffle to lenPerBlock number of elements
    if (threadIdx.x == 0) {
        for(Nd4jLong i = lenPerBlock - 1; i > 0; --i) {
           const Nd4jLong j = rng->relativeLong(ind++) % (i + 1);
            if(i != j) {
                temp = shmem[i];
                shmem[i] = shmem[j];
                shmem[j] = temp;
            }
        }
    }
    __syncthreads();

    // copy from shared memory to global memory
    if(threadIdx.x < lenPerBlock)
        x[(blockOffset + threadIdx.x) * ews] = shmem[threadIdx.x];
}

template <typename T>
static __global__ void mergeShuffleCuda(sd::graph::RandomGenerator* rng, void* vx, const Nd4jLong ews, const Nd4jLong len, const int power, const Nd4jLong iterNum) {


    T* x = reinterpret_cast<T*>(vx);

    __shared__ Nd4jLong ind, blockOffset, factor, beg, mid, totLen, iterExp;

    // *** apply mergeShuffle algorithm
    if(threadIdx.x == 0) {

        factor = blockIdx.x << iterNum;
        iterExp = 1 << (iterNum - 1);
        blockOffset = (len * factor) >> power;
        mid         = ((len * (factor + iterExp)) >> power) - blockOffset;                // middle
        totLen      = ((len * (factor + 2*iterExp)) >> power) - blockOffset;
        ind         = iterNum * len + blockOffset;
        beg = 0;               // beginning

        // printf("m %lld, blockIdx.x %lld, factor %lld, blockOffset %lld, mid %lld, totLen %lld \n", m,k,factor,blockOffset,mid,totLen);

        while (true) {
            if(rng->relativeLong(ind++) % 2) {
                if(mid == totLen)
                    break;
                math::nd4j_swap<T>(x[(blockOffset + beg) * ews], x[(blockOffset + mid++) * ews]);
            } else {
                if(beg == mid)
                    break;
            }
            ++beg;
        }

        // Fisher-Yates
        while (beg < totLen) {
            const Nd4jLong e = rng->relativeLong(ind++) % (beg + 1);
            if(beg != e)
                math::nd4j_swap<T>(x[(blockOffset + beg) * ews], x[(blockOffset + e) * ews]);
            ++beg;
        }
    }
}


//////////////////////////////////////////////////////////////////////////
// Fisher-Yates shuffle
template <typename T>
static void fisherYates(sd::graph::RandomGenerator& rng, T* buff, const Nd4jLong& len, const Nd4jLong& ews, Nd4jLong ind) {

    for(Nd4jLong i = len-1; i > 0; --i) {
        const Nd4jLong j = rng.relativeLong(ind++) % (i + 1);
        if(i != j)
            math::nd4j_swap<T>(buff[i*ews], buff[j*ews]);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void randomShuffle_(sd::LaunchContext* context, NDArray& input, NDArray& output, sd::graph::RandomGenerator& rng, const bool isInplace) {

    const int firstDim = input.sizeAt(0);
    int temp;

    if(input.lengthOf() == 1 || firstDim == 1) {

        if(!isInplace)
            output.assign(input);
    }
    else if (shape::isCommonVector(input.shapeInfo(), temp)) {

        NDArray* arr = &input;

        if (!isInplace) {
            output.assign(input);
            arr = &output;
        }

        const Nd4jLong len = arr->lengthOf();

        const int threadsPerBlock = MAX_NUM_THREADS;

        int power = 0;
        while ((len >> power) > threadsPerBlock)
            ++power;

        const int blocksPerGrid = 1 << power;
        const int sharedMem = threadsPerBlock * input.sizeOfT() + 256;

        PointersManager manager(context, "NDArray::randomShuffle cuda");

        sd::graph::RandomGenerator* pRng = reinterpret_cast<sd::graph::RandomGenerator*>(manager.replicatePointer(&rng, sizeof(sd::graph::RandomGenerator)));

        NDArray::prepareSpecialUse({arr}, {arr});
        fisherYatesCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *context->getCudaStream()>>>(pRng, arr->specialBuffer(), arr->ews(), len, power);
        for (Nd4jLong j = 1, i = 1; j < blocksPerGrid; j += j, ++i)
            mergeShuffleCuda<T><<<blocksPerGrid/(2*j), threadsPerBlock, 256, *context->getCudaStream()>>>(pRng, arr->specialBuffer(), arr->ews(), len, power, i);
        NDArray::registerSpecialUse({arr}, {arr});

        manager.synchronize();

        rng.rewindH((len + 1) * power);
    }
    else {

        auto dimsToExclude = ShapeUtils::evalDimsToExclude(input.rankOf(), {0});

        if(isInplace) {

            auto subArrsList = input.allTensorsAlongDimension(dimsToExclude);

            // Fisher-Yates shuffle
            for(int i = firstDim - 1; i > 0; --i) {
                const int j = rng.relativeInt(i) % (i + 1);
                if(i != j)
                    subArrsList.at(i)->swapUnsafe(*subArrsList.at(j));
            }
        }
        else {

            auto subArrsListIn  = input.allTensorsAlongDimension(dimsToExclude);
            auto subArrsListOut = output.allTensorsAlongDimension(dimsToExclude);

            std::vector<int> indices(firstDim);
            std::iota(indices.begin(), indices.end(), 0);   // 0,1,2,3, ... firstDim-1

            // shuffle indices
            fisherYates<int>(rng, indices.data(), firstDim, 1, 0);

            auto func = PRAGMA_THREADS_FOR {

                for (auto i = start; i < stop; ++i)
                    subArrsListOut.at(i)->assign(subArrsListIn.at(indices[i]));
            };

            samediff::Threads::parallel_for(func, 0, firstDim);
        }

        rng.rewindH(firstDim-1);
    }
}

/////////////////////////////////////////////////////////////////////////
void randomShuffle(sd::LaunchContext * context, NDArray& input, NDArray& output, sd::graph::RandomGenerator& rng, const bool isInplace) {
    BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (context, input, output, rng, isInplace), LIBND4J_TYPES);
}

// BUILD_SINGLE_TEMPLATE(template void randomShuffle_, (sd::LaunchContext* context, NDArray& input, NDArray& output, sd::graph::RandomGenerator& rng, const bool isInplace), LIBND4J_TYPES);



}
}
}