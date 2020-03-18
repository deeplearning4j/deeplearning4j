/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <array/NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void splitCuda(const void* vx, const Nd4jLong* xShapeInfo, void* pVz, const Nd4jLong* zTadShapeInfo, const int axis) {

    const T* x = reinterpret_cast<const T*>(vx);

    __shared__ Nd4jLong xLen, totalThreads;
    __shared__ int xRank, zDim;

    if (threadIdx.x == 0) {
        xLen  = shape::length(xShapeInfo);
        xRank = shape::rank(xShapeInfo);
        zDim  = shape::shapeOf(zTadShapeInfo)[axis];          // same for all input arrays
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    int coords[MAX_RANK];

    for (uint64_t i = tid; i < xLen; i += totalThreads) {

        shape::index2coords(i, xShapeInfo, coords);

        const auto xOffset = shape::getOffset(xShapeInfo, coords);

        auto *z = reinterpret_cast<T*>(reinterpret_cast<void **>(pVz)[coords[axis] / zDim]);

        coords[axis] %= zDim;

        const auto zOffset = shape::getOffset(zTadShapeInfo, coords);

        z[zOffset] = x[xOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void splitCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                       const void* vx, const Nd4jLong* xShapeInfo, void* pVz, const Nd4jLong* zTadShapeInfo, const int axis) {

    splitCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, pVz, zTadShapeInfo, axis);
}
BUILD_SINGLE_TEMPLATE(template void splitCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, void* pVz, const Nd4jLong* zTadShapeInfo, const int axis), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void split(sd::LaunchContext* context, const NDArray& input, std::vector<NDArray*>& outArrs, const int axis) {

    const int numOfSubArrs = outArrs.size();
    const auto sizeofT     = input.sizeOfT();

    for(int i = 0; i < numOfSubArrs; ++i)
        outArrs[i]->syncToDevice();
    input.syncToDevice();

    bool luckCase1 = ((axis == 0 && input.ordering() == 'c') || (axis == input.rankOf() - 1 && input.ordering() == 'f')) && input.ews() == 1;

    if(luckCase1) {
        for (uint i = 0; i < numOfSubArrs; ++i) {
            luckCase1 &= outArrs[i]->ordering() == input.ordering() && outArrs[i]->ews() == 1;
            if(!luckCase1)
                break;
        }
    }

    if(luckCase1) {     // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6} order f

        void* x = static_cast<int8_t*>(input.getSpecialBuffer());

        for (uint i = 0; i < numOfSubArrs; ++i) {
            const auto memAmountToCopy = outArrs[i]->lengthOf() * sizeofT;
            cudaMemcpyAsync(static_cast<int8_t*>(outArrs[i]->getSpecialBuffer()), x, memAmountToCopy, cudaMemcpyDeviceToDevice, *context->getCudaStream());
            x = static_cast<int8_t*>(x) + memAmountToCopy;
        }

        if(cudaStreamSynchronize(*context->getCudaStream()) != 0)
            throw std::runtime_error("split cuda: luckCase1 failed!");

        for(int i = 0; i < numOfSubArrs; ++i)
            outArrs[i]->tickWriteDevice();
        input.tickReadDevice();

        return;
    }

    // const bool isXcontin = input.strideAt(axis) == 1;
    // bool areOutputsContin = true;
    // bool allSameOrder    = true;
    // std::vector<Nd4jLong> strideOfContigStride(outArrs.size());

    // if(isXcontin) {

    //     for (uint i = 0; i < outArrs.size(); ++i) {

    //         areOutputsContin &= outArrs[i]->strideAt(axis) == 1;
    //         allSameOrder     &= input.ordering() == outArrs[i]->ordering();
    //         if(!areOutputsContin || !allSameOrder)
    //             break;

    //         strideOfContigStride[i] = shape::strideOverContigAxis(axis, outArrs[i]->getShapeInfo());
    //     }
    // }

    // const bool luckCase2 = isXcontin && areOutputsContin && allSameOrder;

    // if(luckCase2) {     // for example {2,1,3} + {2,5,3} + {2,10,3} = {2,16,3}, here axis 1 shoud have stride = 1 for all inputs arrays and input array

    //     const auto xStep = shape::strideOverContigAxis(axis, input.getShapeInfo());
    //     const auto zDim = outArrs[0]->sizeAt(axis);     // same for all outArrs

    //     for (uint i = 0; i < input.lengthOf() / input.sizeAt(axis); ++i) {

    //         const auto iShift = i * sizeofT;
    //         void* x = static_cast<int8_t*>(input.getSpecialBuffer()) + xStep * iShift;

    //         for (uint j = 0; j < numOfSubArrs; ++j) {
    //             void* z = static_cast<int8_t*>(outArrs[j]->getSpecialBuffer()) + strideOfContigStride[j] * iShift;
    //             const auto memSizeToCopy = zDim * sizeofT;
    //             cudaMemcpyAsync(z, x, memSizeToCopy, cudaMemcpyDeviceToDevice, *context->getCudaStream());
    //             x = static_cast<int8_t*>(x) + memSizeToCopy;
    //         }
    //     }

    //     if(cudaStreamSynchronize(*context->getCudaStream()) != 0)
    //         throw std::runtime_error("split cuda: luckCase2 failed!");
    // }
    // else {      // general (slower) case

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        // prepare arrays of pointers on buffers and shapes
        std::vector<void*> hOutBuffers(numOfSubArrs);

        for(int i = 0; i < numOfSubArrs; ++i)
            hOutBuffers[i]   = outArrs[i]->getSpecialBuffer();

        PointersManager manager(context, "helpers::split");

        void* dOutBuffers = manager.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void*));

        BUILD_SINGLE_SELECTOR(input.dataType(), splitCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), dOutBuffers, outArrs[0]->specialShapeInfo(), axis), LIBND4J_TYPES);

        manager.synchronize();
    // }

    for(int i = 0; i < numOfSubArrs; ++i)
        outArrs[i]->tickWriteDevice();
    input.tickReadDevice();
}

}
}
}