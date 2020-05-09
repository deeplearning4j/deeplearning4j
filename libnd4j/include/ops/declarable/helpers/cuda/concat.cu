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
__global__ static void concatCuda(void* pVx,  void* pxShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int axis) {

    T* z = reinterpret_cast<T*>(vz);
    __shared__ Nd4jLong zLen, totalThreads;
    __shared__ int rank;

    if (threadIdx.x == 0) {
        zLen = shape::length(zShapeInfo);
        rank = shape::rank(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    int coords[MAX_RANK];

    for (uint64_t i = tid; i < zLen; i += totalThreads) {
        shape::index2coords(i, zShapeInfo, coords);

        const auto zOffset = shape::getOffset(zShapeInfo, coords);

        int inArrIdx = 0;
        Nd4jLong *xShapeInfo = reinterpret_cast<Nd4jLong **>(pxShapeInfo)[inArrIdx];

        while (coords[axis] >= xShapeInfo[axis + 1]) {
            coords[axis] -= xShapeInfo[axis + 1];
            xShapeInfo = reinterpret_cast<Nd4jLong **>(pxShapeInfo)[++inArrIdx];
        }

        const auto *x = reinterpret_cast<T *>(reinterpret_cast<void **>(pVx)[inArrIdx]);
        const auto xOffset = shape::getOffset(xShapeInfo, coords);

        z[zOffset] = x[xOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void concatCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                        void* pVx, void* pxShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int axis) {

    concatCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(pVx, pxShapeInfo, vz, zShapeInfo, axis);
}

//////////////////////////////////////////////////////////////////////////
void concat(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int axis) {

    const int numOfInArrs = inArrs.size();
    const auto sizeofT    = output.sizeOfT();

    NDArray::prepareSpecialUse({&output}, inArrs);

    bool luckCase1 = ((axis == 0 && output.ordering() == 'c') || (axis == output.rankOf() - 1 && output.ordering() == 'f')) && output.ews() == 1;

    if(luckCase1) {
        for (uint i = 0; i < numOfInArrs; ++i) {
            luckCase1 &= inArrs[i]->ordering() == output.ordering() && inArrs[i]->ews() == 1;
            if(!luckCase1)
                break;
        }
    }

    if(luckCase1) {     // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6} order f

        void* z = static_cast<int8_t*>(output.specialBuffer());

        for (uint i = 0; i < numOfInArrs; ++i) {
            const auto memAmountToCopy = inArrs[i]->lengthOf() * sizeofT;
            cudaMemcpyAsync(z, reinterpret_cast<const int8_t*>(inArrs[i]->specialBuffer()), memAmountToCopy, cudaMemcpyDeviceToDevice, *context->getCudaStream());
            z = static_cast<int8_t*>(z) + memAmountToCopy;
        }

        if(cudaStreamSynchronize(*context->getCudaStream()) != 0)
            throw std::runtime_error("concat cuda: luckCase1 failed!");

        for(int i = 0; i < numOfInArrs; ++i)
            inArrs[i]->tickReadDevice();
        output.tickWriteDevice();

        return;
    }

    // const bool isZcontin = output.strideAt(axis) == 1;
    // bool areInputsContin = true;
    // bool allSameOrder    = true;
    // std::vector<Nd4jLong> strideOfContigStride(numOfInArrs);

    // if(isZcontin) {

    //     for (uint i = 0; i < inArrs.size(); ++i) {

    //         areInputsContin &= inArrs[i]->strideAt(axis) == 1;
    //         allSameOrder    &= output.ordering() == inArrs[i]->ordering();
    //         if(!areInputsContin || !allSameOrder)
    //             break;

    //         strideOfContigStride[i] = shape::strideOverContigAxis(axis, inArrs[i]->shapeInfo());
    //     }
    // }

    // const bool luckCase2 = isZcontin && areInputsContin && allSameOrder;

    // if(luckCase2) {     // for example {2,1,3} + {2,5,3} + {2,10,3} = {2,16,3}, here axis 1 shoud have stride = 1 for all inputs arrays and output array

    //     const auto zStep = shape::strideOverContigAxis(axis, output.shapeInfo());

    //     for (uint i = 0; i < output.lengthOf() / output.sizeAt(axis); ++i) {

    //         const auto iShift = i * sizeofT;
    //         void* z = static_cast<int8_t*>(output.specialBuffer()) + zStep * iShift;

    //         for (uint j = 0; j < numOfInArrs; ++j) {
    //             const auto xDim = inArrs[j]->sizeAt(axis);
    //             void* x = static_cast<int8_t*>(inArrs[j]->specialBuffer()) + strideOfContigStride[j] * iShift;
    //             const auto memSizeToCopy = xDim * sizeofT;
    //             cudaMemcpyAsync(z, x, memSizeToCopy, cudaMemcpyDeviceToDevice, *context->getCudaStream());
    //             z = static_cast<int8_t*>(z) + memSizeToCopy;
    //         }
    //     }

    //     if(cudaStreamSynchronize(*context->getCudaStream()) != 0)
    //         throw std::runtime_error("concat cuda: luckCase2 failed!");
    // }
    // else {      // general (slower) case

        const int threadsPerBlock = 256;
        const int blocksPerGrid = 512;
        const int sharedMem = 512;

        // prepare arrays of pointers on buffers and shapes
        std::vector<const void*> hInBuffers(numOfInArrs);
        std::vector<const Nd4jLong*> hInShapeInfo(numOfInArrs);

        for(int i = 0; i < numOfInArrs; ++i) {
            hInBuffers[i]   = inArrs[i]->specialBuffer();
            hInShapeInfo[i] = inArrs[i]->specialShapeInfo();
        }

        PointersManager manager(context, "helpers::concat");

        void* dInBuffers   = manager.replicatePointer(hInBuffers.data(),    hInBuffers.size() * sizeof(void*));
        void* dInShapeInfo = manager.replicatePointer(hInShapeInfo.data(),  hInShapeInfo.size() * sizeof(Nd4jLong*));

        BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), dInBuffers, dInShapeInfo, output.specialBuffer(), output.specialShapeInfo(), axis), LIBND4J_TYPES);

        manager.synchronize();
    // }

    NDArray::registerSpecialUse({&output}, inArrs);
}

}
}
}