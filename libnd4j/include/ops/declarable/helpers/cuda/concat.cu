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

namespace nd4j    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void concatCuda(void* pVx,  void* pxShapeInfo, void* vz, Nd4jLong* zShapeInfo, const int axis) {

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

    Nd4jLong coords[MAX_RANK];

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
                                        void* pVx, void* pxShapeInfo, void* vz, Nd4jLong* zShapeInfo, const int axis) {

    concatCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(pVx, pxShapeInfo, vz, zShapeInfo, axis);
}
BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, void* pVx, void* pxShapeInfo, void* vz, Nd4jLong* zShapeInfo, const int axis), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void concat(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

    const int threadsPerBlock = 256;
    const int blocksPerGrid = 512;
    const int sharedMem = 512;

    const int numOfArrs = inArrs.size();

    for(int i = 0; i < numOfArrs; ++i)
        inArrs[i]->syncToDevice();

    output.syncToDevice();

    // prepare arrays of pointers on buffers and shapes
    std::vector<void*> hInBuffers(numOfArrs);
    std::vector<Nd4jLong*> hInShapeInfo(numOfArrs);

    for(int i = 0; i < numOfArrs; ++i) {
        hInBuffers[i]   = inArrs[i]->getSpecialBuffer();
        hInShapeInfo[i] = inArrs[i]->getSpecialShapeInfo();
    }

    PointersManager manager(context, "helpers::concat");

    void* dInBuffers   = manager.replicatePointer(hInBuffers.data(),    hInBuffers.size() * sizeof(void*));
    void* dInShapeInfo = manager.replicatePointer(hInShapeInfo.data(),  hInShapeInfo.size() * sizeof(Nd4jLong*));

    BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), dInBuffers, dInShapeInfo, output.specialBuffer(), output.specialShapeInfo(), axis), LIBND4J_TYPES);

    manager.synchronize();

    for(int i = 0; i < numOfArrs; ++i)
        inArrs[i]->tickReadDevice();

    output.tickWriteDevice();
}

}
}
}