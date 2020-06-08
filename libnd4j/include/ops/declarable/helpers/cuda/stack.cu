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
// Created by Yurii Shyrma on 02.01.2018
//

#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <helpers/TAD.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
namespace ops {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
static __global__ void stackScalarsCuda(void* pVx, void* vz, const Nd4jLong* zShapeInfo) {

    T* z = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong zLen, totalThreads;

    if (threadIdx.x == 0) {
        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        const T *x = reinterpret_cast<const T*>(reinterpret_cast<void**>(pVx)[i]);
        z[shape::getIndexOffset(i, zShapeInfo)] = *x;
    }
}


///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void stackScalarsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                             void* pVx, void* vz, const Nd4jLong* zShapeInfo) {

    stackScalarsCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(pVx, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void stack_(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int dim) {

    const int numOfSubArrs = inArrs.size();

    NDArray::prepareSpecialUse({&output}, inArrs);

    if(inArrs[0]->rankOf() == 0) {

        std::vector<void const*> hInBuffers(numOfSubArrs);

        for(int i = 0; i < numOfSubArrs; ++i)
            hInBuffers[i] = inArrs[i]->specialBuffer();

        PointersManager manager(context, "helpers::stack cuda");

        void* dInBuffers = manager.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void*));

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        stackScalarsCudaLauncher<T>(blocksPerGrid, threadsPerBlock, context->getCudaStream(), dInBuffers, output.specialBuffer(), output.specialShapeInfo());

        manager.synchronize();
    }
    else {

        auto zTadPack = ConstantTadHelper::getInstance().tadForDimensions(output.shapeInfo(), ShapeUtils::evalDimsToExclude(output.rankOf(), {dim}));
        auto zTadShapeInfo  = zTadPack.primaryShapeInfo();

        for (uint i = 0; i < numOfSubArrs; ++i) {

            void* zBuff = output.specialBufferWithOffset(zTadPack.primaryOffsets()[i]);

            NativeOpExecutioner::execTransformAny(context, transform::Assign,
                                                 nullptr, inArrs[i]->shapeInfo(), inArrs[i]->specialBuffer(), inArrs[i]->specialShapeInfo(),
                                                 nullptr, zTadShapeInfo,             zBuff,                         zTadPack.specialShapeInfo(),
                                                 nullptr, nullptr, nullptr, false/*allowParallelism*/);
        }
    }

   NDArray::registerSpecialUse({&output}, inArrs);
}

////////////////////////////////////////////////////////////////////////
void stack(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int dim) {
    BUILD_SINGLE_SELECTOR(output.dataType(), stack_, (context, inArrs, output, dim), LIBND4J_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void stack_ , (sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int dim), LIBND4J_TYPES);


///////////////////////////////////////////////////////////////////
template <typename T>
static __global__ void unstackScalarsCuda(const void* vx, const Nd4jLong* xShapeInfo, void* pVz) {

    const T* x = reinterpret_cast<const T*>(vx);

    __shared__ Nd4jLong xLen, totalThreads;

    if (threadIdx.x == 0) {
        xLen  = shape::length(xShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < xLen; i += totalThreads) {

        T* z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[i]);
        *z = x[shape::getIndexOffset(i, xShapeInfo)];
    }
}


///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void unstackScalarsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                                const void* vx, const Nd4jLong* xShapeInfo, void* pVz) {

    unstackScalarsCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, pVz);
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void unstack_(sd::LaunchContext* context, const NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {

    const int numOfSubArrs = outArrs.size();

    // NDArray::prepareSpecialUse(outArrs, {&input});
    input.syncToDevice();
    for (const auto a : outArrs)
        a->getDataBuffer()->allocateSpecial();


    if(outArrs[0]->rankOf() == 0) {

        std::vector<void*> hOutBuffers(numOfSubArrs);

        for(int i = 0; i < numOfSubArrs; ++i)
            hOutBuffers[i] = outArrs[i]->specialBuffer();

        PointersManager manager(context, "helpers::unstack cuda");

        void* dOutBuffers = manager.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void*));

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        unstackScalarsCudaLauncher<T>(blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), dOutBuffers);

        manager.synchronize();
    }
    else {

        auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), ShapeUtils::evalDimsToExclude(input.rankOf(), {dim}));
        auto xTadShapeInfo  = xTadPack.primaryShapeInfo();

        for (uint i = 0; i < numOfSubArrs; ++i) {

            auto xBuff = input.specialBufferWithOffset(xTadPack.primaryOffsets()[i]);

            NativeOpExecutioner::execTransformAny(input.getContext(), transform::Assign,
                                                 nullptr, xTadShapeInfo,              xBuff,                       xTadPack.specialShapeInfo(),
                                                 nullptr, outArrs[i]->shapeInfo(), outArrs[i]->specialBuffer(), outArrs[i]->specialShapeInfo(),
                                                 nullptr, nullptr, nullptr, false/*allowParallelism*/);
        }
    }

    // NDArray::registerSpecialUse(outArrs, {&input});
    input.tickReadDevice();
    for (const auto p : outArrs)
        p->tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
void unstack(sd::LaunchContext* context, const NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {
    BUILD_SINGLE_SELECTOR(input.dataType(), unstack_, (context, input, outArrs, dim), LIBND4J_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void unstack_, (sd::LaunchContext* context, const NDArray& input, const std::vector<NDArray*>& outArrs, const int dim), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
// template <typename T>
// static __global__ void unstackCuda(const void* vx, const Nd4jLong* xShapeInfo, void* pVz, const Nd4jLong* zTadShapeInfo, const int axis) {

// 	const T* x = reinterpret_cast<const T*>(vx);
//     __shared__ Nd4jLong xLen, totalThreads;
//     __shared__ int xRank;

//     if (threadIdx.x == 0) {
//         xLen  = shape::length(xShapeInfo);
//         xRank = shape::rank(xShapeInfo);
//         totalThreads = gridDim.x * blockDim.x;
//     }
//     __syncthreads();

//     const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

//     Nd4jLong coords[MAX_RANK];

//     for (uint64_t i = tid; i < xLen; i += totalThreads) {

//         shape::index2coords(i, xShapeInfo, coords);

//         const auto xOffset = shape::getOffset(xShapeInfo, coords);

//         T *z = reinterpret_cast<T*>(reinterpret_cast<void **>(pVz)[coords[axis]]);

//         for (uint j = axis; j < xRank - 1; ++j)	// shift coords staring from axis position
//         	coords[j] = coords[j + 1];

//         const auto zOffset = shape::getOffset(zTadShapeInfo, coords);

//         z[zOffset] = x[xOffset];
//     }
// }

// ///////////////////////////////////////////////////////////////////
// template<typename T>
// __host__ static void unstackCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
// 										 const void* vx, const Nd4jLong* xShapeInfo, void* pVz, const Nd4jLong* zTadShapeInfo, const int axis) {

//     unstackCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, pVz, zTadShapeInfo, axis);
// }
// BUILD_SINGLE_TEMPLATE(template void unstackCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, void* pVz, const Nd4jLong* zTadShapeInfo, const int axis), LIBND4J_TYPES);


// ///////////////////////////////////////////////////////////////////
// void unstack(sd::LaunchContext* context, const NDArray& input, const std::vector<const NDArray*>& outArrs, const int axis) {

// 	const int threadsPerBlock = MAX_NUM_THREADS / 2;
// 	const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

// 	const int numOfSubArrs = outArrs.size();

//     std::vector<void*> hOutBuffers(numOfSubArrs);

//     for(int i = 0; i < numOfSubArrs; ++i)
//         hOutBuffers[i] = outArrs[i]->specialBuffer();

//     PointersManager manager(context, "helpers::unstack");

//     void* dOutBuffers = manager.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void*));

//     for(uint i = 0; i < numOfSubArrs; ++i)
// 		outArrs[i]->syncToDevice();
//     input.syncToDevice();

//     BUILD_SINGLE_SELECTOR(input.dataType(), unstackCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), dOutBuffers, outArrs[0]->special(), axis), LIBND4J_TYPES);

//     manager.synchronize();

//     for(uint i = 0; i < numOfSubArrs; ++i)
//         outArrs[i]->tickReadDevice();
//     input.tickWriteDevice();
// }


// ///////////////////////////////////////////////////////////////////
// template <typename T>
// static __global__ void stackCuda(void* pVx, const Nd4jLong* xTadShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int axis) {

// 	T* z = reinterpret_cast<T*>(vz);

//     __shared__ Nd4jLong zLen, totalThreads;
//     __shared__ int zRank;

//     if (threadIdx.x == 0) {
//         zLen  = shape::length(zShapeInfo);
//         zRank = shape::rank(zShapeInfo);
//         totalThreads = gridDim.x * blockDim.x;
//     }
//     __syncthreads();

//     const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

//     Nd4jLong coords[MAX_RANK];

//     for (uint64_t i = tid; i < zLen; i += totalThreads) {

//         shape::index2coords(i, zShapeInfo, coords);

//         const auto zOffset = shape::getOffset(zShapeInfo, coords);

//         const T *x = reinterpret_cast<const T*>(reinterpret_cast<void**>(pVx)[coords[axis]]);

//         for (uint j = axis; j < zRank - 1; ++j)	// shift coords staring from axis position
//         	coords[j] = coords[j + 1];

//         const auto xOffset = shape::getOffset(xTadShapeInfo, coords);

//         z[zOffset] = x[xOffset];
//     }
// }

// ///////////////////////////////////////////////////////////////////
// template<typename T>
// __host__ static void stackCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
// 					 				   void* pVx, const Nd4jLong* xTadShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int axis) {

//     stackCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(pVx, xTadShapeInfo, vz, zShapeInfo, axis);
// }
// BUILD_SINGLE_TEMPLATE(template void stackCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, void* pVx, const Nd4jLong* xTadShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int axis), LIBND4J_TYPES);


// ///////////////////////////////////////////////////////////////////
// void stack(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int axis) {

// 	const int threadsPerBlock = MAX_NUM_THREADS / 2;
// 	const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

// 	const int numOfSubArrs = inArrs.size();

//     std::vector<void*> hInBuffers(numOfSubArrs);

//     for(int i = 0; i < numOfSubArrs; ++i)
//         hInBuffers[i] = inArrs[i]->specialBuffer();

//     PointersManager manager(context, "helpers::stack");

//     void* dInBuffers = manager.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void*));

//     for(uint i = 0; i < numOfSubArrs; ++i)
// 		inArrs[i]->syncToDevice();
//     output.syncToDevice();

//     BUILD_SINGLE_SELECTOR(output.dataType(), stackCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), dInBuffers, inArrs[0]->specialShapeInfo(), output.specialBuffer(), output.special(), axis), LIBND4J_TYPES);

//     manager.synchronize();

//     for(uint i = 0; i < numOfSubArrs; ++i)
//         inArrs[i]->tickReadDevice();
//     output.tickWriteDevice();
// }

}
}
}

