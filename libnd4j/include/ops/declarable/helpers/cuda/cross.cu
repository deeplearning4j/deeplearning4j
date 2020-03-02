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
// @author Yurii Shyrma, created on 10.06.2019
//


#include <ops/declarable/helpers/cross.h>
#include <helpers/PointersManager.h>


namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void crossCuda(const void* vx, const Nd4jLong* xShapeInfo,
                                 const void* vy, const Nd4jLong* yShapeInfo,
                                 	   void* vz, const Nd4jLong* zShapeInfo) {

    __shared__ const T* x;
    __shared__ const T* y;
    __shared__ 		 T* z;
    __shared__ int rank;
    __shared__ Nd4jLong lenWithoutLastDim, totalThreads, *sharedMem;

    if (threadIdx.x == 0) {
    	x = reinterpret_cast<const T*>(vx);
    	y = reinterpret_cast<const T*>(vy);
    	z = reinterpret_cast<T*>(vz);

        extern __shared__ unsigned char shmem[];
        sharedMem    = reinterpret_cast<Nd4jLong*>(shmem);
        totalThreads = gridDim.x * blockDim.x;

        rank              = shape::rank(xShapeInfo);
        lenWithoutLastDim = shape::length(xShapeInfo) / xShapeInfo[rank]; //  shape::length(xShapeInfo) / 3;
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = tid; i < lenWithoutLastDim; i += totalThreads) {

        shape::index2coords(i, rank - 1, xShapeInfo + 1, coords);

        coords[rank - 1] = 0;

        auto xOffset = shape::getOffset(xShapeInfo, coords);
        auto yOffset = shape::getOffset(yShapeInfo, coords);

        const auto x0 = x[xOffset];
        const auto y0 = y[yOffset];

		xOffset += shape::stride(const_cast<Nd4jLong*>(xShapeInfo))[rank - 1];
		yOffset += shape::stride(const_cast<Nd4jLong*>(yShapeInfo))[rank - 1];

		const auto x1 = x[xOffset];
        const auto y1 = y[yOffset];

        xOffset += shape::stride(const_cast<Nd4jLong*>(xShapeInfo))[rank - 1];
		yOffset += shape::stride(const_cast<Nd4jLong*>(yShapeInfo))[rank - 1];

		const auto x2 = x[xOffset];
        const auto y2 = y[yOffset];

        auto zOffset = shape::getOffset(zShapeInfo, coords);
        z[zOffset] = x1 * y2 - x2 * y1;

        zOffset += shape::stride(const_cast<Nd4jLong*>(zShapeInfo))[rank - 1];
        z[zOffset] = x2 * y0 - x0 * y2;

        zOffset += shape::stride(const_cast<Nd4jLong*>(zShapeInfo))[rank - 1];
		z[zOffset] = x0 * y1 - x1 * y0;
    }
}

template<typename T>
__host__ static void crossCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
							   		   const void* vx, const Nd4jLong* xShapeInfo,
							   		   const void* vy, const Nd4jLong* yShapeInfo,
									 	     void* vz, const Nd4jLong* zShapeInfo) {

    crossCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}
BUILD_SINGLE_TEMPLATE(template void crossCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo), NUMERIC_TYPES);


void crossBatched(sd::LaunchContext* context, NDArray *x, NDArray *y, NDArray *z) {

	const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (x->lengthOf() / x->sizeAt(-1) + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = sizeof(Nd4jLong) * threadsPerBlock * x->rankOf() + 128;

    PointersManager manager(context, "cross");

    NDArray::prepareSpecialUse({z}, {x, y});
    BUILD_SINGLE_SELECTOR(x->dataType(), crossCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), x->getSpecialBuffer(), x->getSpecialShapeInfo(), y->getSpecialBuffer(), y->getSpecialShapeInfo(), z->specialBuffer(), z->specialShapeInfo()), NUMERIC_TYPES);
    NDArray::registerSpecialUse({z}, {x, y});

    manager.synchronize();
}

}
}
}