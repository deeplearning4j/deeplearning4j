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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 30.05.2019
//


#include <ops/declarable/helpers/one_hot.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <array/NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>


namespace sd 		{
namespace ops		{
namespace helpers 	{

///////////////////////////////////////////////////////////////////
// x - indices, z - output
template<typename X, typename Z>
__global__ static void onehotCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const uint axis, const uint depth, const Z on, const Z off) {

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int xRank, zRank;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);
        xRank = shape::rank(xShapeInfo);
        zRank = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    auto coord = sharedMem + threadIdx.x * zRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, coord);
        const auto zOffset = shape::getOffset(zShapeInfo, coord);
        const auto depthCoord = coord[axis];

        for (uint j = axis; j < zRank - 1; ++j)
          coord[j] = coord[j + 1];

        const auto xOffset = shape::getOffset(xShapeInfo, coord);
        const Nd4jLong idx = x[xOffset];
        z[zOffset] = depthCoord == idx ? on : off;
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void onehotCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                               const void *vx, const Nd4jLong *xShapeInfo,
                                     void *vz, const Nd4jLong *zShapeInfo,
                               const uint axis, const uint depth,
                               const double on, const double off) {

    onehotCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, axis, depth, static_cast<Y>(on), static_cast<Y>(off));
}

///////////////////////////////////////////////////////////////////
void onehot(const sd::LaunchContext* context, const NDArray *indices, NDArray *output, const uint axis, const uint depth, const double on, const double off) {

	const auto xType = indices->dataType();
	const auto zType = output->dataType();

	const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (output->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
	const int sharedMem = threadsPerBlock * sizeof(decltype(*output->getShapeInfo())) * output->rankOf() + 128;

	PointersManager manager(context, "onehot");

    NDArray::prepareSpecialUse({output}, {indices});
  	BUILD_DOUBLE_SELECTOR(xType, zType, onehotCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), indices->getSpecialBuffer(), indices->getSpecialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), axis, depth, on, off), LIBND4J_TYPES, LIBND4J_TYPES);
  	NDArray::registerSpecialUse({output}, {indices});

    manager.synchronize();
}


}
}
}