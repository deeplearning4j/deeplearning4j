/*******************************************************************************
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

#include <op_boilerplate.h>
#include <ops/declarable/helpers/imagesHelpers.h>
#include <PointersManager.h>


namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
// for example xShapeInfo = {2,3,4}, zShapeInfo = {2,1,4}
template<typename T>
__global__ void rgbToGrsCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int dimC) {

	const auto x = reinterpret_cast<const T*>(vx);
		  auto z = reinterpret_cast<T*>(vz);

	__shared__ Nd4jLong zLen, *sharedMem;
	__shared__ int rank;	// xRank == zRank

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

		rank = shape::length(xShapeInfo);
	}
	__syncthreads();

	Nd4jLong* coords = sharedMem + threadIdx.x * rank;

	for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i +=  gridDim.x * blockDim.x) {

		if (dimC == (rank - 1) && 'c' == shape::order(xShapeInfo) && 1 == shape::elementWiseStride(xShapeInfo) && 'c' == shape::order(zShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo)) {
			const auto xStep = i*3;
            z[i] = 0.2989f*x[xStep] + 0.5870f*x[xStep + 1] + 0.1140f*x[xStep + 2];
		}
		else {

	    	shape::index2coords(i, zShapeInfo, coords);

            const auto zOffset  = shape::getOffset(zShapeInfo, coords);
            const auto xOffset0 = shape::getOffset(xShapeInfo, coords);
            const auto xOffset1 = xOffset0 + shape::stride(xShapeInfo)[dimC];
            const auto xOffset2 = xOffset1 + shape::stride(xShapeInfo)[dimC];
		}
	}
}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void rgbToGrsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int dimC) {

	rgbToGrsCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, dimC);
}

///////////////////////////////////////////////////////////////////
void rgbToGrs(nd4j::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {

	PointersManager manager(context, "rgbToGrs");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = input.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

	NDArray::prepareSpecialUse({&output}, {&input});
	BUILD_SINGLE_SELECTOR(input.dataType(), rgbToGrsCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), dimC), NUMERIC_TYPES);
	NDArray::registerSpecialUse({&output}, {&input});

	manager.synchronize();
}

}
}
}

