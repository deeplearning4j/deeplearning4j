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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/dilation2d.h>
#include <array/DataTypeUtils.h>
#include <helpers/PointersManager.h>

namespace sd    {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
__global__ static void dilation2dCuda(const void* vx, const Nd4jLong* xShapeInfo,
									  const void* vy, const Nd4jLong* yShapeInfo,
									  		void* vz, const Nd4jLong* zShapeInfo,
									  const int sH, const int sW,
									  const int pH, const int pW,
									  const int dH, const int dW) {

	// x [bS, iH, iW, iC]
	// y [kH, kW, iC]
    // z [bS, oH, oW, iC]

    const X* x = reinterpret_cast<const X*>(vx);
    const X* y = reinterpret_cast<const X*>(vy);
          Z* z = reinterpret_cast<Z*>(vz);

    __shared__ int xzRank, yRank, *sharedMem;
    __shared__ uint iH, iW, kH, kW;
    __shared__ Nd4jLong zLen;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        zLen = shape::length(zShapeInfo);

        xzRank = shape::rank(xShapeInfo);
        yRank  = shape::rank(yShapeInfo);

        iH = xShapeInfo[2];
        iW = xShapeInfo[3];

        kH = yShapeInfo[1];
        kW = yShapeInfo[2];
    }
    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto xzCoords = sharedMem + threadIdx.x * (xzRank + yRank);
    auto yCoords  = xzCoords + xzRank;

    shape::index2coords(zInd, zShapeInfo, xzCoords);

    const auto zOffset = shape::getOffset(zShapeInfo, xzCoords);

    yCoords[2] = xzCoords[3];		// iC coordinate is same for x, y and z

    const auto oh = xzCoords[1];
    const auto ow = xzCoords[2];

    X max = -DataTypeUtils::max<X>();

	for (yCoords[0] = 0; yCoords[0] < kH; ++yCoords[0]) {
    	xzCoords[1] = oh * sH - pH + yCoords[0] * dH;
        if (xzCoords[1] < 0 || xzCoords[1] >= iH) continue;

        for (yCoords[1] = 0; yCoords[1] < kW; ++yCoords[1]) {
        	xzCoords[2] = ow * sW - pW + yCoords[1] * dW;
            if(xzCoords[2] < 0 || xzCoords[2] >= iW) continue;

            const X val = x[shape::getOffset(xShapeInfo, xzCoords)] + y[shape::getOffset(yShapeInfo, yCoords)];
            if (val > max)
            	max = val;
		}
	}

	z[zOffset] = static_cast<Z>(max);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void dilation2dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                   const void* vx, const Nd4jLong* xShapeInfo,
                                   const void* vy, const Nd4jLong* yShapeInfo,
                                         void* vz, const Nd4jLong* zShapeInfo,
                                   const int sH, const int sW,
								   const int pH, const int pW,
								   const int dH, const int dW) {

    dilation2dCuda<X,Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, sH, sW, pH, pW, dH, dW);
}

void dilation2d(sd::LaunchContext* context, NDArray *input, NDArray *weights, NDArray *output, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW) {

   	PointersManager manager(context, "dilation2d");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = (weights->rankOf() + output->rankOf()) * sizeof(int) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({output}, {input, weights});
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), dilation2dCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), weights->getSpecialBuffer(), weights->getSpecialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), sH, sW, pH, pW, dH, dW), FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input, weights});

    manager.synchronize();
}


}
}
}
