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
// @author Yurii Shyrma, created on 25.02.2018
//


#include<ops/declarable/helpers/batchnorm.h>
#include <helpers/ShapeUtils.h>
#include <OmpLaunchHelper.h>
#include <ConstantTadHelper.h>
#include <PointersManager.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void batchnormCuda(const void* vx, const Nd4jLong* xShapeInfo,
									const void* vMean, const Nd4jLong* meanShapeInfo,
									const void* vVariance, const Nd4jLong* varianceShapeInfo,
									const void* vGamma, const Nd4jLong* gammaShapeInfo,
									const void* vBeta, const Nd4jLong* betaShapeInfo,
										  void* vz, const Nd4jLong* zShapeInfo,
									const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
									const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets,
									const T epsilon) {

	const auto x    	= reinterpret_cast<const T*>(vx);
          auto z        = reinterpret_cast<T*>(vz);
	const auto mean 	= reinterpret_cast<const T*>(vMean);
	const auto variance = reinterpret_cast<const T*>(vVariance);
	const auto gamma    = reinterpret_cast<const T*>(vGamma);
	const auto beta     = reinterpret_cast<const T*>(vBeta);
	const auto zRank    = reinterpret_cast<T*>(vz);

    // maxRank = xRank = zRank, minRank = meanRank = varianceRank = gammaRank = betaRank
    __shared__ Nd4jLong minLen, tadLen, totalThreads;

    if (threadIdx.x == 0) {

        totalThreads = gridDim.x * blockDim.x;

        minLen = shape::length(meanShapeInfo);
        tadLen = shape::length(xShapeInfo) / minLen;
    }
    __syncthreads();


    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = tid; i < minLen; i += totalThreads) {

		const auto meanOffset     = shape::getIndexOffset(i, meanShapeInfo, minLen);
    	const auto varianceOffset = shape::getIndexOffset(i, varianceShapeInfo, minLen);

    	T sigmaInvGam = 1. / nd4j::math::nd4j_sqrt<T, T>(variance[varianceOffset] + epsilon);

    	if(gamma != nullptr)
    		sigmaInvGam *= gamma[shape::getIndexOffset(i, gammaShapeInfo, minLen)];

		auto betaOffset = 0;
    	if(beta != nullptr)
    		betaOffset = shape::getIndexOffset(i, betaShapeInfo, minLen);

    	const auto xTad = x + xTadOffsets[i];
    		  auto zTad = z + zTadOffsets[i];

    	for (uint j = 0; j < tadLen; ++j) {
    		const auto xTadOffset = shape::getIndexOffset(j, xTadShapeInfo, tadLen);
    		const auto zTadOffset = shape::getIndexOffset(j, zTadShapeInfo, tadLen);

    		zTad[zTadOffset] = (xTad[xTadOffset] - mean[meanOffset]) * sigmaInvGam;

    		if(beta != nullptr)
				zTad[zTadOffset] += beta[betaOffset];
    	}
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void batchnormCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
											const void* vx, const Nd4jLong* xShapeInfo,
                                           	const void* vMean, const Nd4jLong* meanShapeInfo,
											const void* vVariance, const Nd4jLong* varianceShapeInfo,
											const void* vGamma, const Nd4jLong* gammaShapeInfo,
											const void* vBeta, const Nd4jLong* betaShapeInfo,
												  void* vz, const Nd4jLong* zShapeInfo,
											const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
											const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets,
											const double epsilon) {

    batchnormCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vMean, meanShapeInfo, vVariance, varianceShapeInfo, vGamma, gammaShapeInfo, vBeta, betaShapeInfo, vz, zShapeInfo, xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, static_cast<T>(epsilon));
}
BUILD_SINGLE_TEMPLATE(template void batchnormCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const void* vMean, const Nd4jLong* meanShapeInfo, const void* vVariance, const Nd4jLong* varianceShapeInfo, const void* vGamma, const Nd4jLong* gammaShapeInfo, const void* vBeta, const Nd4jLong* betaShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets, const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets, const double epsilon), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

	std::vector<int> copy = axes;
	if (axes.size() > 1)
        std::sort(copy.begin(), copy.end());

	auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), copy);
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), copy);

    const int threadsPerBlock = MAX_NUM_THREADS;
    const int blocksPerGrid = (mean->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(input->getContext(), "batchnorm");

    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});
    BUILD_SINGLE_SELECTOR(input->dataType(), batchnormCudaLauncher, (blocksPerGrid, threadsPerBlock, input->getContext()->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), mean->getSpecialBuffer(), mean->getSpecialShapeInfo(), variance->getSpecialBuffer(), variance->getSpecialShapeInfo(), gamma ? gamma->getSpecialBuffer() : nullptr, gamma ? gamma->getSpecialShapeInfo() : nullptr, beta ? beta->getSpecialBuffer() : nullptr, beta ? beta->getSpecialShapeInfo() : nullptr, output->specialBuffer(), output->specialShapeInfo(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platformShapeInfo(), packZ.platformOffsets(), epsilon), FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});

    manager.synchronize();
}


}
}
}

