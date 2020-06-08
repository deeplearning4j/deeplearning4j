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
#include <helpers/OmpLaunchHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// template<typename T>
// __global__ static void batchnormCuda(const void* vx, const Nd4jLong* xShapeInfo,
// 									const void* vMean, const Nd4jLong* meanShapeInfo,
// 									const void* vVariance, const Nd4jLong* varianceShapeInfo,
// 									const void* vGamma, const Nd4jLong* gammaShapeInfo,
// 									const void* vBeta, const Nd4jLong* betaShapeInfo,
// 										  void* vz, const Nd4jLong* zShapeInfo,
// 									const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
// 									const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets,
// 									const T epsilon) {

// 	const auto x    	= reinterpret_cast<const T*>(vx);
//           auto z        = reinterpret_cast<T*>(vz);
// 	const auto mean 	= reinterpret_cast<const T*>(vMean);
// 	const auto variance = reinterpret_cast<const T*>(vVariance);
// 	const auto gamma    = reinterpret_cast<const T*>(vGamma);
// 	const auto beta     = reinterpret_cast<const T*>(vBeta);

//     // maxRank = xRank = zRank, minRank = meanRank = varianceRank = gammaRank = betaRank
//     __shared__ Nd4jLong minLen, tadLen, totalThreads;

//     if (threadIdx.x == 0) {
//         totalThreads = gridDim.x * blockDim.x;

//         minLen = shape::length(meanShapeInfo);
//         tadLen = shape::length(xShapeInfo) / minLen;
//     }
//     __syncthreads();

//     const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

//     for (uint i = tid; i < minLen; i += totalThreads) {

// 		const auto meanOffset     = shape::getIndexOffset(i, meanShapeInfo);
//     	const auto varianceOffset = shape::getIndexOffset(i, varianceShapeInfo);

//     	T sigmaInvGam = 1. / sd::math::nd4j_sqrt<T, T>(variance[varianceOffset] + epsilon);

//     	if(gamma != nullptr)
//     		sigmaInvGam *= gamma[shape::getIndexOffset(i, gammaShapeInfo)];

// 		auto betaOffset = 0;
//     	if(beta != nullptr)
//     		betaOffset = shape::getIndexOffset(i, betaShapeInfo);

//     	const auto xTad = x + xTadOffsets[i];
//     		  auto zTad = z + zTadOffsets[i];

//     	for (uint j = 0; j < tadLen; ++j) {

//     		const auto xTadOffset = shape::getIndexOffset(j, xTadShapeInfo);
//     		const auto zTadOffset = shape::getIndexOffset(j, zTadShapeInfo);

//     		zTad[zTadOffset] = (xTad[xTadOffset] - mean[meanOffset]) * sigmaInvGam;

//     		if(beta != nullptr)
// 				zTad[zTadOffset] += beta[betaOffset];
//     	}
//     }
// }

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void batchnormCuda2(const void* vx, const Nd4jLong* xShapeInfo,
                                    const void* vMean, const Nd4jLong* meanShapeInfo,
                                    const void* vVariance, const Nd4jLong* varianceShapeInfo,
                                    const void* vGamma, const Nd4jLong* gammaShapeInfo,
                                    const void* vBeta, const Nd4jLong* betaShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo,
                                    const int numDims, const int* dims,
                                    const T epsilon) {

    const auto x        = reinterpret_cast<const T*>(vx);
          auto z        = reinterpret_cast<T*>(vz);
    const auto mean     = reinterpret_cast<const T*>(vMean);
    const auto variance = reinterpret_cast<const T*>(vVariance);
    const auto gamma    = reinterpret_cast<const T*>(vGamma);
    const auto beta     = reinterpret_cast<const T*>(vBeta);

    __shared__ int xRank, minRank;          // xRank == zRank, minRank = meanRank = varianceRank = gammaRank = betaRank
    __shared__ Nd4jLong xLen, totalThreads; // xLen = zLen


    if (threadIdx.x == 0) {

        totalThreads = gridDim.x * blockDim.x;

        xLen    = shape::length(xShapeInfo);
        xRank   = shape::rank(xShapeInfo);
        minRank = shape::rank(meanShapeInfo);
    }
    __syncthreads();

    int coords[MAX_RANK];

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = tid; i < xLen; i += totalThreads) {

        shape::index2coords(i, xShapeInfo, coords);

        const auto xOffset = shape::getOffset(xShapeInfo, coords);
        const auto zOffset = shape::getOffset(zShapeInfo, coords);

        if(minRank == xRank) {
            for (uint i = 0, j = 0; i < xRank; ++i) {
                if(j < numDims && i != dims[j])
                    coords[i] = 0;
                else
                    ++j;
            }
        }
        else    // minRank = numDims = 1 in this case
            coords[0] = coords[dims[0]];

        const auto meanOffset     = shape::getOffset(meanShapeInfo, coords);
        const auto varianceOffset = shape::getOffset(varianceShapeInfo, coords);

        T sigmaInvGam = 1. / sd::math::nd4j_sqrt<T, T>(variance[varianceOffset] + epsilon);

        if(gamma != nullptr) {
            const auto gammaOffset = shape::getOffset(gammaShapeInfo, coords);
            sigmaInvGam *= gamma[gammaOffset];
        }

        z[zOffset] = (x[xOffset] - mean[meanOffset]) * sigmaInvGam;

        if(beta != nullptr) {
            const auto betaOffset = shape::getOffset(betaShapeInfo, coords);
            z[zOffset] += beta[betaOffset];
        }
    }
}

///////////////////////////////////////////////////////////////////
// template<typename T>
// __host__ static void batchnormCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
// 											const void* vx, const Nd4jLong* xShapeInfo,
//                                            	const void* vMean, const Nd4jLong* meanShapeInfo,
// 											const void* vVariance, const Nd4jLong* varianceShapeInfo,
// 											const void* vGamma, const Nd4jLong* gammaShapeInfo,
// 											const void* vBeta, const Nd4jLong* betaShapeInfo,
// 												  void* vz, const Nd4jLong* zShapeInfo,
// 											const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
// 											const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets,
// 											const double epsilon) {

//     batchnormCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vMean, meanShapeInfo, vVariance, varianceShapeInfo, vGamma, gammaShapeInfo, vBeta, betaShapeInfo, vz, zShapeInfo, xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, static_cast<T>(epsilon));
// }

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void batchnormCudaLauncher2(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                            const void* vx, const Nd4jLong* xShapeInfo,
                                            const void* vMean, const Nd4jLong* meanShapeInfo,
                                            const void* vVariance, const Nd4jLong* varianceShapeInfo,
                                            const void* vGamma, const Nd4jLong* gammaShapeInfo,
                                            const void* vBeta, const Nd4jLong* betaShapeInfo,
                                                  void* vz, const Nd4jLong* zShapeInfo,
                                            const int numDims, const int* dims,
                                            const double epsilon) {

    batchnormCuda2<T><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(vx, xShapeInfo, vMean, meanShapeInfo, vVariance, varianceShapeInfo, vGamma, gammaShapeInfo, vBeta, betaShapeInfo, vz, zShapeInfo, numDims, dims, static_cast<T>(epsilon));
}

//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

	// std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), axes);

	// auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimsToExclude);
 //    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimsToExclude);

 //    const int threadsPerBlock = MAX_NUM_THREADS / 2;
 //    const int blocksPerGrid = (mean->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

 //    PointersManager manager(input->getContext(), "batchnorm");

 //    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});
 //    BUILD_SINGLE_SELECTOR(input->dataType(), batchnormCudaLauncher, (blocksPerGrid, threadsPerBlock, input->getContext()->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), mean->specialBuffer(), mean->specialShapeInfo(), variance->specialBuffer(), variance->specialShapeInfo(), gamma ? gamma->specialBuffer() : nullptr, gamma ? gamma->specialShapeInfo() : nullptr, beta ? beta->specialBuffer() : nullptr, beta ? beta->specialShapeInfo() : nullptr, output->specialBuffer(), output->special(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platform(), packZ.platform(), epsilon), FLOAT_TYPES);
 //    NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});

 //    manager.synchronize();


    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (input->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(input->getContext(), "batchnorm");

    const int* dims = reinterpret_cast<int*>(manager.replicatePointer(axes.data(), axes.size() * sizeof(int)));

    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});
    BUILD_SINGLE_SELECTOR(input->dataType(), batchnormCudaLauncher2, (blocksPerGrid, threadsPerBlock, input->getContext()->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), mean->specialBuffer(), mean->specialShapeInfo(), variance->specialBuffer(), variance->specialShapeInfo(), gamma ? gamma->specialBuffer() : nullptr, gamma ? gamma->specialShapeInfo() : nullptr, beta ? beta->specialBuffer() : nullptr, beta ? beta->specialShapeInfo() : nullptr, output->specialBuffer(), output->specialShapeInfo(), axes.size(), dims, epsilon), FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});

    manager.synchronize();
}


}
}
}

