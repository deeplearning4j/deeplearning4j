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

//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)

#ifndef PAIRWISE_CU
#define PAIRWISE_CU


#include "../pairwise_transform.h"


using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
__global__ static void pairwiseSimpleShaped(void* vx, Nd4jLong *xShapeInfo, 
											void *vy, Nd4jLong *yShapeInfo, 
											void *vz, Nd4jLong *zShapeInfo, 
											void *vextraParams, 
											int *allocationBuffer) {
	
	auto x = reinterpret_cast<X*>(vx);
	auto y = reinterpret_cast<Y*>(vy);
	auto z = reinterpret_cast<Z*>(vz);
	auto extraParams = reinterpret_cast<Z*>(vextraParams);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	Nd4jLong len = shape::length(xShapeInfo);

	auto xEws = shape::elementWiseStride(xShapeInfo);
	auto yEws = shape::elementWiseStride(yShapeInfo);
	auto zEws = shape::elementWiseStride(zShapeInfo);

	auto xOrder = shape::order(xShapeInfo);
	auto yOrder = shape::order(yShapeInfo);
	auto zOrder = shape::order(zShapeInfo);


	if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == yOrder && xOrder == zOrder) {
		for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
			z[i * zEws] = OpType::op(x[i * xEws], y[i * yEws], extraParams);
		}
	} else if (vx == vz) {
		for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
			auto xOffset = shape::getIndexOffset(i, xShapeInfo, len);
			auto yOffset = shape::getIndexOffset(i, yShapeInfo, len);
				
			z[xOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
		}
	} else {
		for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
			auto xOffset = shape::getIndexOffset(i, xShapeInfo, len);
			auto yOffset = shape::getIndexOffset(i, yShapeInfo, len);
			auto zOffset = shape::getIndexOffset(i, zShapeInfo, len);

			z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
		}
	}
}

namespace functions           {
namespace pairwise_transforms {

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void _CUDA_H PairWiseTransform<X,Y,Z>::intermediateShaped(dim3& launchDims, cudaStream_t *stream, 
														void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo,
														void *vy, Nd4jLong *yShapeInfo, Nd4jLong *hyShapeInfo,
														void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo,
														void *vextraParams,
														int *allocPointer){

	pairwiseSimpleShaped<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams, allocPointer);
}

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void PairWiseTransform<X,Y,Z>::executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vy, Nd4jLong *yShapeInfo, Nd4jLong *hyShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void *vextraParams) {
	DISPATCH_BY_OPNUM_TTT(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, hxShapeInfo, vy, yShapeInfo, hyShapeInfo, vz, zShapeInfo, hzShapeInfo, vextraParams, nullptr), PAIRWISE_TRANSFORM_OPS);
}


    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_0);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_1);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_2);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_3);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_4);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_5);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_6);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_7);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_8);
    BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , PAIRWISE_TYPES_9);
}
}

#endif // PAIRWISE_CU