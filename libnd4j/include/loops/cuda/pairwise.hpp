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
template<typename X, typename Y, typename Z, typename OpType>
__device__ void pairwiseSimpleShapedGeneric(void* x, Nd4jLong *xShapeInfo,
									void *y, Nd4jLong *yShapeInfo, 
									void *z, Nd4jLong *zShapeInfo, 
									void *params, 
									int *allocationBuffer) {
   
    functions::pairwise_transforms::PairWiseTransform<X,Y,Z>::template transformCuda<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, params, allocationBuffer, nullptr, nullptr);
}

template<typename X, typename Y, typename Z, typename OpType>
__device__ void pairwiseSimpleStridedGeneric(Nd4jLong length, void* x, Nd4jLong xEws,
									  void *y, Nd4jLong yEws,
									  void *z, Nd4jLong zEws,
									  void *params,
									  int *allocationBuffer) {

	functions::pairwise_transforms::PairWiseTransform<X,Y,Z>::template transformCuda<OpType>(length, x, y, xEws, yEws,  params, z, zEws, allocationBuffer, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
__global__ void pairwiseSimpleShaped(void* x, Nd4jLong *xShapeInfo, 
									void *y, Nd4jLong *yShapeInfo, 
									void *z, Nd4jLong *zShapeInfo, 
									void *params, 
									int *allocationBuffer) {
        
	pairwiseSimpleShapedGeneric<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, params, allocationBuffer);
}

template <typename X, typename Y, typename Z, typename OpType>
__global__ void pairwiseSimpleStrided(Nd4jLong length, void* x, Nd4jLong xEws,
									 void *y, Nd4jLong yEws,
									 void *z, Nd4jLong zEws,
									 void *params,
									 int *allocationBuffer) {

	pairwiseSimpleStridedGeneric<X, Y, Z, OpType>(length, x, xEws, y, yEws, z, zEws, params, allocationBuffer);
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

    auto length = shape::length(hxShapeInfo);
	auto xEWS = shape::elementWiseStride(hxShapeInfo);
	auto xOrder = shape::order(hxShapeInfo);

	auto yEWS = shape::elementWiseStride(hyShapeInfo);
	auto yOrder = shape::order(hyShapeInfo);

	auto zEWS = shape::elementWiseStride(hzShapeInfo);
	auto zOrder = shape::order(hzShapeInfo);

	if (xEWS >= 1 && zEWS >= 1 && yEWS >= 1 && xOrder == yOrder && xOrder == zOrder) {
		pairwiseSimpleStrided<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(length, vx, xEWS, vy, yEWS, vz, zEWS, vextraParams, allocPointer);
	} else {
		pairwiseSimpleShaped<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams, allocPointer);
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T, typename Y, typename Z>
template<typename OpType>
__device__ void PairWiseTransform<T,Y,Z>::transformCuda(Nd4jLong len,
														void *vx, void *vy,
														Nd4jLong xEws, Nd4jLong yEws,
														void *vparams,
														void *vz, Nd4jLong zEws,
														int *allocPointer, 
														UnifiedSharedMemory *manager,
														Nd4jLong *tadOnlyShapeInfo) {
	auto x = reinterpret_cast<T*>(vx);
	auto y = reinterpret_cast<Y*>(vy);
	auto z = reinterpret_cast<Z*>(vz);
	auto params = reinterpret_cast<Z*>(vparams);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
		z[i * zEws] = OpType::op(x[i * xEws], y[i * yEws], params);
	}

}

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
__device__ void PairWiseTransform<X,Y,Z>::transformCuda(void *vx, Nd4jLong *xShapeInfo, 
														void *vy, Nd4jLong *yShapeInfo, 
														void *vz, Nd4jLong *zShapeInfo, 
														void *vextraParams, 
														int *allocPointer, 
														UnifiedSharedMemory *manager, 
														Nd4jLong *tadOnlyShapeInfo) {

	auto x = reinterpret_cast<X*>(vx);
	auto y = reinterpret_cast<Y*>(vy);
	auto z = reinterpret_cast<Z*>(vz);
	auto extraParams = reinterpret_cast<Z*>(vextraParams);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	Nd4jLong len = shape::length(xShapeInfo);

	if (vx == vz) {
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

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void PairWiseTransform<X,Y,Z>::executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vy, Nd4jLong *yShapeInfo, Nd4jLong *hyShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void *vextraParams) {
	DISPATCH_BY_OPNUM_TTT(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, hxShapeInfo, vy, yShapeInfo, hyShapeInfo, vz, zShapeInfo, hzShapeInfo, vextraParams, nullptr), PAIRWISE_TRANSFORM_OPS);
}
      

}
}

#endif // PAIRWISE_CU