/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//  @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018

#ifndef PAIRWISE_BOOL_CU
#define PAIRWISE_BOOL_CU


#include "../pairwise_bool.h"


using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
__global__ static void pairwiseSimpleShaped(void const* vx, Nd4jLong const* xShapeInfo,
											void const* vy, Nd4jLong const* yShapeInfo,
											void *vz, Nd4jLong const* zShapeInfo,
											void *vextraParams) {

	auto x = reinterpret_cast<X const*>(vx);
	auto y = reinterpret_cast<X const*>(vy);
	auto z = reinterpret_cast<Z*>(vz);
	auto extraParams = reinterpret_cast<X*>(vextraParams);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int xEws;
	__shared__ int yEws;
	__shared__ int zEws;
	__shared__ char xOrder;
	__shared__ char yOrder;
	__shared__ char zOrder;
	__shared__ Nd4jLong len;

	if (threadIdx.x == 0) {
		xEws = shape::elementWiseStride(xShapeInfo);
		yEws = shape::elementWiseStride(yShapeInfo);
    	zEws = shape::elementWiseStride(zShapeInfo);
		xOrder = shape::order(xShapeInfo);
		yOrder = shape::order(yShapeInfo);
		zOrder = shape::order(zShapeInfo);
		len = shape::length(xShapeInfo);
	}
	__syncthreads();


	if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == yOrder && xOrder == zOrder) {
		for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
			z[i * zEws] = OpType::op(x[i * xEws], y[i * yEws], extraParams);
		}
	}
	else if (vx == vz) {
		for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
			auto xOffset = shape::getIndexOffset(i, xShapeInfo);
			auto yOffset = shape::getIndexOffset(i, yShapeInfo);

			z[xOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
		}
	}
	else {
		for (Nd4jLong i = tid; i < len; i += gridDim.x * blockDim.x) {
			auto xOffset = shape::getIndexOffset(i, xShapeInfo);
			auto yOffset = shape::getIndexOffset(i, yShapeInfo);
			auto zOffset = shape::getIndexOffset(i, zShapeInfo);

			z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
		}
	}
}


namespace functions           {
namespace pairwise_transforms {

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Z>
template<typename OpType>
void _CUDA_H PairWiseBoolTransform<X,Z>::intermediateShaped(dim3& launchDims, cudaStream_t *stream,
														void const* vx, Nd4jLong const* xShapeInfo,
														void const* vy, Nd4jLong const* yShapeInfo,
														void *vz, Nd4jLong const* zShapeInfo,
														void *vextraParams){

	pairwiseSimpleShaped<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams);
}


////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
void PairWiseBoolTransform<X,Y>::executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void const* vx, Nd4jLong const* xShapeInfo, void const* vy, Nd4jLong const* yShapeInfo, void *vz, Nd4jLong const* zShapeInfo, void *vextraParams) {
    auto xType = sd::DataTypeUtils::fromT<X>();
    auto yType = sd::DataTypeUtils::fromT<Y>();

	DISPATCH_BY_OPNUM_TT(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams), PAIRWISE_BOOL_OPS);
}

    BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT PairWiseBoolTransform, , LIBND4J_TYPES, BOOL_TYPES);
}
}

#endif // PAIRWISE_BOOL_CU