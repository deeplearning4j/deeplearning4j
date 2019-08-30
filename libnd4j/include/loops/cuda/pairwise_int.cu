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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018

#ifndef PAIRWISE_INT_CU
#define PAIRWISE_INT_CU


#include "../pairwise_int.h"


using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
__global__ static void pairwiseSimpleShaped(void* vx, Nd4jLong *xShapeInfo,
											void *vy, Nd4jLong *yShapeInfo,
											void *vz, Nd4jLong *zShapeInfo,
											void *vextraParams) {

	auto x = reinterpret_cast<X*>(vx);
	auto y = reinterpret_cast<X*>(vy);
	auto z = reinterpret_cast<X*>(vz);
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
			auto xOffset = shape::getIndexOffset(i, xShapeInfo, len);
			auto yOffset = shape::getIndexOffset(i, yShapeInfo, len);
				
			z[xOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
		}
	}
	else {
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
template<typename X>
template<typename OpType>
void _CUDA_H PairWiseIntTransform<X>::intermediateShaped(dim3& launchDims, cudaStream_t *stream,
														void *vx, Nd4jLong *xShapeInfo,
														void *vy, Nd4jLong *yShapeInfo,
														void *vz, Nd4jLong *zShapeInfo,
														void *vextraParams){

	pairwiseSimpleShaped<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams);
}


////////////////////////////////////////////////////////////////////////////////
template<typename X>
void PairWiseIntTransform<X>::executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, void *vextraParams) {
    auto xType = nd4j::DataTypeUtils::fromT<X>();

	DISPATCH_BY_OPNUM_T(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams), PAIRWISE_INT_OPS);
}


    template<typename X>
    void PairWiseIntTransform<X>::exec(
            const int opNum,
            void *dx,
            Nd4jLong *xShapeBuffer,
            void *y,
            Nd4jLong *yShapeBuffer,
            void *result,
            Nd4jLong *resultShapeBuffer,
            void *extraParams) {

    }

    template<typename X>
    void PairWiseIntTransform<X>::exec(
            const int opNum,
            void *dx,
            Nd4jLong xStride,
            void *y,
            Nd4jLong yStride,
            void *result,
            Nd4jLong resultStride,
            void *extraParams,
            Nd4jLong n) {

    }


    template<typename X>
    template<typename OpType>
    void PairWiseIntTransform<X>::exec(
            void *vx,
            Nd4jLong* xShapeBuffer,
            void *vy,
            Nd4jLong* yShapeBuffer,
            void *vresult,
            Nd4jLong* resultShapeBuffer,
            void *vextraParams) {

    }

    template<typename X>
    template<typename OpType>
    void PairWiseIntTransform<X>::exec(void *vx,
                     Nd4jLong xStride,
                     void *vy,
                     Nd4jLong yStride,
                     void *vresult,
                     Nd4jLong resultStride,
                     void *vextraParams,
                     const Nd4jLong n) {

    }


      
    BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT PairWiseIntTransform, , INTEGER_TYPES);
}
}

#endif // PAIRWISE_INT_CU