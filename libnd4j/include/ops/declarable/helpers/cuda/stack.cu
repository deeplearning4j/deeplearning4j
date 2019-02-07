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
// Created by Yurii Shyrma on 02.01.2018
//

#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {

//	Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
//	Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets

	template <typename T>
	static __global__ void stackKernelScalar(void const* inputList[], void* outputBuffer, Nd4jLong* outputShape, Nd4jLong outputLength) {
		auto tid = blockIdx.x * blockDim.x + threadIdx.x;
		int totalThreads = gridDim.x * blockDim.x;
		//const auto resultLength = shape::length(outputShape);
		for (Nd4jLong i = tid; i < outputLength; i += totalThreads) {
			//auto yOffset = shape::subArrayOffset(i, outputShape, inputShape);
			//printf(">> %lld\n", i);
			auto xOffset = shape::getIndexOffset(i, outputShape, outputLength);
			printf(">> %lld\n", xOffset);

			//*(reinterpret_cast<T *>(outputBuffer) + xOffset) = *(reinterpret_cast<T const *>(inputList[xOffset]));
		}
	}

	template <typename T>
	static __global__ void stackKernel(void const* inputList[], Nd4jLong const* inputShapeList[], size_t inputListLength, void* outputBuffer, Nd4jLong* outputShape) {

	}
	///////////////////////////////////////////////////////////////////
	template <typename T>
	static void stack_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim) {
		auto stream = context->getCudaStream();
		dim3 launchDims(256, 512, 8192);
		if(inArrs[0]->isScalar()) {
			std::vector<void const*> scalarList(inArrs.size());
			for (size_t i = 0; i < scalarList.size(); ++i)
				scalarList[i] = inArrs[i]->getSpecialBuffer();
			stackKernelScalar<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(scalarList.data(), outArr.specialBuffer(), outArr.specialShapeInfo(), outArr.lengthOf());
		}
		else {
			std::vector<void const*> inputList(inArrs.size());
			std::vector<Nd4jLong const*> inputShapeList(inArrs.size());
			for (size_t i = 0; i < inputList.size(); ++i) {
				inputList[i] = inArrs[i]->getSpecialBuffer();
				inputShapeList[i] = inArrs[i]->getSpecialShapeInfo();
			}

			stackKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputList.data(), inputShapeList.data(), inputList.size(), outArr.specialBuffer(), outArr.specialShapeInfo());

		}
		auto res = cudaStreamSynchronize(*stream);
		if (res != 0)
			throw cuda_exception::build("stack: Failed to continue due to some previous kernel failre", res);
//
//#pragma omp parallel for if(inArrs.size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
//			for(int i=0; i < inArrs.size(); ++i)
//				outArr.p(i, inArrs[i]->e<T>(0));
//		}
//		else {
//
//			std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(outArr.rankOf(), {dim});
//			auto list = outArr.allTensorsAlongDimension(dimsToExclude);		// list.size() == block.width()
//
//#pragma omp parallel for if(list->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
//			for(int i=0; i<list->size(); ++i)
//				list->at(i)->assign(inArrs[i]);
//
//			delete list;
//		}
	}

	void stack(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr.dataType(), stack_, (context, inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim), LIBND4J_TYPES);

}
}
}

