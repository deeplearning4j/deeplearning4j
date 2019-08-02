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
#include <TAD.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {


	template <typename T>
	static __global__ void stackKernel(void** inputList, void** inputShapeList, int inputListLength, Nd4jLong arrLen, void* vz, const Nd4jLong* zShapeInfo, Nd4jLong* tadShape, Nd4jLong *tadOffsets) {

		T* z = reinterpret_cast<T*>(vz);

		if(tadShape == nullptr) {	// scalar case

			for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < inputListLength; i += gridDim.x * blockDim.x)
				z[shape::getIndexOffset(i, zShapeInfo, inputListLength)] = reinterpret_cast<T*>(inputList[i])[0];
		}
		else {

			for (int t = blockIdx.x; t < inputListLength; t += gridDim.x) {

	            auto tZ = z + tadOffsets[t];
			    auto tX = reinterpret_cast<T*>(inputList[t]);
			    auto xShapeInfo = reinterpret_cast<Nd4jLong*>(inputShapeList[t]);

			    for (int e = threadIdx.x; e < arrLen; e += blockDim.x)
			        tZ[shape::getIndexOffset(e, tadShape, arrLen)] = tX[shape::getIndexOffset(e, xShapeInfo, arrLen)];
			}
		}
	}

	///////////////////////////////////////////////////////////////////
	template <typename T>
	static void stack_(nd4j::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray* outArr, const int dim) {

		const bool scalarCase = inArrs[0]->isScalar();

		const int threadsPerBlock = MAX_NUM_THREADS / 2;
		const int blocksPerGrid = scalarCase ? (outArr->lengthOf() + threadsPerBlock - 1) / threadsPerBlock : inArrs.size();

		NDArray::prepareSpecialUse({outArr}, inArrs);

		std::vector<void const*> inputList(inArrs.size());
		std::vector<Nd4jLong const*> inputShapeList(inArrs.size());

		for (size_t i = 0; i < inputList.size(); ++i) {
			inputList[i] = inArrs[i]->getSpecialBuffer();
			inputShapeList[i] = inArrs[i]->getSpecialShapeInfo();
		}

        PointersManager manager(context, "helpers::stack");
        auto dInBuffers = (void **) manager.replicatePointer(inputList.data(), inputList.size() * sizeof(Nd4jLong*));
        auto dInShapeInfo = (void **) manager.replicatePointer(inputShapeList.data(), inputShapeList.size() * sizeof(Nd4jLong*));

        if(scalarCase) {
        	stackKernel<T><<<blocksPerGrid, threadsPerBlock, 1024, *context->getCudaStream()>>>((void**)dInBuffers, (void**)dInShapeInfo, inputList.size(), inArrs[0]->lengthOf(), outArr->specialBuffer(), outArr->getSpecialShapeInfo(), nullptr, nullptr);
        }
        else {
        	std::vector<int> axis = ShapeUtils::evalDimsToExclude(outArr->rankOf(), {dim});
        	auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(outArr->getShapeInfo(), axis);
			stackKernel<T><<<blocksPerGrid, threadsPerBlock, 1024, *context->getCudaStream()>>>((void**)dInBuffers, (void**)dInShapeInfo, inputList.size(), inArrs[0]->lengthOf(), outArr->specialBuffer(), nullptr, packZ.specialShapeInfo(), packZ.specialOffsets());
        }
        manager.synchronize();

        NDArray::registerSpecialUse({outArr}, inArrs);

	}

	void stack(nd4j::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray* outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr->dataType(), stack_, (context, inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (nd4j::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray* outArr, const int dim), LIBND4J_TYPES);

}
}
}

