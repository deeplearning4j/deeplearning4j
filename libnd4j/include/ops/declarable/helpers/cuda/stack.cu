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
	static __global__ void stackKernel(void** inputList, void** inputShapeList, int inputListLength, Nd4jLong arrLen, void* outputBuffer, Nd4jLong* tadShape, Nd4jLong *tadOffsets) {  //, Nd4jLong* tadShape, Nd4jLong* tadOffsets) {

		__shared__ int arrIdx, blocksPerArr;
		__shared__ T *z;
		__shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLenPerBlock, start, end, offsetZ, zLength;

		if (threadIdx.x == 0) {
            z = reinterpret_cast<T*>(outputBuffer);
		}

		__syncthreads();

		for (int t = blockIdx.x; t < inputListLength; t += gridDim.x) {
            auto tZ = z + tadOffsets[t];
		    auto tX = reinterpret_cast<T*>(inputList[t]);
		    auto xShape = reinterpret_cast<Nd4jLong*>(inputShapeList[t]);

		    for (int e = threadIdx.x; e < arrLen; e += blockDim.x) {
		        tZ[shape::getIndexOffset(e, tadShape, arrLen)] = tX[shape::getIndexOffset(e, xShape, arrLen)];
            }
		}
	}
	///////////////////////////////////////////////////////////////////
	template <typename T>
	static void stack_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray* outArr, const int dim) {
		if(inArrs[0]->isScalar()) {

//#pragma omp parallel for
			for (size_t i = 0; i < inArrs.size(); ++i) {
                inArrs[i]->syncToHost();

                outArr->p(i, inArrs[i]->e<T>(0));
            }
			outArr->syncToDevice();
		}
		else {
			//Nd4jLong **dInShapeInfo;
			//void **dInBuffers;
			std::vector<void const*> inputList(inArrs.size());
			std::vector<Nd4jLong const*> inputShapeList(inArrs.size());
			auto stream = context->getCudaStream();

			for (size_t i = 0; i < inputList.size(); ++i) {
				inputList[i] = inArrs[i]->getSpecialBuffer();
				inputShapeList[i] = inArrs[i]->getSpecialShapeInfo();
			}

            std::vector<int> axis = ShapeUtils::evalDimsToExclude(outArr->rankOf(), {dim});


            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(outArr->getShapeInfo(), axis);


            PointersManager manager(context, "helpers::stack");
            auto dInBuffers = (void **) manager.replicatePointer(inputList.data(), inputList.size() * sizeof(Nd4jLong*));
            auto dInShapeInfo = (void **) manager.replicatePointer(inputShapeList.data(), inputShapeList.size() * sizeof(Nd4jLong*));

            dim3 launchDims(inArrs.size(), inArrs[0]->lengthOf(), 1024);

			stackKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>((void**)dInBuffers, (void**)dInShapeInfo, inputList.size(), inArrs[0]->lengthOf(), outArr->specialBuffer(), packX.specialShapeInfo(), packX.specialOffsets()); //, dTadShape, dTadOffsets);
            manager.synchronize();
		}
	}

	void stack(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray* outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr->dataType(), stack_, (context, inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray* outArr, const int dim), LIBND4J_TYPES);

}
}
}

