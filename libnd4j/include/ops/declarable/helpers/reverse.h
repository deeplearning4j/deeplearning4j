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
// @author Yurii Shyrma, created on 16.04.2018
//

#ifndef LIBND4J_REVERSE_H
#define LIBND4J_REVERSE_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

	template <typename T>
	void reverseArray(T* inArr, Nd4jLong *inShapeBuffer, T *result, Nd4jLong *zShapeBuffer, int numOfElemsToReverse = 0);

	
	template <typename T>
	void reverseSequence(const NDArray<T>* input, const NDArray<T>* seqLengths, NDArray<T>* output, int seqDim, const int batchDim);


	template<typename T>
	void reverse(const NDArray<T>* input, NDArray<T>* output, const std::vector<int>* intArgs, bool isLegacy);

    

}
}
}


#endif //LIBND4J_REVERSESEQUENCE_H
