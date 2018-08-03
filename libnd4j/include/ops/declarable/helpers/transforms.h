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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#ifndef LIBND4J_TRANSFORMS_H
#define LIBND4J_TRANSFORMS_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void triu(const NDArray<T>& input, NDArray<T>& output, const int diagonal);

	template <typename T>
	void triuBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI, const int diagonal);

	template <typename T>
	void trace(const NDArray<T>& input, NDArray<T>& output);

	template <typename T>
	void randomShuffle(NDArray<T>& input, NDArray<T>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
    
    // auxiliary function which serves for recursion purpose and is used in pad operation
	template<typename T>
	void recursiveLoopForPad(const int mode, NDArray<T>& input, const NDArray<T>& paddings, NDArray<T>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx);

	template<typename T>
	void invertPermutation(const NDArray<T>& input, NDArray<T>& output);

	template<typename T>
	void gatherND(NDArray<T>& input, NDArray<T>& indices, NDArray<T>& output);

	template<typename T>
	void gather(NDArray<T>* input, const NDArray<T>* indices, NDArray<T>* output, const std::vector<int>& intArgs);

	template<typename T>
	void eye(NDArray<T>& output);

	template<typename T>
	void scatterUpdate(NDArray<T>& operand, NDArray<T>& updates, const std::vector<int>* intArgs);

	template<typename T>
	void mergeMaxIndex(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output);

	template<typename T>
	void mergeMax(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output);

	template<typename T>
	void mergeAvg(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output);

	template<typename T>
	void mergeAdd(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output);

	template<typename T>
	void clipByNorm(NDArray<T>& input, NDArray<T>& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace);

	template<typename T>
	void clipByNormBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI /*output*/, const std::vector<int>& dimensions, const T clipNorm); 

	template<typename T>
	void clipByAveraged(NDArray<T>& input, NDArray<T>& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace);

	template<typename T>
	void mirrorPad(const NDArray<T>& input, const NDArray<T>& paddings, NDArray<T>& output, const int mode);

	template<typename T>
	void concat(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output, const int axis);

	template<typename T>
	void tileBP(const NDArray<T>& gradO /*input*/, NDArray<T>& gradI /*output*/, const std::vector<Nd4jLong> reps);

}
}
}


#endif //LIBND4J_TRANSFORMS_H
