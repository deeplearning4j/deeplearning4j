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

	void triu(const NDArray& input, NDArray& output, const int diagonal);


	void triuBP(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal);

	void trace(const NDArray& input, NDArray& output);

	void randomShuffle(NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
    
    // auxiliary function which serves for recursion purpose and is used in pad operation
	void recursiveLoopForPad(const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, T padValue);

	void invertPermutation(const NDArray& input, NDArray& output);

	void gatherND(NDArray& input, NDArray& indices, NDArray& output);

	void gather(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs);

	void eye(NDArray& output);

	void scatterUpdate(NDArray& operand, NDArray& updates, const std::vector<int>* intArgs);

	void mergeMaxIndex(const std::vector<NDArray*>& inArrs, NDArray& output);

	void mergeMax(const std::vector<NDArray*>& inArrs, NDArray& output);

	void mergeAvg(const std::vector<NDArray*>& inArrs, NDArray& output);

	void mergeAdd(const std::vector<NDArray*>& inArrs, NDArray& output);

	void clipByNorm(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace);

	void clipByNormBP(const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const T clipNorm);

	void clipByAveraged(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace);

	void mirrorPad(const NDArray& input, const NDArray& paddings, NDArray& output, const int mode);

	void concat(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis);

	void tileBP(const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps);

}
}
}


#endif //LIBND4J_TRANSFORMS_H
