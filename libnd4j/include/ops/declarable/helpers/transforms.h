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
#include <helpers/helper_random.h>
#include <graph/RandomGenerator.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

	void triuBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal);

	void trace(nd4j::LaunchContext * context, const NDArray& input, NDArray& output);

	void randomShuffle(nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::graph::RandomGenerator& rng, const bool isInplace);

    // auxiliary function which serves for recursion purpose and is used in pad operation
	// void recursiveLoopForPad(const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue);

	void pad(nd4j::LaunchContext * context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue);

	void invertPermutation(nd4j::LaunchContext * context, const NDArray& input, NDArray& output);

	void gatherND(nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output);

	void gather(nd4j::LaunchContext * context, NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs);

	void eye(nd4j::LaunchContext * context, NDArray& output);

	void scatterUpdate(nd4j::LaunchContext * context, NDArray& operand, NDArray& updates, const std::vector<int>* intArgs);

	void scatterSimple(nd4j::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions);

	void mergeMaxIndex(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output);

	void mergeMax(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output);

	void mergeAvg(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output);

	void mergeAdd(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output);

	void clipByNorm(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace);
	void clipByGlobalNorm(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace);

	void clipByNormBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm);

	void clipByAveraged(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace);
	void clipByValue(nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output);

	void mirrorPad(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode);

	void concat(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis);

	void tileBP(nd4j::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps);

	void split(nd4j::LaunchContext* context, const NDArray& input, std::vector<NDArray*>& outArrs, const int axis);
}
}
}


#endif //LIBND4J_TRANSFORMS_H
