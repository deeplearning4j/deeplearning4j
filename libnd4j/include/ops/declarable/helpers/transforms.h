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

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#ifndef LIBND4J_TRANSFORMS_H
#define LIBND4J_TRANSFORMS_H
#include <graph/Context.h>
#include <graph/RandomGenerator.h>
#include <helpers/helper_random.h>
#include <ops/declarable/helpers/helpers.h>
namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void triuBP(LaunchContext* context, NDArray& input, NDArray& gradO, NDArray& gradI,
                          const int diagonal);

SD_LIB_HIDDEN void trace(LaunchContext* context, NDArray& input, NDArray& output);

SD_LIB_HIDDEN void randomShuffle(LaunchContext* context, NDArray& input, NDArray& output, graph::RandomGenerator& rng, const bool isInplace);

// auxiliary function which serves for recursion purpose and is used in pad operation
// void recursiveLoopForPad(const int mode, NDArray& input, NDArray& paddings, NDArray& output, std::vector<int>
// dimensions, int dim, int inIdx, int outIdx, NDArray& padValue);

SD_LIB_HIDDEN void pad(LaunchContext* context, const int mode, NDArray& input, NDArray& paddings,
                       NDArray& output, NDArray& padValue);

SD_LIB_HIDDEN void invertPermutation(LaunchContext* context, NDArray& input, NDArray& output);

SD_LIB_HIDDEN void gatherND(LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output);

SD_LIB_HIDDEN void gather(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output,
                          const std::vector<int>& intArgs);

SD_LIB_HIDDEN void eye(LaunchContext* context, NDArray& output);

SD_LIB_HIDDEN void scatterUpdate(LaunchContext* context, NDArray& operand, NDArray& updates,
                                 const std::vector<LongType>* intArgs);

SD_LIB_HIDDEN void scatterSimple(LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                                 NDArray& indices, const std::vector<LongType>& dimensions);

SD_LIB_HIDDEN void mergeMaxIndex(LaunchContext* context, const std::vector<NDArray*>& inArrs,
                                 NDArray& output);

SD_LIB_HIDDEN void mergeMax(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output);
SD_LIB_HIDDEN void mergeMaxBp(LaunchContext* context, const std::vector<NDArray*>& inArrs,
                              std::vector<NDArray*>& outArrs);

SD_LIB_HIDDEN void mergeAvg(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output);
SD_LIB_HIDDEN void mergeAvgBp(LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs);

SD_LIB_HIDDEN void mergeAdd(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output);
SD_LIB_HIDDEN void mergeAddBp(LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs);

SD_LIB_HIDDEN void clipByNorm(LaunchContext* context, NDArray& input, NDArray& output,
                              const std::vector<LongType>& dimensions, NDArray& clipNorm, const bool isInplace,
                              const bool useAverage);

SD_LIB_HIDDEN void clipByGlobalNorm(LaunchContext* context, std::vector<NDArray*>& inputs, double clipNorm,
                                    memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace);

SD_LIB_HIDDEN void clipByNormBp(LaunchContext* context, NDArray& input, NDArray& gradO,
                                NDArray& gradI /*output*/, const std::vector<LongType>& dimensions, NDArray& clipNorm,
                                const bool useAverage);

SD_LIB_HIDDEN void clipByAveragedNorm(LaunchContext* context, NDArray& input, NDArray& output,
                                      const std::vector<LongType>& dimensions, NDArray& clipNorm,
                                      const bool isInplace);

SD_LIB_HIDDEN void mirrorPad(LaunchContext* context, NDArray& input, NDArray& paddings, NDArray& output,
                             const int mode);

SD_LIB_HIDDEN void clipByValue(LaunchContext* context, NDArray& input, double leftBound, double rightBound,
                               NDArray& output);

SD_LIB_HIDDEN void mirrorPad(LaunchContext* context, NDArray& input, NDArray& paddings, NDArray& output,
                             const int mode);

SD_LIB_HIDDEN void concat(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output,
                          const int axis);

SD_LIB_HIDDEN void tileBP(LaunchContext* context, NDArray gradO /*input*/, NDArray& gradI /*output*/,
                          const std::vector<LongType> reps);

SD_LIB_HIDDEN void split(LaunchContext* context, NDArray& input, std::vector<NDArray*>& outArrs,
                         const LongType axis);

SD_LIB_HIDDEN void compareAndBitpack(graph::Context& block, NDArray& input, NDArray& threshold,
                                     NDArray& output);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_TRANSFORMS_H
