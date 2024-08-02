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
//  @author sgazeos@gmail.com
//  @brief helpers fuctions for segment_* ops (segment_max, segment_min, etc.)
//  @brief helpers fuctions for unsorted_segment_* ops (unsorted_segment_max, etc.)
//
#ifndef __SEGMENT_HELPERS__
#define __SEGMENT_HELPERS__
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN bool segmentIndicesValidate(LaunchContext* context, NDArray* indices, NDArray& expected,
                                          NDArray& output);

SD_LIB_HIDDEN bool unsortedSegmentIndicesValidate(LaunchContext* context, NDArray* indices, LongType numOfClasses,
                                                  LongType& output);

SD_LIB_HIDDEN void segmentMaxFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentMinFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentSumFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentProdFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentSqrtNFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                               LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentMaxFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                             LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentMinFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                             LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                              LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentSumFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                             LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentProdFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                              LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN Status segmentMaxFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                             NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN Status segmentMinFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                             NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN Status segmentMeanFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                              NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN Status segmentSumFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                             NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN Status segmentProdFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                              NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN Status unsortedSegmentSqrtNFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                                       NDArray* gradOut, LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN Status unsortedSegmentMaxFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                                     NDArray* gradOut, LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN Status unsortedSegmentMinFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                                     NDArray* gradOut, LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN Status unsortedSegmentMeanFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                                      NDArray* gradOut, LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN Status unsortedSegmentSumFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                                     NDArray* gradOut, LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN Status unsortedSegmentProdFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                                      NDArray* gradOut, LongType numOfClasses, NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
