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

SD_LIB_HIDDEN bool segmentIndicesValidate(sd::LaunchContext* context, NDArray* indices, NDArray& expected,
                                          NDArray& output);

SD_LIB_HIDDEN bool unsortedSegmentIndicesValidate(sd::LaunchContext* context, NDArray* indices,
                                                  sd::LongType numOfClasses, sd::LongType& output);

SD_LIB_HIDDEN void segmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentMinFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentMeanFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentSumFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void segmentProdFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentSqrtNFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                               sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentMaxFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                             sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentMinFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                             sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentMeanFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                              sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentSumFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                             sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN void unsortedSegmentProdFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                              sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN sd::Status segmentMaxFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                             NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN sd::Status segmentMinFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                             NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN sd::Status segmentMeanFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                              NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN sd::Status segmentSumFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                             NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN sd::Status segmentProdFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                              NDArray* gradOut, NDArray* output);

SD_LIB_HIDDEN sd::Status unsortedSegmentSqrtNFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                       NDArray* gradOut, sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN sd::Status unsortedSegmentMaxFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                     NDArray* gradOut, sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN sd::Status unsortedSegmentMinFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                     NDArray* gradOut, sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN sd::Status unsortedSegmentMeanFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                      NDArray* gradOut, sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN sd::Status unsortedSegmentSumFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                     NDArray* gradOut, sd::LongType numOfClasses, NDArray* output);

SD_LIB_HIDDEN sd::Status unsortedSegmentProdFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                      NDArray* gradOut, sd::LongType numOfClasses, NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
