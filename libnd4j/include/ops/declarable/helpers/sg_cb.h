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
// @author raver119@gmail.com
//

#ifndef DEV_TESTS_SG_CB_H
#define DEV_TESTS_SG_CB_H
#include <array/NDArray.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

namespace sd {
namespace ops {
namespace helpers {



SD_LIB_HIDDEN void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                            NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes,
                            NDArray &alpha, NDArray &randomValue, NDArray &inferenceVector, const bool preciseMode,
                            const int numWorkers);


SD_LIB_HIDDEN void  skipgramInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                       int ngStarter, int nsRounds, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
                       NDArray &inferenceVector, const bool preciseMode, const int numWorkers);

SD_LIB_HIDDEN void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                        NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords,
                        NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue, NDArray &numLabels,
                        NDArray &inferenceVector, const bool trainWords, const int numWorkers);



SD_LIB_HIDDEN void cbowInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                                 int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
                                 double alpha, sd::LongType randomValue, int numLabels, NDArray &inferenceVector, const bool trainWords,
                                 int numWorkers);

SD_LIB_HIDDEN int binarySearch(const int *haystack, const int needle, const int totalElements);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_SG_CB_H
