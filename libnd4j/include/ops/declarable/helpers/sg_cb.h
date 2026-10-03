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
#include <array/NDArrayFactory.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

/**
 * The integer arguments of an *_inference op (its codes, indices, context and locked words) as an array for the helpers:
 * an INT32 vector, or the empty array when there are none, as an array cannot be made of no data. The caller owns it.
 */
inline NDArray *integerArgumentsArray(const std::vector<int> &values) {
  if (values.empty()) return NDArrayFactory::empty(DataType::INT32);

  std::vector<sd::LongType> shape = {static_cast<sd::LongType>(values.size())};
  return NDArrayFactory::create_<int>('c', shape, values, LaunchContext::defaultContext());
}

SD_LIB_HIDDEN void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                            NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes,
                            NDArray &alpha, NDArray &randomValue, NDArray &inferenceVector, const bool preciseMode,
                            const int numWorkers,const int iterations,double minLearningRate);


SD_LIB_HIDDEN void  skipgramInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                       int ngStarter, int nsRounds, NDArray &indices, NDArray &codes, double alpha, LongType randomValue,
                       NDArray &inferenceVector, const bool preciseMode, const int numWorkers,double minLearningRate,const int iterations);

SD_LIB_HIDDEN void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                        NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords,
                        NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue, NDArray &numLabels,
                        NDArray &inferenceVector, const bool trainWords, const int numWorkers,double minLearningRate,const int iterations);



SD_LIB_HIDDEN void cbowInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                                 int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
                                 double alpha, LongType randomValue, int numLabels, NDArray &inferenceVector, const bool trainWords,
                                 int numWorkers,int iterations,double minLearningRate);

SD_LIB_HIDDEN int binarySearch(const int *haystack, const int needle, const int totalElements);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_SG_CB_H
