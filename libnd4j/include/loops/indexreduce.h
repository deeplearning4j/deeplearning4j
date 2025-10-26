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

/*
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include <helpers/DebugHelper.h>
#include <helpers/OmpLaunchHelper.h>

#include <helpers/shape.h>
#include <loops/legacy_ops.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>

namespace functions {
namespace indexreduce {

template <typename X, typename Z>
class IndexReduce {
 public:
#ifdef SD_CUDA

  static SD_DEVICE void transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                  void *result, const sd::LongType *resultShapeInfo, sd::LongType *dimension,
                                  sd::LongType dimensionLength, int postProcessOrNot, sd::LongType *allocationBuffer,
                                  void *reductionBuffer, const sd::LongType *tadShapeInfo,
                                  const sd::LongType *tadOffset);

  template <typename OpType>
  static SD_DEVICE void aggregatePartials(IndexValue<X> *sPartialsRef, sd::LongType tid, sd::LongType numElements,
                                          void *extraParams);

  template <typename OpType>
  static SD_DEVICE void transform(const void *dx, const sd::LongType *xShapeInfo, void *extraParams, void *result,
                                  const sd::LongType *resultShapeInfo, sd::LongType *dimension,
                                  sd::LongType dimensionLength,
                                  int postProcessOrNot, sd::LongType *allocationBuffer, void *reductionBuffer,
                                  const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets);

  static SD_HOST void executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int op, const void *dx,
                                               const sd::LongType *xShapeInfo, sd::LongType xRank, void *extraParams,
                                               void *result, const sd::LongType *resultShapeInfo, sd::LongType zRank,
                                               sd::LongType *dimension,sd::LongType dimensionLength, int postProcessOrNot,
                                               sd::LongType *allocationBuffer, void *reductionBuffer,
                                               const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets);

  static SD_HOST void executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int op, const void *dx,
                                         const sd::LongType *xShapeInfo, sd::LongType xRank, void *extraParams, void *result,
                                         const sd::LongType *resultShapeInfo,sd::LongType zRank,
                                         sd::LongType *dimension, sd::LongType dimensionLength, int postProcessOrNot,
                                         sd::LongType *allocationBuffer,
                                         void *reductionBuffer, const sd::LongType *tadOnlyShapeInfo,
                                         const sd::LongType *tadOffsets);
#else

  static sd::LongType execScalar(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams);

  static void exec(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *result,
                   const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength,
                   const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset);

  template <typename OpType>
  static SD_HOST sd::LongType execScalar(const void *x, const sd::LongType *xShapeInfo, void *extraParams);

  template <typename OpType>
  static SD_HOST void exec(const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *result,
                           const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension,
                           sd::LongType dimensionLength,
                           const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset);
#endif
};
}  // namespace indexreduce
}  // namespace functions

#endif /* INDEXREDUCE_H_ */
