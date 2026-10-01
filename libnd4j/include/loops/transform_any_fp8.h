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
// TransformAny between the FP8 storage types and the other storage types.
//

#ifndef TRANSFORM_ANY_FP8_H_
#define TRANSFORM_ANY_FP8_H_
#include <loops/transform_any.h>

namespace functions {
namespace transform {

// TransformAny<T, F8> converts into the FP8 storage type F8 and TransformAny<F8, T>
// out of it. FP8 stays out of the SD_COMMON_TYPES matrix, so these pairs are
// instantiated per FP8 encoding, with T over the common types and the FP8
// encodings. T is the only free parameter, so the single-type selectors dispatch
// the pairs over the central type lists.
template <typename F8>
class TransformAnyFp8 {
 public:
#ifdef __CUDACC__
  // x holds T and z holds F8.
  template <typename T>
  static SD_HOST void executeToFp8(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x,
                                   const sd::LongType *xShape, sd::LongType xRank, void *extraParams, void *z,
                                   const sd::LongType *zShape, sd::LongType zRank);

  // x holds F8 and z holds T.
  template <typename T>
  static SD_HOST void executeFromFp8(dim3 launchDims, cudaStream_t *stream, int opNum, const void *x,
                                     const sd::LongType *xShape, sd::LongType xRank, void *extraParams, void *z,
                                     const sd::LongType *zShape, sd::LongType zRank);
#else
  // x holds T and z holds F8.
  template <typename T>
  static void execToFp8(int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                        const sd::LongType *zShapeInfo, void *extraParams, sd::LongType threadId,
                        sd::LongType numThreads);

  // x holds F8 and z holds T.
  template <typename T>
  static void execFromFp8(int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                          const sd::LongType *zShapeInfo, void *extraParams, sd::LongType threadId,
                          sd::LongType numThreads);
#endif
};

}  // namespace transform
}  // namespace functions

#endif /* TRANSFORM_ANY_FP8_H_ */
