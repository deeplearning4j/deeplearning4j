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

#ifndef LIBND4J_RANDOM_H
#define LIBND4J_RANDOM_H

#include <helpers/helper_random.h>
#include <helpers/shape.h>
#include <loops/legacy_ops.h>
#include <ops/random_ops.h>
#include <ops/special_random_ops.h>

namespace functions {
namespace random {

template <typename X>
class RandomFunction {
 public:
#ifdef SD_CUDA
  template <typename OpClass>
  static SD_DEVICE void execTransformCuda(sd::Pointer state, const void *x, const sd::LongType *xShapeBuffer,
                                          const void *y, const sd::LongType *yShapeBuffer, void *z,
                                          const sd::LongType *zShapeBuffer, void *extraArguments);

  template <typename OpClass>
  static SD_DEVICE void execTransformCuda(sd::Pointer state, const void *x, const sd::LongType *xShapeBuffer, void *z,
                                          const sd::LongType *zShapeBuffer, void *extraArguments);

  template <typename OpClass>
  static SD_DEVICE void execTransformCuda(sd::Pointer state, void *z, const sd::LongType *zShapeBuffer,
                                          void *extraArguments);

  static SD_HOST void executeCudaSingle(dim3 &launchDims, cudaStream_t *stream, int opNum, sd::Pointer stateHost,
                                        void *z, const sd::LongType *zShapeBuffer, void *extraArguments);

  static SD_HOST void executeCudaDouble(dim3 &launchDims, cudaStream_t *stream, int opNum, sd::Pointer stateHost,
                                        const void *x, const sd::LongType *xShapeBuffer, void *z,
                                        const sd::LongType *zShapeBuffer, void *extraArguments);

  static SD_HOST void executeCudaTriple(dim3 &launchDims, cudaStream_t *stream, int opNum, sd::Pointer stateHost,
                                        const void *x, const sd::LongType *xShapeBuffer, const void *y,
                                        const sd::LongType *yShapeBuffer, void *z, const sd::LongType *zShapeBuffer,
                                        void *extraArguments);
#else

  template <typename OpClass>
  static void execTransform(sd::Pointer state, const void *x, const sd::LongType *xShapeBuffer, const void *y,
                            const sd::LongType *yShapeBuffer, void *z, const sd::LongType *zShapeBuffer,
                            void *extraArguments);

  template <typename OpClass>
  static void execTransform(sd::Pointer state, const void *x, const sd::LongType *xShapeBuffer, void *z,
                            const sd::LongType *zShapeBuffer, void *extraArguments);

  template <typename OpClass>
  static void execTransform(sd::Pointer state, void *z, const sd::LongType *zShapeBuffer, void *extraArguments);

  static void execTransform(int opNum, sd::Pointer state, const void *x, const sd::LongType *xShapeBuffer, void *z,
                            const sd::LongType *zShapeBuffer, void *extraArguments);
  static void execTransform(int opNum, sd::Pointer state, const void *x, const sd::LongType *xShapeBuffer,
                            const void *y, const sd::LongType *yShapeBuffer, void *z, const sd::LongType *zShapeBuffer,
                            void *extraArguments);
  static void execTransform(int opNum, sd::Pointer state, void *z, const sd::LongType *zShapeBuffer,
                            void *extraArguments);
#endif
};
}  // namespace random
}  // namespace functions

#endif  // LIBND4J_RANDOM_H
