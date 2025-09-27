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
//  @author  raver119@gmail.com
//
#include <helpers/Loops.h>
#include <loops/legacy_ops.h>
#include <loops/transform_same.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace transform {

template <typename X>
void TransformSame<X>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                            const sd::LongType *zShapeInfo, void *extraParams, sd::LongType threadId,
                            sd::LongType numThreads) {
  DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, z, zShapeInfo, extraParams, threadId, numThreads),
                      TRANSFORM_SAME_OPS);
}

template <typename X>
template <typename OpType>
void SD_HOST TransformSame<X>::exec(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                    const sd::LongType *zShapeInfo, void *vextraParams, sd::LongType threadId,
                                    sd::LongType numThreads) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<X *>(vz);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  sd::TransformLoops<X, X, X>::template loopTransform<OpType>(x, xShapeInfo, z, zShapeInfo, extraParams, threadId,
                                                              numThreads);
}

BUILD_SINGLE_TEMPLATE( class TransformSame, , SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE( class TransformSame, , SD_BOOL_TYPES);

}  // namespace transform
}  // namespace functions
