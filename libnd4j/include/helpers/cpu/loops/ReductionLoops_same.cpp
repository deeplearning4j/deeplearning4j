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
#include "ReductionLoops.hpp"

using namespace simdOps;

namespace sd {

template <typename X>
template <typename OpType>
void ReductionSameLoops<X>::innerloopReduce(sd::memory::Workspace *workspace, const X *x,
                                            const sd::LongType *xShapeInfo, X *z, const sd::LongType *zShapeInfo,
                                            const LongType *dims, X *extraParams) {
#ifndef SD_LOOPS_INLINED
  ReductionLoops<X, X, X>::template loopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams);
#endif
}

template <typename X>
void ReductionSameLoops<X>::wrapper(int opNum, sd::memory::Workspace *workspace, const X *vx,
                                    const sd::LongType *xShapeInfo, X *z, const sd::LongType *zShapeInfo,
                                    const LongType *dims, X *vextraParams) {
#ifndef SD_LOOPS_INLINED
  auto x = reinterpret_cast<X *>(vx);
  auto z = reinterpret_cast<X *>(vz);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  DISPATCH_BY_OPNUM_T(innerloopReduce, PARAMS(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams),
                      REDUCE_SAME_OPS);
#endif
}

BUILD_SINGLE_TEMPLATE(template class ReductionSameLoops, , SD_COMMON_TYPES);
}  // namespace sd
