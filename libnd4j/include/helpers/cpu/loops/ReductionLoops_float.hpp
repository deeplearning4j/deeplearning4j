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
#include <types/types.h>

#include "ReductionLoops.hpp"

using namespace simdOps;

namespace sd {

template <typename X, typename Z>
template <typename OpType>
SD_LIB_HIDDEN void ReductionFloatLoops<X, Z>::innerloopReduce(sd::memory::Workspace* workspace, const X* x,
                                                              const sd::LongType* xShapeInfo, Z* z,
                                                              const sd::LongType* zShapeInfo, const LongType* dims,
                                                              Z* extraParams) {
#ifndef SD_LOOPS_INLINED
  ReductionLoops<X, Z, Z>::template loopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams);
#endif
}

template <typename X, typename Y>
SD_LIB_HIDDEN void ReductionFloatLoops<X, Y>::wrapper(int opNum, sd::memory::Workspace* workspace, const X* x,
                                                      const sd::LongType* xShapeInfo, Y* z,
                                                      const sd::LongType* zShapeInfo, const LongType* dims, Y* extraParams) {
#ifndef SD_LOOPS_INLINED
  DISPATCH_BY_OPNUM_TT(innerloopReduce, PARAMS(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams),
                       REDUCE_FLOAT_OPS);
#endif
}

}  // namespace sd
