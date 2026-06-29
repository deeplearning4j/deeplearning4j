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
// dense → COO sparse conversion op
// Input:   [0] dense matrix [rows, cols], floating point
// TArgs:   [0] threshold (default 0.0 → keep |x| > threshold)
// Outputs: [0] indices [nnz, 2] INT64 (each row k = {rowOf(k), colOf(k)}, row-major scan order)
//          [1] values  [nnz] same float dtype as input
//
// nnz is computed in DECLARE_SHAPE_FN via a coherence-managed reduction
// (reduce::CountNonZero or reduce::MatchCondition mode-7), never via a
// manual syncToHost + host loop.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dense_to_coo)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_coo.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dense_to_coo, 1, 2, false, 1, 0) {
  auto dense = INPUT_VARIABLE(0);

  REQUIRE_TRUE(dense->rankOf() == 2, 0,
               "dense_to_coo: input must be 2D (matrix), got rank %d", dense->rankOf());

  const double threshold = T_ARG(0);

  auto outIndices = OUTPUT_VARIABLE(0);
  auto outValues  = OUTPUT_VARIABLE(1);

  sd::ops::helpers::denseToCoo(block.launchContext(), dense, outIndices, outValues, threshold);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dense_to_coo) {
  auto dense = INPUT_VARIABLE(0);

  REQUIRE_TRUE(dense->rankOf() == 2, 0,
               "dense_to_coo: input must be 2D (matrix), got rank %d", dense->rankOf());

  const double threshold = block.numT() > 0 ? T_ARG(0) : 0.0;

  // Count nnz via a coherence-managed reduction — reads current device data
  // without any manual syncToHost/syncToDevice (banned per policy).
  // For threshold==0: CountNonZero counts all non-zero entries.
  // For threshold>0:  MatchCondition mode=7 counts entries where |x|>threshold;
  //   extraParams layout expected by MatchCondition::op: [compare, eps, mode].
  LongType nnz = 0;
  if (threshold == 0.0) {
    NDArray* cnt = dense->reduceNumber(reduce::CountNonZero);
    nnz = cnt->e<LongType>(0);
    delete cnt;
  } else {
    ExtraArguments args({threshold, 0.0, 7.0});
    NDArray* cnt = dense->reduceNumber(reduce::MatchCondition,
                                       args.argumentsAsT(dense->dataType()));
    nnz = cnt->e<LongType>(0);
    delete cnt;
  }

  auto floatDtype = dense->dataType();
  auto idxDtype   = sd::DataType::INT64;

  auto indShape = ConstantShapeHelper::getInstance().createShapeInfo(idxDtype,   'c', {nnz, 2});
  auto valShape = ConstantShapeHelper::getInstance().createShapeInfo(floatDtype, 'c', {nnz});

  return SHAPELIST(indShape, valShape);
}

DECLARE_TYPES(dense_to_coo) {
  getOpDescriptor()
      ->setAllowedInputTypes(0,  {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_INTS})
      ->setAllowedOutputTypes(1, {ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_dense_to_coo)
