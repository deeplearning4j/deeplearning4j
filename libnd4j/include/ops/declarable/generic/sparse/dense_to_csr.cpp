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
// dense → CSR sparse conversion op
// Input:   [0] dense matrix [rows, cols]
// TArgs:   [0] threshold (default 0.0)
// Outputs: [0] values [nnz]
//          [1] colIdx [nnz] (INT32)
//          [2] rowPtr [rows+1] (INT32)
//
// nnz is computed in DECLARE_SHAPE_FN via a coherence-managed reduction
// (reduce::CountNonZero or reduce::MatchCondition mode-7), never via a
// manual syncToHost + host loop.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dense_to_csr)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_csr.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dense_to_csr, 1, 3, false, 1, 0) {
  auto dense = INPUT_VARIABLE(0);

  REQUIRE_TRUE(dense->rankOf() == 2, 0,
               "dense_to_csr: input must be 2D (matrix), got rank %d", dense->rankOf());

  const double threshold = T_ARG(0);

  auto outValues = OUTPUT_VARIABLE(0);
  auto outColIdx = OUTPUT_VARIABLE(1);
  auto outRowPtr = OUTPUT_VARIABLE(2);

  sd::ops::helpers::dense_to_csr(block.launchContext(), *dense, *outValues, *outColIdx, *outRowPtr, threshold);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dense_to_csr) {
  auto dense = INPUT_VARIABLE(0);

  REQUIRE_TRUE(dense->rankOf() == 2, 0,
               "dense_to_csr: input must be 2D (matrix), got rank %d", dense->rankOf());

  const double threshold = block.numT() > 0 ? T_ARG(0) : 0.0;

  const LongType rows = dense->sizeAt(0);

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

  auto dtype   = dense->dataType();
  auto idxType = sd::DataType::INT32;

  auto valShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,   'c', {nnz});
  auto ciShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {nnz});
  auto rpShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {rows + 1});

  return SHAPELIST(valShape, ciShape, rpShape);
}

DECLARE_TYPES(dense_to_csr) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->setAllowedOutputTypes(1, {ALL_INTS})
      ->setAllowedOutputTypes(2, {ALL_INTS});

  getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_dense_to_csr)
