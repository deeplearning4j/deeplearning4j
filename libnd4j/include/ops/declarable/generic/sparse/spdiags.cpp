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
// spdiags — build a diagonal CSR from a vector.
//
// Given a float vector diag[n], produces the CSR representation of the n×n
// diagonal matrix with diag on the main diagonal.  This is a trivial
// (data-independent) shape operation:
//   nnz = n,  rowPtr[i] = i,  colIdx[i] = i,  values[i] = diag[i].
//
// Input:
//   [0] diag  [n], floating dtype
// IArgs:
//   [0] n  — size of the diagonal (and of the square matrix)
// Outputs:
//   [0] values  [n]   — same dtype as diag  (= copy of diag)
//   [1] colIdx  [n]   — INT32, colIdx[i] = i
//   [2] rowPtr  [n+1] — INT32, rowPtr[i] = i
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_spdiags)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_spdiags.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(spdiags, 1, 3, false, 0, 1) {
  auto diag = INPUT_VARIABLE(0);

  const LongType n = INT_ARG(0);

  REQUIRE_TRUE(diag->rankOf() == 1, 0,
               "spdiags: diag must be 1D, got rank %d", diag->rankOf());
  REQUIRE_TRUE(diag->lengthOf() >= n, 0,
               "spdiags: diag length %lld must be >= n=%lld",
               (long long)diag->lengthOf(), (long long)n);
  REQUIRE_TRUE(n >= 0, 0, "spdiags: n must be non-negative, got %lld", (long long)n);

  auto values = OUTPUT_VARIABLE(0);
  auto colIdx = OUTPUT_VARIABLE(1);
  auto rowPtr = OUTPUT_VARIABLE(2);

  sd::ops::helpers::spdiags(block.launchContext(), *diag, *values, *colIdx, *rowPtr, n);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(spdiags) {
  auto diag = INPUT_VARIABLE(0);

  const LongType n       = INT_ARG(0);
  const auto     dtype   = diag->dataType();
  const auto     idxType = sd::DataType::INT32;

  auto valShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,   'c', {n});
  auto ciShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {n});
  auto rpShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {n + 1});

  return SHAPELIST(valShape, ciShape, rpShape);
}

DECLARE_TYPES(spdiags) {
  getOpDescriptor()
      ->setAllowedInputTypes(0,  {ALL_FLOATS})  // diag
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // values
      ->setAllowedOutputTypes(1, {ALL_INTS})    // colIdx
      ->setAllowedOutputTypes(2, {ALL_INTS});   // rowPtr
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_spdiags)
