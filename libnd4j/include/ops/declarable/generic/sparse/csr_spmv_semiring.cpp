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
// CSR sparse matrix-vector product under a semiring: y = A ⊗ x
//
// Inputs:
//   [0] values  – 1D [nnz], float dtype   (non-zero values of A)
//   [1] colIdx  – 1D [nnz], INT32/INT64   (column indices)
//   [2] rowPtr  – 1D [rows+1], same INT   (row pointer array)
//   [3] x       – dense 1D vector, length = cols
// IArgs:
//   [0] rows
//   [1] cols
//   [2] semiring  (0=PLUS_TIMES, 1=MIN_PLUS, 2=MAX_PLUS, 3=OR_AND, 4=MIN_TIMES)
// Output:
//   [0] y dense 1D, length = rows
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spmv_semiring)

#include <ops/declarable/headers/sparse_semiring.h>
#include <ops/declarable/helpers/sparse_blas_semiring.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_spmv_semiring, 4, 1, false, 0, 3) {
  auto values = INPUT_VARIABLE(0);
  auto colIdx = INPUT_VARIABLE(1);
  auto rowPtr = INPUT_VARIABLE(2);
  auto x      = INPUT_VARIABLE(3);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const int      semiring = static_cast<int>(INT_ARG(2));

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_spmv_semiring: values must be 1D, got rank %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_spmv_semiring: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_spmv_semiring: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(x->rankOf() == 1, 0,
               "csr_spmv_semiring: x must be 1D, got rank %d", x->rankOf());
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_spmv_semiring: values and colIdx must have the same length");
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_spmv_semiring: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "csr_spmv_semiring: colIdx and rowPtr must have the same integer dtype");
  REQUIRE_TRUE(semiring >= 0 && semiring <= 4, 0,
               "csr_spmv_semiring: semiring must be in [0,4], got %d", semiring);

  auto y = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::csr_spmv_semiring(*values, *colIdx, *rowPtr, *x, *y, rows, cols, semiring);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spmv_semiring) {
  auto values = INPUT_VARIABLE(0);

  const LongType rows = INT_ARG(0);

  auto dtype    = values->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {rows});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_spmv_semiring) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // values
      ->setAllowedInputTypes(1, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})  // x
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spmv_semiring)
