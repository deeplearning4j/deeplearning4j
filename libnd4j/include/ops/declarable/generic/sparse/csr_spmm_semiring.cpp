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
// CSR sparse matrix-matrix product under a semiring: C = A ⊗ B
//
// Inputs:
//   [0] values  – 1D [nnz], float dtype   (non-zero values of A)
//   [1] colIdx  – 1D [nnz], INT32/INT64   (column indices)
//   [2] rowPtr  – 1D [rows+1], same INT   (row pointer array)
//   [3] B       – dense 2D matrix, shape [cols, n]
// IArgs:
//   [0] rows
//   [1] cols
//   [2] semiring  (0=PLUS_TIMES, 1=MIN_PLUS, 2=MAX_PLUS, 3=OR_AND, 4=MIN_TIMES)
// Output:
//   [0] C dense 2D, shape [rows, n]
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spmm_semiring)

#include <ops/declarable/headers/sparse_semiring.h>
#include <ops/declarable/helpers/sparse_blas_semiring.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_spmm_semiring, 4, 1, false, 0, 3) {
  auto values = INPUT_VARIABLE(0);
  auto colIdx = INPUT_VARIABLE(1);
  auto rowPtr = INPUT_VARIABLE(2);
  auto B      = INPUT_VARIABLE(3);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const int      semiring = static_cast<int>(INT_ARG(2));

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_spmm_semiring: values must be 1D, got rank %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_spmm_semiring: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_spmm_semiring: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(B->rankOf() == 2, 0,
               "csr_spmm_semiring: B must be 2D, got rank %d", B->rankOf());
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_spmm_semiring: values and colIdx must have the same length");
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_spmm_semiring: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "csr_spmm_semiring: colIdx and rowPtr must have the same integer dtype");
  REQUIRE_TRUE(B->sizeAt(0) == cols, 0,
               "csr_spmm_semiring: B must have %lld rows (cols of A), got %lld",
               (long long)cols, (long long)B->sizeAt(0));
  REQUIRE_TRUE(semiring >= 0 && semiring <= 4, 0,
               "csr_spmm_semiring: semiring must be in [0,4], got %d", semiring);

  auto C = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::csr_spmm_semiring(*values, *colIdx, *rowPtr, *B, *C, rows, cols, semiring);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spmm_semiring) {
  auto values = INPUT_VARIABLE(0);
  auto B      = INPUT_VARIABLE(3);

  const LongType rows = INT_ARG(0);
  const LongType n    = B->sizeAt(1);

  auto dtype    = values->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {rows, n});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_spmm_semiring) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // values
      ->setAllowedInputTypes(1, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})  // B
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spmm_semiring)
