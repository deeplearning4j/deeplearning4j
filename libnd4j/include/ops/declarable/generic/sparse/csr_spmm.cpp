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
// CSR sparse matrix-matrix product: C = op(A) * B
//
// Inputs:
//   [0] values   – 1D [nnz], float dtype   (non-zero values of A)
//   [1] colIdx   – 1D [nnz], INT32/INT64   (column indices)
//   [2] rowPtr   – 1D [rows+1], same INT   (row pointer array)
//   [3] B        – dense 2D matrix
//                  shape = (transposeA == 0) ? [cols, n] : [rows, n]
// IArgs:
//   [0] rows
//   [1] cols
//   [2] transposeA  (0 = C = A*B, 1 = C = A^T*B)
// Output:
//   [0] C dense 2D, shape = (transposeA == 0) ? [rows, n] : [cols, n]
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spmm)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_blas.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_spmm, 4, 1, false, 0, 3) {
  auto values     = INPUT_VARIABLE(0);
  auto colIdx     = INPUT_VARIABLE(1);
  auto rowPtr     = INPUT_VARIABLE(2);
  auto B          = INPUT_VARIABLE(3);

  const LongType rows       = INT_ARG(0);
  const LongType cols       = INT_ARG(1);
  const int      transposeA = static_cast<int>(INT_ARG(2));

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_spmm: values must be 1D, got rank %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_spmm: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_spmm: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(B->rankOf() == 2, 0,
               "csr_spmm: B must be 2D, got rank %d", B->rankOf());
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_spmm: values and colIdx must have the same length");
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_spmm: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "csr_spmm: colIdx and rowPtr must have the same integer dtype");

  const LongType BrowsExpected = (transposeA == 0) ? cols : rows;
  REQUIRE_TRUE(B->sizeAt(0) == BrowsExpected, 0,
               "csr_spmm: B must have %lld rows for transposeA=%d, got %lld",
               (long long)BrowsExpected, transposeA, (long long)B->sizeAt(0));

  auto C = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::csr_spmm(*values, *colIdx, *rowPtr, *B, *C, rows, cols, transposeA);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spmm) {
  auto values = INPUT_VARIABLE(0);
  auto B      = INPUT_VARIABLE(3);

  const LongType rows       = INT_ARG(0);
  const LongType cols       = INT_ARG(1);
  const int      transposeA = static_cast<int>(INT_ARG(2));

  const LongType n     = B->sizeAt(1);
  // Output rows: A has [rows,cols]; A^T has [cols,rows]; so C rows = transposeA? cols : rows
  const LongType Crows = (transposeA == 0) ? rows : cols;

  auto dtype    = values->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {Crows, n});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_spmm) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // values
      ->setAllowedInputTypes(1, {ALL_INTS})         // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})         // rowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})   // B
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spmm)
