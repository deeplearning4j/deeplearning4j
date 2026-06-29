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
// CSR sparse matrix-vector product: y = op(A) * x
//
// Inputs:
//   [0] values   – 1D [nnz], float dtype   (non-zero values of A)
//   [1] colIdx   – 1D [nnz], INT32/INT64   (column indices)
//   [2] rowPtr   – 1D [rows+1], same INT   (row pointer array)
//   [3] x        – dense 1D vector
//                  length = (transposeA == 0) ? cols : rows
// IArgs:
//   [0] rows
//   [1] cols
//   [2] transposeA  (0 = y = A*x, 1 = y = A^T*x)
// Output:
//   [0] y dense 1D, length = (transposeA == 0) ? rows : cols
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spmv)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_blas.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_spmv, 4, 1, false, 0, 3) {
  auto values     = INPUT_VARIABLE(0);
  auto colIdx     = INPUT_VARIABLE(1);
  auto rowPtr     = INPUT_VARIABLE(2);
  auto x          = INPUT_VARIABLE(3);

  const LongType rows       = INT_ARG(0);
  const LongType cols       = INT_ARG(1);
  const int      transposeA = static_cast<int>(INT_ARG(2));

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_spmv: values must be 1D, got rank %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_spmv: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_spmv: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(x->rankOf() == 1, 0,
               "csr_spmv: x must be 1D, got rank %d", x->rankOf());
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_spmv: values and colIdx must have the same length");
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_spmv: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "csr_spmv: colIdx and rowPtr must have the same integer dtype");

  const LongType xExpected = (transposeA == 0) ? cols : rows;
  REQUIRE_TRUE(x->lengthOf() == xExpected, 0,
               "csr_spmv: x must have length %lld for transposeA=%d, got %lld",
               (long long)xExpected, transposeA, (long long)x->lengthOf());

  auto y = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::csr_spmv(*values, *colIdx, *rowPtr, *x, *y, rows, cols, transposeA);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spmv) {
  auto values = INPUT_VARIABLE(0);

  const LongType rows       = INT_ARG(0);
  const LongType cols       = INT_ARG(1);
  const int      transposeA = static_cast<int>(INT_ARG(2));

  // Output length: rows for normal, cols for transpose
  const LongType yLen = (transposeA == 0) ? rows : cols;

  auto dtype   = values->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {yLen});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_spmv) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // values
      ->setAllowedInputTypes(1, {ALL_INTS})         // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})         // rowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})   // x
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spmv)
