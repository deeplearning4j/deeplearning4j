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
// coo_to_csr_bp: backward pass for coo_to_csr.
//
// Forward recap:
//   Inputs:  [0] cooIndices [coo_nnz, 2]  INT
//            [1] cooValues  [coo_nnz]     float
//   IArgs:   [0] rows, [1] cols
//   Outputs: [0] csrValues [csr_nnz]     float   ← forward may coalesce duplicates
//            [1] csrColIdx [csr_nnz]     INT32
//            [2] csrRowPtr [rows+1]      INT32
//
// Backward:
//   Inputs:  [0] cooIndices    [coo_nnz, 2]  INT    (forward input[0])
//            [1] csrColIdx     [csr_nnz]     INT32  (forward output[1])
//            [2] csrRowPtr     [rows+1]      INT32  (forward output[2])
//            [3] gradCsrValues [csr_nnz]     float  (upstream gradient for csrValues)
//   IArgs:   [0] rows, [1] cols
//   Output:  [0] dCooValues [coo_nnz] float
//
//   Math: the gradient of a coalescing SUM distributes (copies) to every COO entry
//   that fed the corresponding CSR slot.  For each COO entry k with
//   (i, j) = cooIndices[k]:
//     binary-search csrColIdx[ csrRowPtr[i] .. csrRowPtr[i+1] ) for column j
//     → dCooValues[k] = gradCsrValues[foundPos]
//   No atomics: each COO entry maps to at most one CSR slot.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_coo_to_csr_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_coo_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(coo_to_csr_bp, 4, 1, false, 0, 2) {
  auto cooIndices    = INPUT_VARIABLE(0);  // [coo_nnz, 2] INT
  auto csrColIdx     = INPUT_VARIABLE(1);  // [csr_nnz]    INT32
  auto csrRowPtr     = INPUT_VARIABLE(2);  // [rows+1]     INT32
  auto gradCsrValues = INPUT_VARIABLE(3);  // [csr_nnz]    float

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(cooIndices->rankOf() == 2, 0,
               "coo_to_csr_bp: cooIndices must be 2D [coo_nnz, 2], got rank %d",
               cooIndices->rankOf());
  REQUIRE_TRUE(cooIndices->sizeAt(1) == 2, 0,
               "coo_to_csr_bp: cooIndices second dim must be 2, got %lld",
               (long long)cooIndices->sizeAt(1));
  REQUIRE_TRUE(csrColIdx->rankOf() == 1, 0,
               "coo_to_csr_bp: csrColIdx must be 1D, got rank %d", csrColIdx->rankOf());
  REQUIRE_TRUE(csrRowPtr->rankOf() == 1, 0,
               "coo_to_csr_bp: csrRowPtr must be 1D, got rank %d", csrRowPtr->rankOf());
  REQUIRE_TRUE(csrRowPtr->lengthOf() == rows + 1, 0,
               "coo_to_csr_bp: csrRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)csrRowPtr->lengthOf());
  REQUIRE_TRUE(gradCsrValues->lengthOf() == csrColIdx->lengthOf(), 0,
               "coo_to_csr_bp: gradCsrValues and csrColIdx lengths must match (%lld vs %lld)",
               (long long)gradCsrValues->lengthOf(), (long long)csrColIdx->lengthOf());

  auto dCooValues = OUTPUT_NULLIFIED(0);  // [coo_nnz] float

  sd::ops::helpers::cooToCsr_bp(block.launchContext(),
                                  cooIndices, csrColIdx, csrRowPtr,
                                  gradCsrValues, dCooValues, rows, cols);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(coo_to_csr_bp) {
  // dCooValues has shape [coo_nnz] = [cooIndices.sizeAt(0)], dtype from gradCsrValues (input 3).
  // This is data-independent: coo_nnz == cooIndices.sizeAt(0) is statically known.
  auto cooIndices    = INPUT_VARIABLE(0);
  auto gradCsrValues = INPUT_VARIABLE(3);

  const LongType cooNnz = cooIndices->sizeAt(0);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradCsrValues->dataType(), 'c', {cooNnz});

  return SHAPELIST(outShape);
}

DECLARE_TYPES(coo_to_csr_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // cooIndices
      ->setAllowedInputTypes(1, {ALL_INTS})    // csrColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // csrRowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})  // gradCsrValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dCooValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_coo_to_csr_bp)
