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
// COO → CSR sparse conversion op
// Inputs: [0] indices [nnz, 2] INT (col 0 = row index, col 1 = col index)
//         [1] values  [nnz] float
// IArgs:  [0] rows, [1] cols
// Outputs:[0] csrValues [nnz] float (same dtype as values)
//         [1] colIdx    [nnz] INT32
//         [2] rowPtr    [rows+1] INT32
//
// Entries are sorted by (row, col) in the output CSR representation.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_coo_to_csr)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_coo.h>
#include <algorithm>
#include <vector>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(coo_to_csr, 2, 3, false, 0, 2) {
  auto cooIndices = INPUT_VARIABLE(0);
  auto cooValues  = INPUT_VARIABLE(1);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(cooIndices->rankOf() == 2, 0,
               "coo_to_csr: indices must be 2D [nnz, 2], got rank %d", cooIndices->rankOf());
  REQUIRE_TRUE(cooIndices->sizeAt(1) == 2, 0,
               "coo_to_csr: indices second dim must be 2, got %lld",
               (long long)cooIndices->sizeAt(1));
  REQUIRE_TRUE(cooValues->rankOf() == 1, 0,
               "coo_to_csr: values must be 1D, got rank %d", cooValues->rankOf());
  REQUIRE_TRUE(cooIndices->sizeAt(0) == cooValues->lengthOf(), 0,
               "coo_to_csr: indices rows (%lld) must match values length (%lld)",
               (long long)cooIndices->sizeAt(0), (long long)cooValues->lengthOf());

  auto csrValues = OUTPUT_VARIABLE(0);
  auto colIdx    = OUTPUT_VARIABLE(1);
  auto rowPtr    = OUTPUT_VARIABLE(2);

  sd::ops::helpers::cooToCsr(block.launchContext(), cooIndices, cooValues,
                              csrValues, colIdx, rowPtr, rows);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(coo_to_csr) {
  auto cooIndices = INPUT_VARIABLE(0);
  auto cooValues  = INPUT_VARIABLE(1);

  const LongType nnz  = cooValues->lengthOf();
  const LongType rows = INT_ARG(0);

  auto floatDtype = cooValues->dataType();
  auto idxDtype   = sd::DataType::INT32;

  // Build sorted list of (row, col) pairs and count distinct ones.
  // cooIndices has shape [nnz, 2]; linear index k*2+0 = row, k*2+1 = col.
  std::vector<std::pair<LongType, LongType>> pairs(static_cast<size_t>(nnz));
  for (LongType k = 0; k < nnz; ++k) {
    pairs[k].first  = cooIndices->e<LongType>(k * 2);
    pairs[k].second = cooIndices->e<LongType>(k * 2 + 1);
  }
  std::sort(pairs.begin(), pairs.end());

  LongType coalescedNnz = nnz > 0 ? 1 : 0;
  for (LongType k = 1; k < nnz; ++k) {
    if (pairs[k] != pairs[k - 1]) ++coalescedNnz;
  }

  // values and colIdx are sized to the coalesced count; rowPtr is always rows+1.
  auto cvShape = ConstantShapeHelper::getInstance().createShapeInfo(floatDtype, 'c', {coalescedNnz});
  auto ciShape = ConstantShapeHelper::getInstance().createShapeInfo(idxDtype,   'c', {coalescedNnz});
  auto rpShape = ConstantShapeHelper::getInstance().createShapeInfo(idxDtype,   'c', {rows + 1});

  return SHAPELIST(cvShape, ciShape, rpShape);
}

DECLARE_TYPES(coo_to_csr) {
  getOpDescriptor()
      ->setAllowedInputTypes(0,  {ALL_INTS})    // indices
      ->setAllowedInputTypes(1,  {ALL_FLOATS})  // values
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // csrValues
      ->setAllowedOutputTypes(1, {ALL_INTS})    // colIdx
      ->setAllowedOutputTypes(2, {ALL_INTS});   // rowPtr
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_coo_to_csr)
