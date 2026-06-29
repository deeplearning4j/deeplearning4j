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
// csr_subgraph_extract — extract the induced-subgraph CSR for K selected nodes
// from a CSR graph of N nodes.
//
// Inputs:
//   [0] values   [nnz]   float  — edge weights
//   [1] colIdx   [nnz]   int    — column indices (destination node ids)
//   [2] rowPtr   [N+1]   int    — row pointers
//   [3] nodeIdx  [K]     int    — SORTED ascending selected node ids
// IArgs:
//   [0] N  — original node count
//   [1] K  — selected node count
// Outputs:
//   [0] newValues  [nnz']  float  — kept edge weights (same dtype as values)
//   [1] newColIdx  [nnz']  INT32  — remapped destination ids (0..K-1)
//   [2] newRowPtr  [K+1]   INT32  — subgraph row pointers
//
// nnz' is DATA-DEPENDENT: the DECLARE_SHAPE_FN reads the integer structural
// arrays (colIdx, rowPtr, nodeIdx) via e<LongType>() and counts how many
// original edges have BOTH endpoints in nodeIdx.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_subgraph_extract)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_subgraph.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_subgraph_extract, 4, 3, false, 0, 2) {
  auto values  = INPUT_VARIABLE(0);  // [nnz]  float
  auto colIdx  = INPUT_VARIABLE(1);  // [nnz]  int
  auto rowPtr  = INPUT_VARIABLE(2);  // [N+1]  int
  auto nodeIdx = INPUT_VARIABLE(3);  // [K]    int  (sorted)

  const sd::LongType N = INT_ARG(0);
  const sd::LongType K = INT_ARG(1);

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_subgraph_extract: values must be rank-1, got %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_subgraph_extract: colIdx must be rank-1, got %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_subgraph_extract: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(nodeIdx->rankOf() == 1, 0,
               "csr_subgraph_extract: nodeIdx must be rank-1, got %d", nodeIdx->rankOf());
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_subgraph_extract: values and colIdx must have the same length");
  REQUIRE_TRUE(rowPtr->lengthOf() == N + 1, 0,
               "csr_subgraph_extract: rowPtr length must be N+1=%lld, got %lld",
               (long long)(N + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(nodeIdx->lengthOf() == K, 0,
               "csr_subgraph_extract: nodeIdx length must be K=%lld, got %lld",
               (long long)K, (long long)nodeIdx->lengthOf());

  // Empty subgraph: K==0 or nnz==0 → outputs already shaped to zero by shape-fn
  if (K == 0 || values->lengthOf() == 0) return sd::Status::OK;

  auto newValues = OUTPUT_VARIABLE(0);  // [nnz'] float
  auto newColIdx = OUTPUT_VARIABLE(1);  // [nnz'] INT32
  auto newRowPtr = OUTPUT_VARIABLE(2);  // [K+1]  INT32

  sd::ops::helpers::csr_subgraph_extract(
      block.launchContext(),
      *values, *colIdx, *rowPtr, *nodeIdx,
      *newValues, *newColIdx, *newRowPtr,
      N, K);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_subgraph_extract) {
  auto values  = INPUT_VARIABLE(0);
  auto colIdx  = INPUT_VARIABLE(1);
  auto rowPtr  = INPUT_VARIABLE(2);
  auto nodeIdx = INPUT_VARIABLE(3);

  const sd::LongType N   = INT_ARG(0);
  const sd::LongType K   = INT_ARG(1);
  const sd::LongType nnz = values->lengthOf();

  const auto floatType = values->dataType();
  const auto idxType   = sd::DataType::INT32;

  // Count nnz' by scanning: for each selected row s, scan its edges and
  // binary-search the destination column in nodeIdx.
  // Reading integer structural arrays via e<LongType>() is acceptable in the
  // shape function (coherence-managed; NOT a manual syncToHost).
  sd::LongType nnzPrime = 0;
  if (K > 0 && nnz > 0) {
    for (sd::LongType s = 0; s < K; ++s) {
      const sd::LongType origRow  = nodeIdx->e<sd::LongType>(s);
      const sd::LongType eStart   = rowPtr->e<sd::LongType>(origRow);
      const sd::LongType eEnd     = rowPtr->e<sd::LongType>(origRow + 1);

      for (sd::LongType e = eStart; e < eEnd; ++e) {
        const sd::LongType col = colIdx->e<sd::LongType>(e);
        // Binary search col in nodeIdx[0..K-1]
        sd::LongType lo = 0, hi = K - 1;
        bool found = false;
        while (lo <= hi) {
          sd::LongType mid = (lo + hi) / 2;
          sd::LongType mv  = nodeIdx->e<sd::LongType>(mid);
          if (mv == col) { found = true; break; }
          else if (mv < col) lo = mid + 1;
          else               hi = mid - 1;
        }
        if (found) ++nnzPrime;
      }
    }
  }

  auto valShape  = ConstantShapeHelper::getInstance().createShapeInfo(floatType, 'c', {nnzPrime});
  auto ciShape   = ConstantShapeHelper::getInstance().createShapeInfo(idxType,   'c', {nnzPrime});
  auto rpShape   = ConstantShapeHelper::getInstance().createShapeInfo(idxType,   'c', {K + 1});

  return SHAPELIST(valShape, ciShape, rpShape);
}

DECLARE_TYPES(csr_subgraph_extract) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // values
      ->setAllowedInputTypes(1, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(3, {ALL_INTS})    // nodeIdx
      ->setAllowedOutputTypes(0, {ALL_FLOATS}) // newValues
      ->setAllowedOutputTypes(1, {ALL_INTS})   // newColIdx
      ->setAllowedOutputTypes(2, {ALL_INTS});  // newRowPtr
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_subgraph_extract)
