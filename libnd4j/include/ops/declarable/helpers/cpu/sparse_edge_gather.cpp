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
// csr_edge_gather / csr_edge_gather_bp — CPU implementation.
//
// Forward — for each edge e in [0, nnz) and each feature dim f in [0, F):
//   edgeFeat[e, f] = X[colIdx[e], f]
//   Both X and edgeFeat are accessed stride-aware.
//
// Backward — scatter-add upstream gradients back to node features:
//   dX[colIdx[e], f] += gradEdge[e, f]
//   CPU is sequential so no atomics needed here; dX is zeroed via
//   std::memset before the accumulation loop.
//

#include <ops/declarable/helpers/sparse_edge_gather.h>
#include <system/op_boilerplate.h>

#include <cstring>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrEdgeGather_(NDArray& colIdx, NDArray& xArr, NDArray& edgeFeat) {
  const I* ciBuf  = colIdx.bufferAsT<I>();
  const X* xBuf   = xArr.bufferAsT<X>();
  X*       efBuf  = edgeFeat.bufferAsT<X>();

  const sd::LongType nnz  = colIdx.lengthOf();
  const sd::LongType F    = xArr.sizeAt(1);
  const sd::LongType xs0  = xArr.strideAt(0);
  const sd::LongType xs1  = xArr.strideAt(1);
  const sd::LongType es0  = edgeFeat.strideAt(0);
  const sd::LongType es1  = edgeFeat.strideAt(1);

  for (sd::LongType e = 0; e < nnz; ++e) {
    const sd::LongType node = static_cast<sd::LongType>(ciBuf[e]);
    for (sd::LongType f = 0; f < F; ++f) {
      efBuf[e * es0 + f * es1] = xBuf[node * xs0 + f * xs1];
    }
  }
}

void csr_edge_gather(NDArray& colIdx, NDArray& xArr, NDArray& edgeFeat) {
  NDArray::preparePrimaryUse({&edgeFeat}, {&colIdx, &xArr});

  BUILD_DOUBLE_SELECTOR(xArr.dataType(), colIdx.dataType(), csrEdgeGather_,
                        (colIdx, xArr, edgeFeat),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&edgeFeat}, {&colIdx, &xArr});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrEdgeGatherBp_(NDArray& colIdx, NDArray& xArr, NDArray& gradEdge, NDArray& dX) {
  // xArr is passed for its shape (n = xArr.sizeAt(0), F = xArr.sizeAt(1));
  // its values are NOT read — dX is already shaped [n, F] by DECLARE_SHAPE_FN.
  const I* ciBuf  = colIdx.bufferAsT<I>();
  const X* geBuf  = gradEdge.bufferAsT<X>();
  X*       dxBuf  = dX.bufferAsT<X>();

  const sd::LongType nnz  = colIdx.lengthOf();
  const sd::LongType F    = dX.sizeAt(1);
  const sd::LongType ges0 = gradEdge.strideAt(0);
  const sd::LongType ges1 = gradEdge.strideAt(1);
  const sd::LongType ds0  = dX.strideAt(0);
  const sd::LongType ds1  = dX.strideAt(1);

  // Zero-initialise dX (scatter target)
  std::memset(dxBuf, 0, sizeof(X) * static_cast<size_t>(dX.lengthOf()));

  for (sd::LongType e = 0; e < nnz; ++e) {
    const sd::LongType node = static_cast<sd::LongType>(ciBuf[e]);
    for (sd::LongType f = 0; f < F; ++f) {
      dxBuf[node * ds0 + f * ds1] += geBuf[e * ges0 + f * ges1];
    }
  }
}

void csr_edge_gather_bp(NDArray& colIdx, NDArray& xArr, NDArray& gradEdge, NDArray& dX) {
  NDArray::preparePrimaryUse({&dX}, {&colIdx, &xArr, &gradEdge});

  BUILD_DOUBLE_SELECTOR(gradEdge.dataType(), colIdx.dataType(), csrEdgeGatherBp_,
                        (colIdx, xArr, gradEdge, dX),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dX}, {&colIdx, &xArr, &gradEdge});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
