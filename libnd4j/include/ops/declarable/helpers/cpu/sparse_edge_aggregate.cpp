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
// csr_edge_aggregate / csr_edge_aggregate_bp — CPU implementation.
//
// Forward — outer loop over rows i, inner loop over feature dims f:
//   mode 0 (SUM):  out[i, f] = Σ edgeMsg[e, f]  for e in row i
//   mode 1 (MEAN): same, then divide by degree(i); empty row → 0
//   mode 2 (MAX):  out[i, f] = max edgeMsg[e, f] over e in row i; empty → 0
//
// Backward — routes gradients back to edges:
//   mode 0 (SUM):  dEdgeMsg[e, f] = gradOut[i, f]          for all e in row i
//   mode 1 (MEAN): dEdgeMsg[e, f] = gradOut[i, f] / deg(i)
//   mode 2 (MAX):  recompute argmax e* per (i, f); dEdgeMsg[e*, f] += gradOut[i, f]
//   dEdgeMsg is zeroed via std::memset before the loops.
//

#include <ops/declarable/helpers/sparse_edge_aggregate.h>
#include <system/op_boilerplate.h>

#include <cstring>
#include <limits>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrEdgeAggregate_(NDArray& rowPtr, NDArray& edgeMsg, NDArray& out,
                               sd::LongType rows, int mode) {
  const I* rpBuf  = rowPtr.bufferAsT<I>();
  const X* emBuf  = edgeMsg.bufferAsT<X>();
  X*       oBuf   = out.bufferAsT<X>();

  const sd::LongType F    = edgeMsg.sizeAt(1);
  const sd::LongType ems0 = edgeMsg.strideAt(0);
  const sd::LongType ems1 = edgeMsg.strideAt(1);
  const sd::LongType os0  = out.strideAt(0);
  const sd::LongType os1  = out.strideAt(1);

  for (sd::LongType i = 0; i < rows; ++i) {
    const sd::LongType eStart = static_cast<sd::LongType>(rpBuf[i]);
    const sd::LongType eEnd   = static_cast<sd::LongType>(rpBuf[i + 1]);
    const sd::LongType deg    = eEnd - eStart;

    for (sd::LongType f = 0; f < F; ++f) {
      X* o = oBuf + i * os0 + f * os1;

      if (mode == 2) {
        // MAX
        if (deg == 0) {
          *o = static_cast<X>(0);
          continue;
        }
        X m = emBuf[eStart * ems0 + f * ems1];
        for (sd::LongType e = eStart + 1; e < eEnd; ++e) {
          X v = emBuf[e * ems0 + f * ems1];
          if (v > m) m = v;
        }
        *o = m;
      } else {
        // SUM or MEAN
        X acc = static_cast<X>(0);
        for (sd::LongType e = eStart; e < eEnd; ++e) {
          acc += emBuf[e * ems0 + f * ems1];
        }
        if (mode == 1 && deg > 0) {
          acc /= static_cast<X>(deg);
        }
        *o = acc;
      }
    }
  }
}

void csr_edge_aggregate(NDArray& rowPtr, NDArray& edgeMsg, NDArray& out,
                         sd::LongType rows, int mode) {
  NDArray::preparePrimaryUse({&out}, {&rowPtr, &edgeMsg});

  BUILD_DOUBLE_SELECTOR(edgeMsg.dataType(), rowPtr.dataType(), csrEdgeAggregate_,
                        (rowPtr, edgeMsg, out, rows, mode),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&out}, {&rowPtr, &edgeMsg});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrEdgeAggregateBp_(NDArray& rowPtr, NDArray& edgeMsg, NDArray& gradOut,
                                 NDArray& dEdgeMsg, sd::LongType rows, int mode) {
  const I* rpBuf  = rowPtr.bufferAsT<I>();
  const X* emBuf  = edgeMsg.bufferAsT<X>();
  const X* goBuf  = gradOut.bufferAsT<X>();
  X*       deBuf  = dEdgeMsg.bufferAsT<X>();

  const sd::LongType F    = edgeMsg.sizeAt(1);
  const sd::LongType ems0 = edgeMsg.strideAt(0);
  const sd::LongType ems1 = edgeMsg.strideAt(1);
  const sd::LongType gos0 = gradOut.strideAt(0);
  const sd::LongType gos1 = gradOut.strideAt(1);
  const sd::LongType des0 = dEdgeMsg.strideAt(0);
  const sd::LongType des1 = dEdgeMsg.strideAt(1);

  // Zero-initialise dEdgeMsg (needed for MAX which only writes to argmax edges)
  std::memset(deBuf, 0, sizeof(X) * static_cast<size_t>(dEdgeMsg.lengthOf()));

  for (sd::LongType i = 0; i < rows; ++i) {
    const sd::LongType eStart = static_cast<sd::LongType>(rpBuf[i]);
    const sd::LongType eEnd   = static_cast<sd::LongType>(rpBuf[i + 1]);
    const sd::LongType deg    = eEnd - eStart;
    if (deg == 0) continue;

    for (sd::LongType f = 0; f < F; ++f) {
      const X g = goBuf[i * gos0 + f * gos1];

      if (mode == 2) {
        // MAX backward: recompute argmax edge for this (i, f) cell
        sd::LongType argmaxE = eStart;
        X            maxVal  = emBuf[eStart * ems0 + f * ems1];
        for (sd::LongType e = eStart + 1; e < eEnd; ++e) {
          X v = emBuf[e * ems0 + f * ems1];
          if (v > maxVal) { maxVal = v; argmaxE = e; }
        }
        // CPU is sequential — no atomic needed
        deBuf[argmaxE * des0 + f * des1] += g;
      } else if (mode == 1) {
        // MEAN backward
        const X scale = g / static_cast<X>(deg);
        for (sd::LongType e = eStart; e < eEnd; ++e) {
          deBuf[e * des0 + f * des1] = scale;
        }
      } else {
        // SUM backward
        for (sd::LongType e = eStart; e < eEnd; ++e) {
          deBuf[e * des0 + f * des1] = g;
        }
      }
    }
  }
}

void csr_edge_aggregate_bp(NDArray& rowPtr, NDArray& edgeMsg, NDArray& gradOut,
                            NDArray& dEdgeMsg, sd::LongType rows, int mode) {
  NDArray::preparePrimaryUse({&dEdgeMsg}, {&rowPtr, &edgeMsg, &gradOut});

  BUILD_DOUBLE_SELECTOR(edgeMsg.dataType(), rowPtr.dataType(), csrEdgeAggregateBp_,
                        (rowPtr, edgeMsg, gradOut, dEdgeMsg, rows, mode),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dEdgeMsg}, {&rowPtr, &edgeMsg, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
