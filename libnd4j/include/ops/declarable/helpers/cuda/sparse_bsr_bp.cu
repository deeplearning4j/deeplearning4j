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
// CUDA backward-pass helpers for BSR sparse operations:
//
// bsr_to_dense_bp:
//   One thread per BSR value element (total = nnzb * bd * bd).
//   Thread e → block_k = e / (bd*bd), lr = (e%(bd*bd)) / bd, lc = e%bd.
//   Binary-search bsrRowPtr to find block-row br, then bc = bsrColIdx[block_k].
//   dBsrValues[e] = gradDense[br*bd+lr, bc*bd+lc].
//   Pure gather — no atomics.
//
// bsr_spmm_bp:
//   dBsrValues: one thread per BSR value element; serial dot-product over n_cols.
//   dB:         one thread per BSR value element; atomicAdd scatter into dB[bc*bd+lc, n].
//   Both outputs must be pre-zeroed (OUTPUT_NULLIFIED).
//
// csr_to_bsr_bp:
//   One thread per CSR entry k.
//   Binary-search csrRowPtr for global row r; col = csrColIdx[k].
//   br = r/bd, bc = col/bd, lr = r%bd, lc = col%bd.
//   Binary-search bsrColIdx[bsrRowPtr[br]..bsrRowPtr[br+1]) for bc → block_k.
//   dCsrValues[k] = gradBsrValues[block_k*bd*bd + lr*bd + lc].
//   Pure gather — no atomics.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_bsr_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ─── device utilities ────────────────────────────────────────────────────────

// Binary-search rowPtr[0..nblocks] to find block index b such that
//   rowPtr[b] <= k < rowPtr[b+1].
template <typename I>
static SD_INLINE SD_DEVICE LongType bsrFindBlock(const I* rowPtr, LongType nblocks, LongType k) {
  LongType lo = 0, hi = nblocks - 1, blk = 0;
  while (lo <= hi) {
    const LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(rowPtr[mid]) <= k &&
        k < static_cast<LongType>(rowPtr[mid + 1])) {
      blk = mid;
      break;
    } else if (static_cast<LongType>(rowPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return blk;
}

// Lower-bound search for bc in a sorted slice bsrColIdx[clo, chi).
// Returns the absolute index of the found entry.
template <typename I>
static SD_INLINE SD_DEVICE LongType bsrFindBlockCol(const I* bsrColIdx, I clo, I chi, I bc) {
  while (clo < chi) {
    const I cmid = clo + (chi - clo) / 2;
    if (bsrColIdx[cmid] < bc) clo = cmid + 1;
    else chi = cmid;
  }
  return static_cast<LongType>(clo);
}

// ═══════════════════════════════════════════════════════════════════════════
// Section A: bsr_to_dense_bp
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void bsrToDenseBpKernel(
    const I* bsrColIdx, const I* bsrRowPtr,
    const X* gradDense, X* dBsrValues,
    LongType nnzb, LongType bd, LongType mb,
    LongType gdStride0, LongType gdStride1) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType total = nnzb * bd * bd;
  if (e >= total) return;

  const LongType block_k = e / (bd * bd);
  const LongType inner   = e % (bd * bd);
  const LongType lr      = inner / bd;
  const LongType lc      = inner % bd;

  // Find block-row br by binary-searching bsrRowPtr for block_k.
  const LongType br = bsrFindBlock(bsrRowPtr, mb, block_k);
  const LongType bc = static_cast<LongType>(bsrColIdx[block_k]);

  const LongType dr = br * bd + lr;
  const LongType dc = bc * bd + lc;
  dBsrValues[e] = gradDense[dr * gdStride0 + dc * gdStride1];
}

template <typename X, typename I>
static void bsrToDenseBpCuda_(
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradDense, NDArray& dBsrValues,
    LongType rows, LongType cols, LongType blockDim) {
  const LongType bd   = blockDim;
  const LongType mb   = rows / bd;
  const LongType nnzb = bsrColIdx.lengthOf();

  const I* bCI  = reinterpret_cast<const I*>(bsrColIdx.specialBuffer());
  const I* bRP  = reinterpret_cast<const I*>(bsrRowPtr.specialBuffer());
  const X* gD   = reinterpret_cast<const X*>(gradDense.specialBuffer());
  X*       dBV  = reinterpret_cast<X*>(dBsrValues.specialBuffer());

  const LongType gdS0 = gradDense.stridesOf()[0];
  const LongType gdS1 = gradDense.stridesOf()[1];

  const LongType total     = nnzb * bd * bd;
  const int      blockSize = 256;
  const int      gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  auto* stream = dBsrValues.getContext()->getCudaStream();
  bsrToDenseBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      bCI, bRP, gD, dBV, nnzb, bd, mb, gdS0, gdS1);
}

void bsr_to_dense_bp(
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradDense, NDArray& dBsrValues,
    LongType rows, LongType cols, LongType blockDim) {
  NDArray::prepareSpecialUse({&dBsrValues}, {&bsrColIdx, &bsrRowPtr, &gradDense});

  BUILD_DOUBLE_SELECTOR(gradDense.dataType(), bsrColIdx.dataType(), bsrToDenseBpCuda_,
                        (bsrColIdx, bsrRowPtr, gradDense, dBsrValues, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dBsrValues}, {&bsrColIdx, &bsrRowPtr, &gradDense});
}

// ═══════════════════════════════════════════════════════════════════════════
// Section B: bsr_spmm_bp
//
// Forward:  C[rows, n] = A_bsr[rows, cols] * B[cols, n]
// Backward:
//   dBsrValues[e] = gradC[br*bd+lr, :] · B[bc*bd+lc, :]   (dot over n_cols)
//   dB[bc*bd+lc, n] += bsrValues[e] * gradC[br*bd+lr, n]  (atomicAdd)
// ═══════════════════════════════════════════════════════════════════════════

// Kernel for dBsrValues: one thread per BSR value element.
template <typename X, typename I>
static SD_KERNEL void bsrSpmmBpDValKernel(
    const I* bsrColIdx, const I* bsrRowPtr,
    const X* B, const X* gradC, X* dBsrValues,
    LongType nnzb, LongType bd, LongType mb, LongType n_cols,
    LongType bS0, LongType bS1, LongType gcS0, LongType gcS1) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnzb * bd * bd) return;

  const LongType block_k = e / (bd * bd);
  const LongType inner   = e % (bd * bd);
  const LongType lr      = inner / bd;
  const LongType lc      = inner % bd;

  const LongType br = bsrFindBlock(bsrRowPtr, mb, block_k);
  const LongType bc = static_cast<LongType>(bsrColIdx[block_k]);

  const LongType grow = br * bd + lr;   // row in gradC
  const LongType bcol = bc * bd + lc;   // row in B (B is [cols, n])

  X acc = static_cast<X>(0);
  for (LongType n = 0; n < n_cols; ++n) {
    acc += gradC[grow * gcS0 + n * gcS1] * B[bcol * bS0 + n * bS1];
  }
  dBsrValues[e] = acc;
}

// Kernel for dB: one thread per BSR value element, atomicAdd into dB.
template <typename X, typename I>
static SD_KERNEL void bsrSpmmBpDBKernel(
    const I* bsrColIdx, const I* bsrRowPtr,
    const X* bsrValues, const X* gradC, X* dB,
    LongType nnzb, LongType bd, LongType mb, LongType n_cols,
    LongType gcS0, LongType gcS1, LongType dbS0, LongType dbS1) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnzb * bd * bd) return;

  const LongType block_k = e / (bd * bd);
  const LongType inner   = e % (bd * bd);
  const LongType lr      = inner / bd;
  const LongType lc      = inner % bd;

  const LongType br = bsrFindBlock(bsrRowPtr, mb, block_k);
  const LongType bc = static_cast<LongType>(bsrColIdx[block_k]);

  const LongType grow = br * bd + lr;   // row in gradC
  const LongType bcol = bc * bd + lc;   // row in B / row in dB

  const X aval = bsrValues[e];
  for (LongType n = 0; n < n_cols; ++n) {
    sd::math::atomics::sd_atomicAdd(
        &dB[bcol * dbS0 + n * dbS1],
        static_cast<X>(aval * gradC[grow * gcS0 + n * gcS1]));
  }
}

template <typename X, typename I>
static void bsrSpmmBpCuda_(
    NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& B, NDArray& gradC,
    NDArray& dBsrValues, NDArray& dB,
    LongType rows, LongType cols, LongType blockDim) {
  const LongType bd     = blockDim;
  const LongType mb     = rows / bd;
  const LongType nnzb   = bsrColIdx.lengthOf();
  const LongType n_cols = B.sizeAt(1);

  const I* bCI  = reinterpret_cast<const I*>(bsrColIdx.specialBuffer());
  const I* bRP  = reinterpret_cast<const I*>(bsrRowPtr.specialBuffer());
  const X* bsrV = reinterpret_cast<const X*>(bsrValues.specialBuffer());
  const X* bBuf = reinterpret_cast<const X*>(B.specialBuffer());
  const X* gC   = reinterpret_cast<const X*>(gradC.specialBuffer());
  X*       dBV  = reinterpret_cast<X*>(dBsrValues.specialBuffer());
  X*       dBBuf = reinterpret_cast<X*>(dB.specialBuffer());

  const LongType bS0  = B.stridesOf()[0];
  const LongType bS1  = B.stridesOf()[1];
  const LongType gcS0 = gradC.stridesOf()[0];
  const LongType gcS1 = gradC.stridesOf()[1];
  const LongType dbS0 = dB.stridesOf()[0];
  const LongType dbS1 = dB.stridesOf()[1];

  const LongType totalElems = nnzb * bd * bd;
  const int      blockSize  = 256;
  const int      gridSize   = static_cast<int>((totalElems + blockSize - 1) / blockSize);

  auto* stream = dBsrValues.getContext()->getCudaStream();

  // dBsrValues kernel
  bsrSpmmBpDValKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      bCI, bRP, bBuf, gC, dBV,
      nnzb, bd, mb, n_cols,
      bS0, bS1, gcS0, gcS1);

  // dB kernel (atomicAdd scatter)
  bsrSpmmBpDBKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      bCI, bRP, bsrV, gC, dBBuf,
      nnzb, bd, mb, n_cols,
      gcS0, gcS1, dbS0, dbS1);
}

void bsr_spmm_bp(
    NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& B, NDArray& gradC,
    NDArray& dBsrValues, NDArray& dB,
    LongType rows, LongType cols, LongType blockDim) {
  NDArray::prepareSpecialUse({&dBsrValues, &dB},
                              {&bsrValues, &bsrColIdx, &bsrRowPtr, &B, &gradC});

  BUILD_DOUBLE_SELECTOR(gradC.dataType(), bsrColIdx.dataType(), bsrSpmmBpCuda_,
                        (bsrValues, bsrColIdx, bsrRowPtr, B, gradC,
                         dBsrValues, dB, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dBsrValues, &dB},
                               {&bsrValues, &bsrColIdx, &bsrRowPtr, &B, &gradC});
}

// ═══════════════════════════════════════════════════════════════════════════
// Section C: csr_to_bsr_bp
//
// One thread per CSR nonzero k.
// Find global row r → br, lr; col → bc, lc.
// Binary-search bsrColIdx slice for block-row br to get block_k.
// dCsrValues[k] = gradBsrValues[block_k*bd*bd + lr*bd + lc].
// Pure gather — no atomics.
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void csrToBsrBpKernel(
    const I* csrColIdx, const I* csrRowPtr,
    const I* bsrColIdx, const I* bsrRowPtr,
    const X* gradBsrValues, X* dCsrValues,
    LongType nnz, LongType rows, LongType bd) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  // Find global row r of CSR entry k.
  const LongType rows_count = rows;
  LongType lo = 0, hi = rows_count - 1, r = 0;
  while (lo <= hi) {
    const LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(csrRowPtr[mid]) <= k &&
        k < static_cast<LongType>(csrRowPtr[mid + 1])) {
      r = mid;
      break;
    } else if (static_cast<LongType>(csrRowPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  const LongType col = static_cast<LongType>(csrColIdx[k]);
  const LongType br  = r   / bd;
  const LongType lr  = r   % bd;
  const LongType bc  = col / bd;
  const LongType lc  = col % bd;

  // Binary-search bsrColIdx[bsrRowPtr[br] .. bsrRowPtr[br+1]) for bc.
  I clo = bsrRowPtr[br], chi = bsrRowPtr[br + 1];
  const LongType block_k = bsrFindBlockCol(bsrColIdx, clo, chi, static_cast<I>(bc));

  dCsrValues[k] = gradBsrValues[block_k * bd * bd + lr * bd + lc];
}

template <typename X, typename I>
static void csrToBsrBpCuda_(
    NDArray& csrColIdx, NDArray& csrRowPtr,
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradBsrValues, NDArray& dCsrValues,
    LongType rows, LongType cols, LongType blockDim) {
  const LongType nnz = csrColIdx.lengthOf();
  const LongType bd  = blockDim;

  const I* csrCI  = reinterpret_cast<const I*>(csrColIdx.specialBuffer());
  const I* csrRP  = reinterpret_cast<const I*>(csrRowPtr.specialBuffer());
  const I* bCI    = reinterpret_cast<const I*>(bsrColIdx.specialBuffer());
  const I* bRP    = reinterpret_cast<const I*>(bsrRowPtr.specialBuffer());
  const X* gBV    = reinterpret_cast<const X*>(gradBsrValues.specialBuffer());
  X*       dCV    = reinterpret_cast<X*>(dCsrValues.specialBuffer());

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  auto* stream = dCsrValues.getContext()->getCudaStream();
  csrToBsrBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      csrCI, csrRP, bCI, bRP, gBV, dCV, nnz, rows, bd);
}

void csr_to_bsr_bp(
    NDArray& csrColIdx, NDArray& csrRowPtr,
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradBsrValues, NDArray& dCsrValues,
    LongType rows, LongType cols, LongType blockDim) {
  NDArray::prepareSpecialUse({&dCsrValues},
                              {&csrColIdx, &csrRowPtr, &bsrColIdx, &bsrRowPtr, &gradBsrValues});

  BUILD_DOUBLE_SELECTOR(gradBsrValues.dataType(), csrColIdx.dataType(), csrToBsrBpCuda_,
                        (csrColIdx, csrRowPtr, bsrColIdx, bsrRowPtr,
                         gradBsrValues, dCsrValues, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dCsrValues},
                               {&csrColIdx, &csrRowPtr, &bsrColIdx, &bsrRowPtr, &gradBsrValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
