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

#include <helpers/CutlassGemmHelper.h>
#include <helpers/CutlassHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/DebugHelper.h>
#include <config.h>

#if HAVE_CUTLASS

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float8.h>

#include <helpers/PointersManager.h>

namespace sd {

// =============================================================================
// CUTLASS GEMM kernel instantiations
// =============================================================================

// Helper: map NDArray ordering to CUTLASS layout
// NDArray 'c' order = row-major, 'f' order = column-major
// CUTLASS expects column-major by default for cuBLAS compatibility.
// We handle transposition by swapping A/B when needed.

namespace {

// ---- FP32 GEMM (SM80+) ----
using GemmFP32 = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    float,                           // ElementB
    cutlass::layout::RowMajor,       // LayoutB
    float,                           // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator
    cutlass::arch::OpClassSimt,      // OperatorClass (CUDA cores)
    cutlass::arch::Sm80              // ArchTag
>;

// ---- FP16 GEMM (SM80+, tensor cores) ----
using GemmFP16 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                 // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    cutlass::half_t,                 // ElementB
    cutlass::layout::RowMajor,       // LayoutB
    cutlass::half_t,                 // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator (FP32 accumulation)
    cutlass::arch::OpClassTensorOp,  // OperatorClass (tensor cores)
    cutlass::arch::Sm80              // ArchTag
>;

// ---- BF16 GEMM (SM80+, tensor cores) ----
using GemmBF16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,             // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    cutlass::bfloat16_t,             // ElementB
    cutlass::layout::RowMajor,       // LayoutB
    cutlass::bfloat16_t,             // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator
    cutlass::arch::OpClassTensorOp,  // OperatorClass
    cutlass::arch::Sm80              // ArchTag
>;

// FP8 E4M3 GEMM (SM89+ Ada Lovelace) is only available when building
// with -gencode compute_89 or higher. On SM86 builds the CUTLASS Sm89
// template specializations are incomplete and cannot be instantiated.
// The runtime shouldUseCutlass() already returns false for FP8 on SM < 89,
// so the FP8 dispatch path below simply returns false when not compiled in.

template <typename CutlassGemmOp>
bool runCutlassGemm(NDArray* A, NDArray* B, NDArray* C,
                    double alpha, double beta) {
  using ElementA = typename CutlassGemmOp::ElementA;
  using ElementB = typename CutlassGemmOp::ElementB;
  using ElementC = typename CutlassGemmOp::ElementC;

  const auto M = static_cast<int>(A->sizeAt(0));
  const auto K = static_cast<int>(A->sizeAt(1));
  const auto N = static_cast<int>(B->sizeAt(1));

  // Leading dimensions for row-major layout
  int lda = static_cast<int>(A->strideAt(0));
  int ldb = static_cast<int>(B->strideAt(0));
  int ldc = static_cast<int>(C->strideAt(0));

  // For row-major, lda should be >= K, ldb >= N, ldc >= N
  // NDArray strides are in elements, which is what CUTLASS expects
  if (lda < K) lda = K;
  if (ldb < N) ldb = N;
  if (ldc < N) ldc = N;

  typename CutlassGemmOp::Arguments args(
      {M, N, K},                                          // problem_size
      {reinterpret_cast<const ElementA*>(A->specialBuffer()), lda},  // ref_A
      {reinterpret_cast<const ElementB*>(B->specialBuffer()), ldb},  // ref_B
      {reinterpret_cast<ElementC*>(C->specialBuffer()), ldc},        // ref_C
      {reinterpret_cast<ElementC*>(C->specialBuffer()), ldc},        // ref_D (output)
      {static_cast<ElementC>(alpha), static_cast<ElementC>(beta)}    // epilogue
  );

  CutlassGemmOp gemm_op;
  auto status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  // Get workspace size
  size_t workspace_size = gemm_op.get_workspace_size(args);
  void* workspace = nullptr;
  // Get the CUDA stream from the NDArray context
  auto stream = A->getContext()->getCudaStream();

  int deviceId = sd::AffinityManager::currentDeviceId();
  if (workspace_size > 0) {
    workspace = sd::memory::CudaMemoryPool::getInstance().allocate(workspace_size, deviceId, *stream);
  }

  status = gemm_op.initialize(args, workspace, *stream);
  if (status != cutlass::Status::kSuccess) {
    if (workspace) sd::memory::CudaMemoryPool::getInstance().free(workspace, deviceId, *stream);
    return false;
  }

  status = gemm_op(*stream);

  if (workspace) sd::memory::CudaMemoryPool::getInstance().free(workspace, deviceId, *stream);

  return status == cutlass::Status::kSuccess;
}

}  // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

bool CutlassGemmHelper::shouldUseCutlass(LongType M, LongType N, LongType K,
                                         DataType aType, DataType bType,
                                         int smVersion) {
  // CUTLASS is not beneficial for GEMV (M=1) — cuBLAS GEMV is faster
  if (M <= 1) return false;

  // Must have SM80+ minimum
  if (smVersion < 80) return false;

  // FP8 requires SM89+
  if ((aType == DataType::FLOAT8 || bType == DataType::FLOAT8) && smVersion < 89) return false;

  // Only support matching types for now
  if (aType != bType) return false;

  // Supported type combinations
  switch (aType) {
    case DataType::FLOAT32:
    case DataType::HALF:
    case DataType::BFLOAT16:
      return true;
    case DataType::FLOAT8:
      return smVersion >= 89;
    default:
      return false;
  }
}

bool CutlassGemmHelper::gemm(NDArray* A, NDArray* B, NDArray* C,
                              double alpha, double beta) {
  if (A == nullptr || B == nullptr || C == nullptr) return false;
  if (A->rankOf() != 2 || B->rankOf() != 2 || C->rankOf() != 2) return false;

  // CUTLASS RowMajor kernels require contiguous row-major layout: strideAt(1)==1.
  // Permuted views (e.g. weight.permute(1,0)) have strideAt(0)==1 (column-major).
  // Dispatching such views to a RowMajor kernel reads elements at wrong offsets,
  // producing incorrect results (e.g., 7x attenuated matmul output).
  if (A->strideAt(1) != 1 || B->strideAt(1) != 1 || C->strideAt(1) != 1) return false;

  // CUTLASS RowMajor also requires the LEADING (row) stride to equal the column count — i.e.
  // fully contiguous, no row padding and not a window/slice of a larger tensor. runCutlassGemm
  // takes lda/ldb/ldc = strideAt(0) and only clamps them UP (if (lda < K) lda = K), never down.
  // A view whose strideAt(0) > sizeAt(1) (e.g. a reshape/slice view over a weight that lives in
  // a larger allocation) makes CUTLASS read M*strideAt(0) elements and overrun the operand's
  // actual extent → CUDA err700 (the close-weight-between-replays root: a view over the rebound
  // weight read ~9.7MB past its buffer). Reject such operands; cuBLAS handles a non-unit leading
  // dimension correctly via its explicit lda.
  if (A->strideAt(0) != A->sizeAt(1) || B->strideAt(0) != B->sizeAt(1) ||
      C->strideAt(0) != C->sizeAt(1)) {
    return false;
  }

  // Ensure data is on device
  NDArray::prepareSpecialUse({C}, {A, B});

  bool result = false;
  auto aType = A->dataType();

  switch (aType) {
    case DataType::FLOAT32:
      result = runCutlassGemm<GemmFP32>(A, B, C, alpha, beta);
      break;
    case DataType::HALF:
      result = runCutlassGemm<GemmFP16>(A, B, C, alpha, beta);
      break;
    case DataType::BFLOAT16:
      result = runCutlassGemm<GemmBF16>(A, B, C, alpha, beta);
      break;
    case DataType::FLOAT8:
      // FP8 CUTLASS GEMM requires SM89+ build target (compute_89 gencode).
      // shouldUseCutlass() already gates this at runtime.
      result = false;
      break;
    default:
      result = false;
      break;
  }

  if (result) {
    NDArray::registerSpecialUse({C}, {A, B});
  }

  return result;
}

}  // namespace sd

#else  // !HAVE_CUTLASS

namespace sd {

bool CutlassGemmHelper::shouldUseCutlass(LongType M, LongType N, LongType K,
                                         DataType aType, DataType bType,
                                         int smVersion) {
  return false;
}

bool CutlassGemmHelper::gemm(NDArray* A, NDArray* B, NDArray* C,
                              double alpha, double beta) {
  return false;
}

}  // namespace sd

#endif  // HAVE_CUTLASS
