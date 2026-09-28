/* SPDX-License-Identifier: Apache-2.0 */
//
// Weight-only quantized GEMM on tensor cores (ADR 0123).
//
// z[rows, N] = x[rows, K] . dequant(W)[N, K]^T with mma.m16n8k16 (BF16/FP16
// inputs, FP32 accumulation) through CUTLASS's SM80 warp-level primitive, which
// sm_80 and later — including sm_121 — execute. A weight-format policy loads a
// lane's packed weights and dequantizes them in registers with the format's
// exact arithmetic; the mainloop is format-independent.
//
#include <helpers/WeightOnlyGemm.h>

#include <config.h>
#include <execution/AffinityManager.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/CutlassHelper.h>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <ops/declarable/helpers/modelopt_linear.h>

#include <string>

#if HAVE_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>
#endif

#if HAVE_CUTLASS && NOT_EXCLUDED(OP_modelopt_nvfp4_linear)
#define SD_WEIGHT_ONLY_GEMM_AVAILABLE 1
#else
#define SD_WEIGHT_ONLY_GEMM_AVAILABLE 0
#endif

namespace sd {

#if SD_WEIGHT_ONLY_GEMM_AVAILABLE

// Activation dtypes the MMA consumes unchanged.
#define SD_WEIGHT_ONLY_MMA_TYPES SKIP_FIRST_COMMA(TTYPE_HALF TTYPE_BFLOAT)

using sd::device::WARP_SIZE;

// ─── Tile geometry (algorithm constants, ADR 0123) ──────────────────────────
//
// MMA fragment ownership (PTX m16n8k16): lane / 4 ("group") selects the A row
// and the B column, lane % 4 ("member") selects K positions. The 4 lanes of a
// group consume a chunk of kChunkDepth K elements; member m owns the contiguous
// K range [32m, 32m + 32) of the chunk, and in MMA step s the fragment K
// positions 2m, 2m+1, 2m+8, 2m+9 carry the member's K elements 4s .. 4s+3. The
// same permutation is applied to A and B, so the MMA forms exactly the same
// products; the permutation only fixes the (deterministic, row-independent)
// accumulation order. It lets a lane read its weights with one coalesced
// 16-byte load per chunk and its activations with one 8-byte load per row and
// step.
static constexpr int kMmaRows = 16;
static constexpr int kMmaColumns = 8;
static constexpr int kMmaDepth = 16;
static constexpr int kLaneDepth = 32;
static constexpr int kChunkDepth = 4 * kLaneDepth;
static constexpr int kStepsPerChunk = kChunkDepth / kMmaDepth;
// Each warp computes kColumnTilesPerWarp adjacent 8-column tiles over its own
// fixed K range; the kSplitK warps of a block cover the whole K and reduce
// their partial tiles through shared memory in a fixed order. Decode GEMVs are
// latency-bound, so many small blocks beat register reuse: on GB10, 1 tile x 8
// warps reached 166-253 GB/s at the Qwen3.6-27B shapes against 130-195 GB/s for
// 4 x 4 and 80-112 GB/s for 8 x 4.
static constexpr int kColumnTilesPerWarp = 1;
static constexpr int kSplitK = 8;
static constexpr int kBlockThreads = kSplitK * WARP_SIZE;
static constexpr int kColumnsPerBlock = kColumnTilesPerWarp * kMmaColumns;

template <typename X>
struct TensorCoreElement;
template <>
struct TensorCoreElement<bfloat16> {
  using type = cutlass::bfloat16_t;
};
template <>
struct TensorCoreElement<float16> {
  using type = cutlass::half_t;
};

// ─── Weight-format policies ─────────────────────────────────────────────────
//
// A format provides: Constants prepare() — per-thread constants, validated on
// the stream; Packed fetch(column, k) — the raw storage of the lane's K range
// [k, k + kLaneDepth) for one output column, loaded without being consumed so
// the mainloop can prefetch it a chunk ahead; LaneWeights unpack(packed) — its
// validated, decoded metadata; and dequantize(lane, step, constants, weights) —
// the lane's K elements 4*step .. 4*step+3 in the format's exact arithmetic, as
// tensor-core elements.

// ModelOpt NVFP4 (ADR 0122): FP32 block x global scale, FP32 E2M1 x scale
// (modelOptNvfp4WeightFp32), then one round-to-nearest-even into the tensor-core
// element — the hardware conversion, bit-identical to the X rounding of
// modelOptNvfp4Weight<X> in the native kernels for every finite value (scales
// are validated finite, so the product is never NaN).
template <typename X>
struct ModelOptNvfp4Weights {
  const uint8_t* packed;     // [N, K/2], even K in the low nibble
  const float8* blockScales;  // [N, K/16]
  const float* globalScale;   // device scalar
  LongType depth;

  using Constants = float;

  struct Packed {
    uint4 packed;       // 32 FP4 values
    uint16_t scales;    // the E4M3 scales of the two 16-element blocks they span
  };

  struct LaneWeights {
    uint4 packed;
    float blockScale[2];
  };

  SD_DEVICE Constants prepare() const { return ops::helpers::modelOptScaleChecked(globalScale[0]); }

  // k is a multiple of kLaneDepth and K of 32 (admission), so the lane's two
  // scales are adjacent at an even index: one 2-byte load.
  SD_DEVICE Packed fetch(LongType column, LongType k) const {
    Packed raw;
    raw.packed = *reinterpret_cast<const uint4*>(packed + column * (depth / 2) + k / 2);
    raw.scales = *reinterpret_cast<const uint16_t*>(blockScales + column * (depth / 16) + k / 16);
    return raw;
  }

  SD_DEVICE LaneWeights unpack(const Packed& raw) const {
    LaneWeights lane;
    lane.packed = raw.packed;
    CUTLASS_PRAGMA_UNROLL
    for (int block = 0; block < 2; ++block) {
      quarter_e4m3 scale;
      scale.x = static_cast<unsigned char>(raw.scales >> (8 * block));
      lane.blockScale[block] = ops::helpers::modelOptScaleChecked(cpu_e4m3_2float(scale));
    }
    return lane;
  }

  template <typename Element>
  SD_DEVICE void dequantize(const LaneWeights& lane, int step, Constants global, Element (&weights)[4]) const {
    // Elements 4*step .. 4*step+3 are the bytes 2*step and 2*step+1 of the lane's 16.
    const uint32_t word = reinterpret_cast<const uint32_t*>(&lane.packed)[step / 2];
    const uint32_t nibbles = word >> (16 * (step % 2));
    const float blockScale = lane.blockScale[step / 4];
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < 4; ++e)
      weights[e] = Element(ops::helpers::modelOptNvfp4WeightFp32(
          static_cast<unsigned char>((nibbles >> (4 * e)) & 15), blockScale, global));
  }
};

// ─── Mainloop ───────────────────────────────────────────────────────────────

// One lane's raw storage for one chunk: its K range [k, k + kLaneDepth) in each
// of the warp's column tiles (the column is the lane's group within the tile).
template <typename Weights>
struct ChunkStorage {
  typename Weights::Packed lanes[kColumnTilesPerWarp];
  bool active[kColumnTilesPerWarp] = {};
};

template <typename Weights>
SD_DEVICE static ChunkStorage<Weights> fetchChunk(const Weights& weights, LongType blockColumn, LongType columns,
                                                  LongType chunk, LongType depth, int group, int member) {
  ChunkStorage<Weights> storage;
  const LongType k = chunk * kChunkDepth + member * kLaneDepth;
  CUTLASS_PRAGMA_UNROLL
  for (int c = 0; c < kColumnTilesPerWarp; ++c) {
    const LongType column = blockColumn + c * kMmaColumns + group;
    storage.active[c] = k < depth && column < columns;
    if (storage.active[c]) storage.lanes[c] = weights.fetch(column, k);
  }
  return storage;
}

// The launch bound fixes the block size and asks for kMinBlocksPerSm resident
// blocks, so the compiler budgets registers for occupancy (<= 64 per thread).
// Unbounded, the kernel compiled to 90 registers (2 resident blocks per SM) and
// ran at half the throughput of the same code at 52: decode GEMVs are
// latency-bound and need the warps. (A 16-warp split-K for the few-wave long-K
// shape, and two column tiles per warp, were both measured slower on GB10.)
static constexpr int kMinBlocksPerSm = 4;

template <typename X, typename Weights>
SD_KERNEL static __launch_bounds__(kBlockThreads, kMinBlocksPerSm) void weightOnlyGemmKernel(
    const X* __restrict__ x, const Weights weights, void* __restrict__ z, LongType rows, LongType columns,
    LongType depth, bool floatOutput) {
  using Element = typename TensorCoreElement<X>::type;
  using Mma = cutlass::arch::Mma<cutlass::gemm::GemmShape<kMmaRows, kMmaColumns, kMmaDepth>, WARP_SIZE, Element,
                                 cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor, float,
                                 cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
  __shared__ float partials[kSplitK][kColumnTilesPerWarp][WARP_SIZE][4];

  const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int group = lane / 4;
  const int member = lane % 4;
  const typename Weights::Constants constants = weights.prepare();

  // The warp's K range depends only on K: identical for every row count.
  const LongType chunks = (depth + kChunkDepth - 1) / kChunkDepth;
  const LongType chunkBegin = warp * chunks / kSplitK;
  const LongType chunkEnd = (warp + 1) * chunks / kSplitK;

  // Block-uniform loops: every thread reaches every barrier.
  for (LongType blockColumn = static_cast<LongType>(blockIdx.x) * kColumnsPerBlock; blockColumn < columns;
       blockColumn += static_cast<LongType>(gridDim.x) * kColumnsPerBlock) {
    for (LongType rowBase = 0; rowBase < rows; rowBase += kMmaRows) {
      const LongType rowLow = rowBase + group;
      const LongType rowHigh = rowLow + kMmaRows / 2;
      typename Mma::FragmentC accumulators[kColumnTilesPerWarp];
      CUTLASS_PRAGMA_UNROLL
      for (auto& accumulator : accumulators) accumulator.clear();

      // Software pipeline: chunk c + 1's storage is in flight while chunk c is
      // dequantized and multiplied, so each warp keeps a weight load
      // outstanding instead of exposing the full memory latency per chunk.
      ChunkStorage<Weights> next;
      if (chunkBegin < chunkEnd)
        next = fetchChunk(weights, blockColumn, columns, chunkBegin, depth, group, member);
      for (LongType chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
        const LongType k = chunk * kChunkDepth + member * kLaneDepth;
        const bool laneActive = k < depth;
        const ChunkStorage<Weights> fetched = next;
        if (chunk + 1 < chunkEnd)
          next = fetchChunk(weights, blockColumn, columns, chunk + 1, depth, group, member);
        typename Weights::LaneWeights current[kColumnTilesPerWarp];
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kColumnTilesPerWarp; ++c)
          if (fetched.active[c]) current[c] = weights.unpack(fetched.lanes[c]);

        CUTLASS_PRAGMA_UNROLL
        for (int step = 0; step < kStepsPerChunk; ++step) {
          // A fragment words: 0 = row g, 1 = row g+8 (K positions 2m, 2m+1);
          // 2 = row g, 3 = row g+8 (K positions 2m+8, 2m+9).
          typename Mma::FragmentA a;
          auto* aWords = reinterpret_cast<uint32_t*>(&a);
          const uint2 low = laneActive && rowLow < rows
                                ? *reinterpret_cast<const uint2*>(x + rowLow * depth + k + 4 * step)
                                : make_uint2(0, 0);
          const uint2 high = laneActive && rowHigh < rows
                                 ? *reinterpret_cast<const uint2*>(x + rowHigh * depth + k + 4 * step)
                                 : make_uint2(0, 0);
          aWords[0] = low.x;
          aWords[1] = high.x;
          aWords[2] = low.y;
          aWords[3] = high.y;

          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < kColumnTilesPerWarp; ++c) {
            // B fragment: elements 0,1 at K positions 2m, 2m+1; 2,3 at 2m+8, 2m+9.
            Element dequantized[4] = {Element(0.0f), Element(0.0f), Element(0.0f), Element(0.0f)};
            if (fetched.active[c]) weights.dequantize(current[c], step, constants, dequantized);
            typename Mma::FragmentB b;
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < 4; ++e) b[e] = dequantized[e];
            Mma()(accumulators[c], a, b, accumulators[c]);
          }
        }
      }

      // Fixed-order split-K reduction: warp w sums column tiles w, w + kSplitK,
      // ... over the warps' partials in ascending warp order.
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < kColumnTilesPerWarp; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) partials[warp][c][lane][i] = accumulators[c][i];
      }
      __syncthreads();

      for (int tile = warp; tile < kColumnTilesPerWarp; tile += kSplitK) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          float total = partials[0][tile][lane][i];
          CUTLASS_PRAGMA_UNROLL
          for (int part = 1; part < kSplitK; ++part) total += partials[part][tile][lane][i];
          // C fragment: elements 0,1 at row g, 2,3 at row g+8; columns 2m, 2m+1.
          const LongType row = rowBase + group + (i / 2) * (kMmaRows / 2);
          const LongType column = blockColumn + tile * kMmaColumns + member * 2 + i % 2;
          if (row < rows && column < columns) {
            const LongType zOffset = row * columns + column;
            if (floatOutput)
              static_cast<float*>(z)[zOffset] = total;
            else
              static_cast<X*>(z)[zOffset] = static_cast<X>(total);
          }
        }
      }
      __syncthreads();  // partials are rewritten by the next row tile
    }
  }
}

template <typename X>
static void weightOnlyGemmNvfp4_(LaunchContext* context, NDArray* x, NDArray* w, NDArray* blockScales,
                                 NDArray* globalScale, NDArray* z, bool floatOutput, unsigned int blocks) {
  const LongType depth = x->sizeAt(-1);
  const LongType rows = x->lengthOf() / depth;
  const LongType columns = w->sizeAt(0);
  const ModelOptNvfp4Weights<X> weights{static_cast<const uint8_t*>(w->specialBuffer()),
                                        static_cast<const float8*>(blockScales->specialBuffer()),
                                        static_cast<const float*>(globalScale->specialBuffer()), depth};
  weightOnlyGemmKernel<X><<<blocks, kBlockThreads, 0, *context->getCudaStream()>>>(
      static_cast<const X*>(x->specialBuffer()), weights, z->specialBuffer(), rows, columns, depth, floatOutput);
}

BUILD_SINGLE_TEMPLATE(void weightOnlyGemmNvfp4_,
    (LaunchContext* context, NDArray* x, NDArray* w, NDArray* blockScales, NDArray* globalScale, NDArray* z,
     bool floatOutput, unsigned int blocks),
    SD_WEIGHT_ONLY_MMA_TYPES);

static bool alignedTo16(NDArray* array) {
  return reinterpret_cast<uintptr_t>(array->specialBuffer()) % 16 == 0;
}

#endif  // SD_WEIGHT_ONLY_GEMM_AVAILABLE

bool WeightOnlyGemm::isAdmitted(WeightOnlyFormat format, NDArray* x, NDArray* w, NDArray* blockScales,
                                NDArray* z) {
#if SD_WEIGHT_ONLY_GEMM_AVAILABLE
  if (format != WeightOnlyFormat::MODELOPT_NVFP4) return false;
  if (CutlassHelper::getSmVersion(AffinityManager::currentDeviceId()) < 80) return false;
  if (x->dataType() != BFLOAT16 && x->dataType() != HALF) return false;
  if (x->isEmpty() || w->isEmpty() || blockScales->isEmpty() || z->isEmpty()) return false;
  const LongType depth = x->sizeAt(-1);
  const LongType columns = w->sizeAt(0);
  // A lane owns 32 K elements (one 16-byte FP4 load spanning two scale
  // blocks); output columns come in whole 8-column MMA tiles.
  if (depth % kLaneDepth != 0 || columns % kMmaColumns != 0) return false;
  if (!shape::isDenseRowMajor(x->shapeInfo()) || !shape::isDenseRowMajor(w->shapeInfo()) ||
      !shape::isDenseRowMajor(blockScales->shapeInfo()) || !shape::isDenseRowMajor(z->shapeInfo()))
    return false;
  // Scales are read as adjacent pairs (2-byte loads).
  return alignedTo16(x) && alignedTo16(w) && alignedTo16(z) &&
         reinterpret_cast<uintptr_t>(blockScales->specialBuffer()) % 2 == 0;
#else
  return false;
#endif
}

void WeightOnlyGemm::run(LaunchContext* context, WeightOnlyFormat format, NDArray* x, NDArray* w,
                         NDArray* blockScales, NDArray* globalScale, NDArray* z, bool floatOutput) {
#if SD_WEIGHT_ONLY_GEMM_AVAILABLE
  if (!isAdmitted(format, x, w, blockScales, z))
    THROW_EXCEPTION("WeightOnlyGemm::run: operands are not admitted to the tensor-core path");
  // The block size is fixed by the algorithm (kSplitK warps); the grid is a
  // cap, and blocks grid-stride over column groups.
  const dim3 dims = getLaunchDims("weight_only_gemm");
  if (dims.y != static_cast<unsigned int>(kBlockThreads))
    THROW_EXCEPTION(("WeightOnlyGemm: BLOCK_SIZE_WEIGHT_ONLY_GEMM must be " + std::to_string(kBlockThreads)).c_str());
  if (dims.x == 0) THROW_EXCEPTION("WeightOnlyGemm: GRID_SIZE_WEIGHT_ONLY_GEMM must be positive");
  const LongType needed = (w->sizeAt(0) + kColumnsPerBlock - 1) / kColumnsPerBlock;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  BUILD_SINGLE_SELECTOR(x->dataType(), weightOnlyGemmNvfp4_,
                        (context, x, w, blockScales, globalScale, z, floatOutput, blocks), SD_WEIGHT_ONLY_MMA_TYPES);
#else
  THROW_EXCEPTION("WeightOnlyGemm::run: built without CUTLASS or without modelopt_nvfp4_linear");
#endif
}

}  // namespace sd
