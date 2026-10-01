/* SPDX-License-Identifier: Apache-2.0 */
//
// Weight-only quantized GEMM on tensor cores (ADR 0123).
//
// z[rows, N] = x[rows, K] . dequant(W)[N, K]^T with mma.m16n8k16 through
// CUTLASS's SM80 warp-level primitive, which sm_80 and later — including sm_121
// — execute. Activation types that are tensor-core elements (TensorCoreElement)
// feed the MMA unchanged and accumulate in their aggregate type; z takes any
// float type. A weight-format policy loads a lane's packed weights and
// dequantizes them in registers with the format's exact arithmetic; the
// mainloop is format-independent.
//
// Two kernels share the mainloop, the reduction and the split combine, so they
// produce bit-identical results. The direct kernel has every lane load its
// weights from global memory. On sm_90 and later, the bulk kernel has one
// thread copy the unit's whole weight tile into shared memory with bulk
// asynchronous copies (cp.async.bulk, completed on an mbarrier); lanes then read
// their weights from shared memory. The launcher picks the bulk kernel for the
// shapes and row counts it was measured faster on (bulkStagingPays).
//
#include <helpers/WeightOnlyGemm.h>

#include <config.h>
#include <execution/AffinityManager.h>
#include <execution/cuda/LaunchDims.h>
#include <graph/DspDiagnostics.h>
#include <helpers/CutlassHelper.h>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <ops/declarable/helpers/modelopt_linear.h>
#include <ops/op_types.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

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

// Bulk asynchronous copies need sm_90 or later. The bulk kernel compiles to a
// trap for older targets; the host never launches that code (ptxVersion check).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define SD_WEIGHT_ONLY_BULK_DEVICE 1
#else
#define SD_WEIGHT_ONLY_BULK_DEVICE 0
#endif

namespace sd {

#if SD_WEIGHT_ONLY_GEMM_AVAILABLE

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

// The MMA's element for activation type X: the same bits as X, as CUTLASS
// spells them. X without a specialization is not a tensor-core element.
template <typename X>
struct TensorCoreElement {};
template <>
struct TensorCoreElement<bfloat16> {
  using type = cutlass::bfloat16_t;
};
template <>
struct TensorCoreElement<float16> {
  using type = cutlass::half_t;
};

template <typename X, typename = void>
struct IsTensorCoreElement : std::false_type {};
template <typename X>
struct IsTensorCoreElement<X, std::void_t<typename TensorCoreElement<X>::type>> : std::true_type {};

// The MMA accumulates, and the split-K partials are kept, in X's aggregate type.
template <typename X>
using WeightOnlyAccumulator = typename simdOps::AggregateType<X>::type;

template <typename X>
using WeightOnlyMma =
    cutlass::arch::Mma<cutlass::gemm::GemmShape<kMmaRows, kMmaColumns, kMmaDepth>, WARP_SIZE,
                       typename TensorCoreElement<X>::type, cutlass::layout::RowMajor,
                       typename TensorCoreElement<X>::type, cutlass::layout::ColumnMajor,
                       WeightOnlyAccumulator<X>, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

// ─── Bulk asynchronous copies (sm_90+) ──────────────────────────────────────
//
// One thread arms an mbarrier with the byte count of a tile, issues the tile's
// cp.async.bulk copies (global -> shared, completing on the barrier), and every
// thread waits on the barrier's phase. The copies read the weights once, so they
// carry an L2 evict-first policy.
#if SD_WEIGHT_ONLY_BULK_DEVICE
static SD_DEVICE SD_INLINE uint32_t bulkSharedAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

static SD_DEVICE SD_INLINE void bulkInitBarrier(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bulkSharedAddress(barrier)), "r"(1) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

static SD_DEVICE SD_INLINE void bulkExpectBytes(uint64_t* barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bulkSharedAddress(barrier)),
               "r"(bytes)
               : "memory");
}

// source and destination 16-byte aligned, bytes a multiple of 16.
static SD_DEVICE SD_INLINE void bulkCopy(void* destination, const void* source, uint32_t bytes, uint64_t* barrier,
                                        uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          bulkSharedAddress(destination)),
      "l"(source), "r"(bytes), "r"(bulkSharedAddress(barrier)), "l"(policy)
      : "memory");
}

static SD_DEVICE SD_INLINE void bulkWait(uint64_t* barrier, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::"r"(
          bulkSharedAddress(barrier)),
      "r"(parity)
      : "memory");
}

static SD_DEVICE SD_INLINE uint64_t bulkCopyPolicy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

// Orders this thread's earlier shared-memory accesses (generic proxy) before
// its later bulk copies into the same memory (async proxy).
static SD_DEVICE SD_INLINE void bulkFenceAsyncProxy() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
#endif  // SD_WEIGHT_ONLY_BULK_DEVICE

// ─── Weight-format policies ─────────────────────────────────────────────────
//
// A format provides: Constants prepare() — per-thread constants, validated on
// the stream; Packed fetch(column, k) — the raw storage of the lane's K range
// [k, k + kLaneDepth) for one output column, loaded without being consumed so
// the mainloop can prefetch it a chunk ahead; LaneWeights unpack(packed) — its
// validated, decoded metadata; and dequantize(lane, step, constants, weights) —
// the lane's K elements 4*step .. 4*step+3 in the format's exact arithmetic, as
// tensor-core elements. For the bulk kernel it also provides Tile tileFor(chunks)
// — the shared-memory layout of a unit's weights; stage(...) — the unit's bulk
// copies; and stagedLane(...) / staged(lane, chunk) — fetch() from that tile.

// ModelOpt NVFP4 (ADR 0122): each weight is ops::helpers::modelOptNvfp4Weight
// of its E2M1 code, block scale and global scale — the native kernels'
// arithmetic, with the one rounding into the tensor-core element (the hardware
// round to nearest even), as into X.
struct ModelOptNvfp4Weights {
  using Format = ops::helpers::ModelOptNvfp4;
  using Storage = Format::Storage;
  using ScaleStorage = Format::ScaleStorage;
  using Scale = Format::Scale;
  // A lane's kLaneDepth weights: kLaneStorage storage words spanning kLaneBlocks
  // scale blocks.
  static constexpr int kLaneStorage = kLaneDepth / Format::kWeightsPerStorage;
  static constexpr int kLaneBlocks = kLaneDepth / Format::kBlockLength;
  static_assert(kLaneDepth % Format::kBlockLength == 0, "a lane covers whole scale blocks");
  static_assert(kLaneStorage * sizeof(Storage) == sizeof(uint4), "a lane's codes are one 16-byte load");

  const Storage* packed;            // [N, K / kWeightsPerStorage], code 0 of a word in its low bits
  const ScaleStorage* blockScales;  // [N, K / kBlockLength]
  const Scale* globalScale;         // device scalar
  LongType depth;

  using Constants = Scale;

  // The scales of a lane's blocks, adjacent in their row: one load.
  struct alignas(kLaneBlocks * sizeof(ScaleStorage)) LaneScales {
    ScaleStorage values[kLaneBlocks];
  };

  struct Packed {
    uint4 packed;       // the lane's kLaneDepth codes
    LaneScales scales;  // the scales of the blocks they span
  };

  struct LaneWeights {
    uint4 packed;
    Scale blockScale[kLaneBlocks];
  };

  // Bytes of the storage, and of the scales, of `elements` consecutive weights
  // of a row.
  SD_HOST_DEVICE static constexpr LongType packedBytesOf(LongType elements) {
    return elements / Format::kWeightsPerStorage * static_cast<LongType>(sizeof(Storage));
  }
  SD_HOST_DEVICE static constexpr LongType scaleBytesOf(LongType elements) {
    return elements / Format::kBlockLength * static_cast<LongType>(sizeof(ScaleStorage));
  }

  SD_DEVICE const Storage* packedRow(LongType column, LongType k) const {
    return packed + column * (depth / Format::kWeightsPerStorage) + k / Format::kWeightsPerStorage;
  }

  SD_DEVICE const ScaleStorage* scaleRow(LongType column, LongType k) const {
    return blockScales + column * (depth / Format::kBlockLength) + k / Format::kBlockLength;
  }

  SD_DEVICE Constants prepare() const { return ops::helpers::modelOptScaleChecked(globalScale[0]); }

  // k and K are multiples of kLaneDepth (admission), so the lane's scales start
  // at a multiple of kLaneBlocks in the row: LaneScales-aligned.
  SD_DEVICE Packed fetch(LongType column, LongType k) const {
    Packed raw;
    raw.packed = *reinterpret_cast<const uint4*>(packedRow(column, k));
    raw.scales = *reinterpret_cast<const LaneScales*>(scaleRow(column, k));
    return raw;
  }

  SD_DEVICE LaneWeights unpack(const Packed& raw) const {
    LaneWeights lane;
    lane.packed = raw.packed;
    math::sd_convert_n<ScaleStorage, Scale, kLaneBlocks>(raw.scales.values, lane.blockScale);
    CUTLASS_PRAGMA_UNROLL
    for (int block = 0; block < kLaneBlocks; ++block)
      lane.blockScale[block] = ops::helpers::modelOptScaleChecked(lane.blockScale[block]);
    return lane;
  }

  // The lane's K elements 4*step .. 4*step+3: codes of one 32-bit word of its
  // storage, all in one scale block.
  template <typename Element>
  SD_DEVICE void dequantize(const LaneWeights& lane, int step, Constants global, Element (&weights)[4]) const {
    using Word = uint32_t;
    constexpr int kCodesPerWord = Format::Weight::codesPer<Word>();
    static_assert(kCodesPerWord % 4 == 0 && Format::kBlockLength % 4 == 0, "a step's codes share a word and a block");
    const Word word = reinterpret_cast<const Word*>(&lane.packed)[4 * step / kCodesPerWord];
    const Scale blockScale = lane.blockScale[4 * step / Format::kBlockLength];
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < 4; ++e)
      weights[e] = ops::helpers::modelOptNvfp4Weight<Element, Scale>(
          Format::Weight::unpack(word, (4 * step + e) % kCodesPerWord), blockScale, global);
  }

  // A unit's staged tile: its kColumnsPerBlock packed rows over the unit's K
  // range, then the rows' scales.
  struct Tile {
    LongType rowStride;    // bytes between the packed rows 2p and 2p + 2
    LongType scaleOffset;  // first scale row
    LongType scaleStride;  // bytes between scale rows
    LongType bytes;        // dynamic shared memory
  };

  // chunks: the most K chunks one unit spans.
  static Tile tileFor(LongType chunks) {
    Tile tile;
    tile.rowStride = packedBytesOf(chunks * kChunkDepth);
    tile.scaleOffset = kColumnsPerBlock * tile.rowStride + 64;
    // A scale row is copied as its 16-byte-aligned superset: at most 16 bytes
    // more than the row. An odd multiple of 16 bytes between rows puts the 8
    // rows a warp reads at once in distinct banks.
    tile.scaleStride = (scaleBytesOf(chunks * kChunkDepth) + 15) / 16 * 16 + 16;
    if ((tile.scaleStride / 16) % 2 == 0) tile.scaleStride += 16;
    tile.bytes = tile.scaleOffset + kColumnsPerBlock * tile.scaleStride;
    return tile;
  }

  // Rows 2p and 2p + 1 are 64 bytes apart modulo 128: a quarter-warp reads 64
  // contiguous bytes of each, and they fall in distinct banks.
  SD_HOST_DEVICE static LongType packedRowOffset(int row, LongType rowStride) {
    return (row >> 1) * rowStride + (row & 1) * (4 * rowStride + 64);
  }

#if SD_WEIGHT_ONLY_BULK_DEVICE
  // One thread: copies the rows [blockColumn, blockColumn + kColumnsPerBlock)
  // over K range [kb, ke) into the tile and arms ready with the byte count.
  // Packed rows are 16-byte aligned (K % kLaneDepth == 0 and a 16-byte-aligned
  // base). A scale row starts only LaneScales-aligned, so it is copied as its
  // 16-byte-aligned superset; with a 16-byte-aligned base the superset stays
  // inside the buffer, whose size is a multiple of 16 bytes.
  SD_DEVICE void stage(uint8_t* tile, const Tile& geometry, LongType blockColumn, LongType kb, LongType ke,
                       uint64_t* ready, uint64_t policy) const {
    const uint32_t packedBytes = static_cast<uint32_t>(packedBytesOf(ke - kb));
    const uint32_t scaleBytes = static_cast<uint32_t>(scaleBytesOf(ke - kb));
    uint32_t bytes = kColumnsPerBlock * packedBytes;
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < kColumnsPerBlock; ++row) {
      const uintptr_t first = reinterpret_cast<uintptr_t>(scaleRow(blockColumn + row, kb));
      bytes += static_cast<uint32_t>(((first + scaleBytes + 15) & ~uintptr_t(15)) - (first & ~uintptr_t(15)));
    }
    bulkExpectBytes(ready, bytes);
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < kColumnsPerBlock; ++row) {
      bulkCopy(tile + packedRowOffset(row, geometry.rowStride), packedRow(blockColumn + row, kb), packedBytes, ready,
               policy);
      const uintptr_t first = reinterpret_cast<uintptr_t>(scaleRow(blockColumn + row, kb));
      const uintptr_t begin = first & ~uintptr_t(15);
      bulkCopy(tile + geometry.scaleOffset + row * geometry.scaleStride, reinterpret_cast<const void*>(begin),
               static_cast<uint32_t>(((first + scaleBytes + 15) & ~uintptr_t(15)) - begin), ready, policy);
    }
  }
#endif

  // The lane's storage in a tile staged from K offset kb: its 16 bytes of its
  // row's packed chunk and its LaneScales, which sit (first scale & 15) bytes
  // into the row's superset.
  struct StagedLane {
    const uint8_t* packed;
    const uint8_t* scales;
  };

  SD_DEVICE StagedLane stagedLane(const uint8_t* tile, const Tile& geometry, LongType blockColumn, LongType kb,
                                  int group, int member) const {
    const uintptr_t head = reinterpret_cast<uintptr_t>(scaleRow(blockColumn + group, kb)) & 15;
    StagedLane lane;
    lane.packed = tile + packedRowOffset(group, geometry.rowStride) + member * packedBytesOf(kLaneDepth);
    lane.scales = tile + geometry.scaleOffset + group * geometry.scaleStride + head + member * scaleBytesOf(kLaneDepth);
    return lane;
  }

  // fetch() of the staged unit's chunk-th chunk.
  SD_DEVICE Packed staged(const StagedLane& lane, LongType chunk) const {
    Packed raw;
    raw.packed = *reinterpret_cast<const uint4*>(lane.packed + chunk * packedBytesOf(kChunkDepth));
    raw.scales = *reinterpret_cast<const LaneScales*>(lane.scales + chunk * scaleBytesOf(kChunkDepth));
    return raw;
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

// One chunk of one row tile: the lane's kLaneDepth K elements of rows g and
// g + 8 against its dequantized weights of each column tile, one MMA per step.
template <typename X, typename Weights>
SD_DEVICE SD_INLINE static void multiplyChunk(
    const X* __restrict__ x, const Weights& weights, const typename Weights::Constants constants,
    const typename Weights::LaneWeights (&current)[kColumnTilesPerWarp], const bool (&active)[kColumnTilesPerWarp],
    bool laneActive, LongType rowLow, LongType rowHigh, LongType rows, LongType depth, LongType k,
    typename WeightOnlyMma<X>::FragmentC (&accumulators)[kColumnTilesPerWarp]) {
  using Mma = WeightOnlyMma<X>;
  using Element = typename TensorCoreElement<X>::type;
  CUTLASS_PRAGMA_UNROLL
  for (int step = 0; step < kStepsPerChunk; ++step) {
    // A fragment words: 0 = row g, 1 = row g+8 (K positions 2m, 2m+1);
    // 2 = row g, 3 = row g+8 (K positions 2m+8, 2m+9).
    typename Mma::FragmentA a;
    auto* aWords = reinterpret_cast<uint32_t*>(&a);
    const uint2 low = laneActive && rowLow < rows ? *reinterpret_cast<const uint2*>(x + rowLow * depth + k + 4 * step)
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
      Element dequantized[4] = {};
      if (active[c]) weights.dequantize(current[c], step, constants, dequantized);
      typename Mma::FragmentB b;
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < 4; ++e) b[e] = dequantized[e];
      Mma()(accumulators[c], a, b, accumulators[c]);
    }
  }
}

// Fixed-order split-K reduction of one row tile: warp w sums column tiles w,
// w + kSplitK, ... over the warps' partials in ascending warp order, then stores
// the sums — for a split shape, as the block's partial in the accumulator type.
// Block-wide: every thread calls it.
template <typename X, typename Z>
SD_DEVICE SD_INLINE static void reduceRowTile(
    WeightOnlyAccumulator<X> (&partials)[kSplitK][kColumnTilesPerWarp][WARP_SIZE][4],
    const typename WeightOnlyMma<X>::FragmentC (&accumulators)[kColumnTilesPerWarp], Z* __restrict__ z,
    WeightOnlyAccumulator<X>* __restrict__ scratch, LongType rowBase, LongType blockColumn, LongType rows,
    LongType columns, int splitBlocks, int splitIndex, int warp, int lane) {
  const int group = lane / 4;
  const int member = lane % 4;
  CUTLASS_PRAGMA_UNROLL
  for (int c = 0; c < kColumnTilesPerWarp; ++c) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) partials[warp][c][lane][i] = accumulators[c][i];
  }
  __syncthreads();

  for (int tile = warp; tile < kColumnTilesPerWarp; tile += kSplitK) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      WeightOnlyAccumulator<X> total = partials[0][tile][lane][i];
      CUTLASS_PRAGMA_UNROLL
      for (int part = 1; part < kSplitK; ++part) total += partials[part][tile][lane][i];
      // C fragment: elements 0,1 at row g, 2,3 at row g+8; columns 2m, 2m+1.
      const LongType row = rowBase + group + (i / 2) * (kMmaRows / 2);
      const LongType column = blockColumn + tile * kMmaColumns + member * 2 + i % 2;
      if (row < rows && column < columns) {
        const LongType zOffset = row * columns + column;
        if (splitBlocks > 1)
          scratch[static_cast<LongType>(splitIndex) * rows * columns + zOffset] = total;
        else
          z[zOffset] = static_cast<Z>(total);
      }
    }
  }
  __syncthreads();  // partials are rewritten by the next row tile
}

// Split shapes, after a unit's row tiles: publish the block's partials and take
// the column group's ticket; the last block to arrive (which resets the ticket)
// adds the partials in ascending split order. Block-wide.
//
// reduceRowTile's closing barrier orders the block's partial stores before
// thread 0 takes the ticket with a release RMW; the last arrival's acquire
// fence synchronizes with every split's release through the ticket, and the
// barrier after it extends that to the threads reading the scratch (the
// pattern of a grid barrier). A __threadfence in every thread instead
// compiles to a sequentially consistent fence plus an L1 invalidation per
// warp, which kept evicting the resident blocks' activation rows: the
// 5120 x 17408 down projection of the Qwen3.6-27B decode ran 8% slower
// (249 vs 230 us) whenever the L2 held the preceding GEMMs' lines.
template <typename AccT, typename Z>
SD_DEVICE SD_INLINE static void combineSplits(bool& lastArrival, unsigned int* __restrict__ tickets,
                                              const AccT* __restrict__ scratch, Z* __restrict__ z,
                                              LongType columnGroup, LongType blockColumn, LongType rows,
                                              LongType columns, int splitBlocks) {
  if (threadIdx.x == 0) {
    unsigned int ticket;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(ticket) : "l"(tickets + columnGroup) : "memory");
    lastArrival = ticket == static_cast<unsigned int>(splitBlocks - 1);
    if (lastArrival) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      tickets[columnGroup] = 0;
    }
  }
  __syncthreads();
  if (lastArrival) {
    const LongType tileColumns = columns - blockColumn < kColumnsPerBlock ? columns - blockColumn : kColumnsPerBlock;
    for (LongType e = threadIdx.x; e < rows * tileColumns; e += blockDim.x) {
      const LongType zOffset = (e / tileColumns) * columns + blockColumn + e % tileColumns;
      AccT total = __ldcg(scratch + zOffset);
      for (int split = 1; split < splitBlocks; ++split)
        total += __ldcg(scratch + static_cast<LongType>(split) * rows * columns + zOffset);
      z[zOffset] = static_cast<Z>(total);
    }
  }
  __syncthreads();  // lastArrival is rewritten by the next unit
}

// The launch bound fixes the block size and asks for kMinBlocksPerSm resident
// blocks, so the compiler budgets registers for occupancy (<= 64 per thread).
// Unbounded, the kernel compiled to 90 registers (2 resident blocks per SM) and
// ran at half the throughput of the same code at 52: decode GEMVs are
// latency-bound and need the warps. (A 16-warp split-K for the few-wave long-K
// shape, and two column tiles per warp, were both measured slower on GB10.)
static constexpr int kMinBlocksPerSm = 4;

// Few-wave shapes (the long-K down projection) leave most of the last wave of
// blocks idle. They split K across kSplitBlocks blocks per column group: each
// block reduces its warps as usual and stores its partial tile, in the
// accumulator type, in a persistent scratch buffer; the last block of the group to arrive (per-group
// ticket, reset by that block) adds the partials in ascending block order. The
// split depends only on the shape (never on the row count), so every row count
// of a shape shares one accumulation order.
static constexpr int kSplitBlocks = 4;

// A column group's splits are consecutive units, so the blocks running at once
// read whole rows of consecutive column groups. Split-major order (consecutive
// column groups of one split, each block reading a quarter of each row) ran the
// 5120x17408 down projection 12% slower in the direct kernel and 5% slower in
// the bulk kernel on GB10. The order only schedules units; no result depends
// on it.
SD_DEVICE SD_INLINE static LongType unitColumnGroup(LongType unit, int splitBlocks) { return unit / splitBlocks; }

SD_DEVICE SD_INLINE static int unitSplit(LongType unit, int splitBlocks) {
  return static_cast<int>(unit % splitBlocks);
}

template <typename X, typename Z, typename Weights>
SD_KERNEL static __launch_bounds__(kBlockThreads, kMinBlocksPerSm) void weightOnlyGemmKernel(
    const X* __restrict__ x, const Weights weights, Z* __restrict__ z, LongType rows, LongType columns,
    LongType depth, int splitBlocks, WeightOnlyAccumulator<X>* __restrict__ scratch,
    unsigned int* __restrict__ tickets) {
  using Mma = WeightOnlyMma<X>;
  __shared__ WeightOnlyAccumulator<X> partials[kSplitK][kColumnTilesPerWarp][WARP_SIZE][4];

  const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int group = lane / 4;
  const int member = lane % 4;
  const typename Weights::Constants constants = weights.prepare();

  __shared__ bool lastArrival;

  // The warp's K range depends only on K and the shape's split: identical for
  // every row count.
  const LongType chunks = (depth + kChunkDepth - 1) / kChunkDepth;
  const LongType parts = static_cast<LongType>(kSplitK) * splitBlocks;
  const LongType columnGroups = (columns + kColumnsPerBlock - 1) / kColumnsPerBlock;

  // Block-uniform loops: every thread reaches every barrier.
  for (LongType unit = blockIdx.x; unit < columnGroups * splitBlocks; unit += gridDim.x) {
    const LongType columnGroup = unitColumnGroup(unit, splitBlocks);
    const int splitIndex = unitSplit(unit, splitBlocks);
    const LongType blockColumn = columnGroup * kColumnsPerBlock;
    const LongType part = static_cast<LongType>(splitIndex) * kSplitK + warp;
    const LongType chunkBegin = part * chunks / parts;
    const LongType chunkEnd = (part + 1) * chunks / parts;
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
        multiplyChunk(x, weights, constants, current, fetched.active, laneActive, rowLow, rowHigh, rows, depth, k,
                      accumulators);
      }
      reduceRowTile<X, Z>(partials, accumulators, z, scratch, rowBase, blockColumn, rows, columns, splitBlocks,
                          splitIndex, warp, lane);
    }

    if (splitBlocks > 1)
      combineSplits(lastArrival, tickets, scratch, z, columnGroup, blockColumn, rows, columns, splitBlocks);
  }
}

// The same units, K ranges, reduction and split combine as weightOnlyGemmKernel
// — so bit-identical results — with each unit's weights staged in shared memory
// by bulk copies instead of loaded by the lanes. A unit's warps cover the K
// chunks [stageBegin, stageEnd) together; thread 0 copies those chunks of the
// unit's kColumnsPerBlock rows in one transaction group, and the block waits on
// it once per unit. Needs geometry.bytes of dynamic shared memory.
template <typename X, typename Z, typename Weights>
SD_KERNEL static __launch_bounds__(kBlockThreads, kMinBlocksPerSm) void weightOnlyGemmBulkKernel(
    const X* __restrict__ x, const Weights weights, Z* __restrict__ z, LongType rows, LongType columns,
    LongType depth, int splitBlocks, WeightOnlyAccumulator<X>* __restrict__ scratch,
    unsigned int* __restrict__ tickets, const typename Weights::Tile geometry) {
#if SD_WEIGHT_ONLY_BULK_DEVICE
  static_assert(kColumnTilesPerWarp == 1, "a staged tile holds one column tile");
  using Mma = WeightOnlyMma<X>;
  extern __shared__ __align__(128) uint8_t weightOnlyStage[];
  __shared__ __align__(8) uint64_t stageReady;
  __shared__ WeightOnlyAccumulator<X> partials[kSplitK][kColumnTilesPerWarp][WARP_SIZE][4];
  __shared__ bool lastArrival;

  const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int group = lane / 4;
  const int member = lane % 4;

  const LongType chunks = (depth + kChunkDepth - 1) / kChunkDepth;
  const LongType parts = static_cast<LongType>(kSplitK) * splitBlocks;
  const LongType columnGroups = (columns + kColumnsPerBlock - 1) / kColumnsPerBlock;
  const LongType units = columnGroups * splitBlocks;
  const uint64_t policy = bulkCopyPolicy();

  // The unit's staged K range: [stageBegin, stageEnd) chunks, the last one
  // possibly partial.
  auto stageRange = [&](LongType unit, LongType& stageBegin, LongType& kb, LongType& ke) {
    const LongType splitIndex = unitSplit(unit, splitBlocks);
    stageBegin = splitIndex * kSplitK * chunks / parts;
    const LongType stageEnd = (splitIndex + 1) * kSplitK * chunks / parts;
    kb = stageBegin * kChunkDepth;
    ke = stageEnd * kChunkDepth < depth ? stageEnd * kChunkDepth : depth;
  };

  // The first unit's copies overlap the constants and the barrier's publication.
  if (threadIdx.x == 0) {
    bulkInitBarrier(&stageReady);
    if (blockIdx.x < units) {
      LongType stageBegin, kb, ke;
      stageRange(blockIdx.x, stageBegin, kb, ke);
      weights.stage(weightOnlyStage, geometry, unitColumnGroup(blockIdx.x, splitBlocks) * kColumnsPerBlock, kb, ke,
                    &stageReady, policy);
    }
  }
  const typename Weights::Constants constants = weights.prepare();
  __syncthreads();

  uint32_t phase = 0;
  // Block-uniform loops: every thread reaches every barrier.
  for (LongType unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const LongType columnGroup = unitColumnGroup(unit, splitBlocks);
    const int splitIndex = unitSplit(unit, splitBlocks);
    const LongType blockColumn = columnGroup * kColumnsPerBlock;
    const LongType part = static_cast<LongType>(splitIndex) * kSplitK + warp;
    const LongType chunkBegin = part * chunks / parts;
    const LongType chunkEnd = (part + 1) * chunks / parts;
    LongType stageBegin, kb, ke;
    stageRange(unit, stageBegin, kb, ke);
    // The previous unit's last barrier ended every read of the tile.
    if (threadIdx.x == 0 && unit != blockIdx.x) {
      bulkFenceAsyncProxy();
      weights.stage(weightOnlyStage, geometry, blockColumn, kb, ke, &stageReady, policy);
    }
    const typename Weights::StagedLane staged =
        weights.stagedLane(weightOnlyStage, geometry, blockColumn, kb, group, member);
    bulkWait(&stageReady, phase);
    phase ^= 1;

    for (LongType rowBase = 0; rowBase < rows; rowBase += kMmaRows) {
      const LongType rowLow = rowBase + group;
      const LongType rowHigh = rowLow + kMmaRows / 2;
      typename Mma::FragmentC accumulators[kColumnTilesPerWarp];
      CUTLASS_PRAGMA_UNROLL
      for (auto& accumulator : accumulators) accumulator.clear();

      for (LongType chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
        const LongType k = chunk * kChunkDepth + member * kLaneDepth;
        const bool laneActive = k < depth;
        // Columns come in whole column tiles (admission), so only K masks a lane.
        const bool active[kColumnTilesPerWarp] = {laneActive};
        typename Weights::LaneWeights current[kColumnTilesPerWarp];
        if (laneActive) current[0] = weights.unpack(weights.staged(staged, chunk - stageBegin));
        multiplyChunk(x, weights, constants, current, active, laneActive, rowLow, rowHigh, rows, depth, k,
                      accumulators);
      }
      reduceRowTile<X, Z>(partials, accumulators, z, scratch, rowBase, blockColumn, rows, columns, splitBlocks,
                          splitIndex, warp, lane);
    }

    if (splitBlocks > 1)
      combineSplits(lastArrival, tickets, scratch, z, columnGroup, blockColumn, rows, columns, splitBlocks);
  }
#else
  __trap();
#endif
}

// Persistent per-device split scratch and zeroed tickets, shared by every
// activation and output type. They grow only while the stream is not capturing
// (a plan's warmup executes every shape first), so captured graphs keep stable
// pointers.
struct SplitScratch {
  void* partials = nullptr;
  LongType partialBytes = 0;
  unsigned int* tickets = nullptr;
  LongType ticketCapacity = 0;
};

static std::mutex splitScratchLock;
static std::vector<SplitScratch> splitScratchByDevice;

static void ensureSplitScratch(cudaStream_t stream, LongType partialBytes, LongType groups, void** partialsOut,
                               unsigned int** ticketsOut) {
  const int device = AffinityManager::currentDeviceId();
  std::lock_guard<std::mutex> guard(splitScratchLock);
  if (static_cast<int>(splitScratchByDevice.size()) <= device) splitScratchByDevice.resize(device + 1);
  SplitScratch& scratch = splitScratchByDevice[device];
  if (scratch.partialBytes < partialBytes || scratch.ticketCapacity < groups) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
      THROW_EXCEPTION("WeightOnlyGemm: split scratch must grow during capture; execute the shape before capture");
    // Tickets are reset by their last arrival, so they are zeroed only when allocated.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      THROW_EXCEPTION("WeightOnlyGemm: stream synchronize before split scratch growth failed");
    if (scratch.partialBytes < partialBytes) {
      if (scratch.partials != nullptr) cudaFree(scratch.partials);
      scratch.partials = nullptr;
      if (cudaMalloc(&scratch.partials, partialBytes) != cudaSuccess)
        THROW_EXCEPTION("WeightOnlyGemm: split scratch allocation failed");
      scratch.partialBytes = partialBytes;
    }
    if (scratch.ticketCapacity < groups) {
      if (scratch.tickets != nullptr) cudaFree(scratch.tickets);
      scratch.tickets = nullptr;
      if (cudaMalloc(&scratch.tickets, groups * sizeof(unsigned int)) != cudaSuccess ||
          cudaMemset(scratch.tickets, 0, groups * sizeof(unsigned int)) != cudaSuccess)
        THROW_EXCEPTION("WeightOnlyGemm: split ticket allocation failed");
      scratch.ticketCapacity = groups;
    }
  }
  *partialsOut = scratch.partials;
  *ticketsOut = scratch.tickets;
}

// Scratch rows: at least 64 and a power of two, so a plan's later, longer
// prefill rarely has to grow it.
static LongType splitScratchRows(LongType rows) {
  LongType capacity = 64;
  while (capacity < rows) capacity *= 2;
  return capacity;
}

// Blocks resident per SM at the launch bound, times the SM count: one wave.
static LongType residentBlocks() {
  int multiprocessors = 0;
  if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, AffinityManager::currentDeviceId()) !=
      cudaSuccess)
    THROW_EXCEPTION("WeightOnlyGemm: multiprocessor count query failed");
  return static_cast<LongType>(multiprocessors) * kMinBlocksPerSm;
}

// A shape splits when its column groups fill fewer than kSplitWaveLimit waves:
// the partial last wave then dominates. Depends on the shape only.
static constexpr LongType kSplitWaveLimit = 4;

// SD_WEIGHT_ONLY_SPLIT_BLOCKS (1, 2 or 4; empty = kSplitBlocks) overrides the
// split for few-wave shapes; 1 disables it.
static int configuredSplitBlocks() {
  static const int configured = [] {
    const char* value = std::getenv("SD_WEIGHT_ONLY_SPLIT_BLOCKS");
    if (value == nullptr || value[0] == '\0') return kSplitBlocks;
    const int parsed = std::atoi(value);
    if (parsed != 1 && parsed != 2 && parsed != 4)
      THROW_EXCEPTION("SD_WEIGHT_ONLY_SPLIT_BLOCKS must be 1, 2 or 4");
    return parsed;
  }();
  return configured;
}

static int splitBlocksFor(LongType columns, LongType depth) {
  const int split = configuredSplitBlocks();
  if (split == 1) return 1;
  const LongType groups = (columns + kColumnsPerBlock - 1) / kColumnsPerBlock;
  const LongType chunks = (depth + kChunkDepth - 1) / kChunkDepth;
  // Each split warp keeps at least two chunks to pipeline.
  if (chunks < 2LL * kSplitK * split) return 1;
  return groups < kSplitWaveLimit * residentBlocks() ? split : 1;
}

// The largest staged tile (dynamic shared memory) at which two bulk blocks stay
// resident per SM; 0 when the device, or the kernel image loaded for it, has no
// bulk copies. Staging pays for itself only with a second block per SM to
// overlap one block's copies with another's MMAs: on GB10, unsplit K = 12288
// and 17408 tiles (one block per SM) ran 5-7% slower than the direct kernel,
// K = 9216 (two per SM) 14% faster. Cached per device and kernel instance (each
// instance opts in to its own dynamic shared memory).
template <typename X, typename Z>
static LongType bulkTileLimit() {
  static std::mutex lock;
  static std::vector<LongType> limits;
  const int device = AffinityManager::currentDeviceId();
  std::lock_guard<std::mutex> guard(lock);
  if (static_cast<int>(limits.size()) <= device) limits.resize(device + 1, -1);
  if (limits[device] >= 0) return limits[device];
  LongType limit = 0;
  if (CutlassHelper::getSmVersion(device) >= 90) {
    const auto kernel = weightOnlyGemmBulkKernel<X, Z, ModelOptNvfp4Weights>;
    cudaFuncAttributes attributes;
    if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess)
      THROW_EXCEPTION("WeightOnlyGemm: bulk kernel attribute query failed");
    // ptxVersion is the virtual architecture of the image loaded for this
    // device: below 90 the kernel was built without bulk copies.
    if (attributes.ptxVersion >= 90) {
      int optIn = 0;
      if (cudaDeviceGetAttribute(&optIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess)
        THROW_EXCEPTION("WeightOnlyGemm: shared memory limit query failed");
      const int dynamicMax = optIn - static_cast<int>(attributes.sharedSizeBytes);
      if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicMax) != cudaSuccess)
        THROW_EXCEPTION("WeightOnlyGemm: bulk kernel shared memory opt-in failed");
      size_t twoPerSm = 0;
      if (cudaOccupancyAvailableDynamicSMemPerBlock(&twoPerSm, kernel, 2, kBlockThreads) != cudaSuccess)
        THROW_EXCEPTION("WeightOnlyGemm: bulk kernel occupancy query failed");
      limit = std::min<LongType>(static_cast<LongType>(twoPerSm), dynamicMax);
    }
  }
  limits[device] = limit;
  return limit;
}

static bool alignedTo16(NDArray* array) {
  return reinterpret_cast<uintptr_t>(array->specialBuffer()) % 16 == 0;
}

// The CUDA memory type backing an operand (diagnostics: managed and host
// memory stream more slowly than device memory).
static const char* memoryType(NDArray* array) {
  cudaPointerAttributes attributes;
  if (cudaPointerGetAttributes(&attributes, array->specialBuffer()) != cudaSuccess) {
    cudaGetLastError();
    return "unknown";
  }
  switch (attributes.type) {
    case cudaMemoryTypeDevice: return "device";
    case cudaMemoryTypeManaged: return "managed";
    case cudaMemoryTypeHost: return "host";
    default: return "unregistered";
  }
}

// Measured on GB10 (sm_121: 48 SMs, 128 KB of L1 and shared memory per SM),
// staging beats the direct kernel only when, besides the tile leaving room for
// two resident blocks (bulkTileLimit):
//  - a unit spans at least kBulkMinUnitDepth of K, so the copy each unit waits
//    for is amortized (N = 17408, one row: K = 2048 and 3072 are 1-2% slower,
//    K = 4096 4% faster, K = 5120 11% faster);
//  - the shape has at least kSplitWaveLimit waves of units, so every SM keeps
//    cycling units and one block's copy overlaps the others' MMAs (K = 5120,
//    one row: 512 units 11% slower, 640 5% slower, 768 2% faster, 2176 11%
//    faster);
//  - a unit's activation slice, rows by its K range, fits kBulkActivationBytes:
//    the tiles' shared memory is carved out of L1, which must still hold the
//    slice for the SM's other units (N = 17408, K = 5120: 6 rows (60 KB) 1%
//    faster, 7 rows (70 KB) 13% slower, 16 rows 49% slower; the direct kernel
//    alone, forced to the same carve-out, runs 33% slower at 8 rows).
// Both kernels accumulate in the same order, so the choice may depend on rows
// without changing any result.
static constexpr LongType kBulkMinUnitDepth = 4096;
static constexpr LongType kBulkActivationBytes = 64 * 1024;

template <typename X>
static bool bulkStagingPays(LongType rows, LongType unitDepth, LongType units) {
  return unitDepth >= kBulkMinUnitDepth && units >= kSplitWaveLimit * residentBlocks() &&
         rows * unitDepth * static_cast<LongType>(sizeof(X)) <= kBulkActivationBytes;
}

// SD_WEIGHT_ONLY_STAGING overrides bulkStagingPays, to A/B the kernels in one
// thermal state or re-measure the rule on another device: "direct" never
// stages, "bulk" stages every shape the device and operands admit; empty =
// bulkStagingPays. Only timing changes.
enum class Staging { Measured, Direct, Bulk };

static Staging configuredStaging() {
  static const Staging configured = [] {
    const char* value = std::getenv("SD_WEIGHT_ONLY_STAGING");
    if (value == nullptr || value[0] == '\0') return Staging::Measured;
    const std::string choice(value);
    if (choice != "direct" && choice != "bulk")
      THROW_EXCEPTION("SD_WEIGHT_ONLY_STAGING must be direct, bulk or empty");
    return choice == "direct" ? Staging::Direct : Staging::Bulk;
  }();
  return configured;
}

template <typename X, typename Z>
static void weightOnlyGemmNvfp4_(LaunchContext* context, NDArray* x, NDArray* w, NDArray* blockScales,
                                 NDArray* globalScale, NDArray* z, unsigned int blocks) {
  if constexpr (IsTensorCoreElement<X>::value) {
    using Weights = ModelOptNvfp4Weights;
    using AccT = WeightOnlyAccumulator<X>;
    const LongType depth = x->sizeAt(-1);
    const LongType rows = x->lengthOf() / depth;
    const LongType columns = w->sizeAt(0);
    const Weights weights{static_cast<const Weights::Storage*>(w->specialBuffer()),
                          static_cast<const Weights::ScaleStorage*>(blockScales->specialBuffer()),
                          static_cast<const Weights::Scale*>(globalScale->specialBuffer()), depth};
    const int splitBlocks = splitBlocksFor(columns, depth);
    void* partials = nullptr;
    unsigned int* tickets = nullptr;
    const LongType groups = (columns + kColumnsPerBlock - 1) / kColumnsPerBlock;
    if (splitBlocks > 1)
      ensureSplitScratch(*context->getCudaStream(),
                         static_cast<LongType>(splitBlocks) * splitScratchRows(rows) * columns *
                             static_cast<LongType>(sizeof(AccT)),
                         groups, &partials, &tickets);
    AccT* scratch = static_cast<AccT*>(partials);
    const LongType units = groups * splitBlocks;
    const unsigned int launched = units < blocks ? static_cast<unsigned int>(units) : blocks;

    // The largest K range a unit stages (the kernel's split of the chunks).
    const LongType chunks = (depth + kChunkDepth - 1) / kChunkDepth;
    const LongType parts = static_cast<LongType>(kSplitK) * splitBlocks;
    LongType unitChunks = 0;
    for (LongType split = 0; split < splitBlocks; ++split)
      unitChunks = std::max(unitChunks, (split + 1) * kSplitK * chunks / parts - split * kSplitK * chunks / parts);
    const Weights::Tile tile = Weights::tileFor(unitChunks);
    const Staging staging = configuredStaging();
    const bool bulk = staging != Staging::Direct && alignedTo16(blockScales) &&
                      (staging == Staging::Bulk || bulkStagingPays<X>(rows, unitChunks * kChunkDepth, units)) &&
                      tile.bytes <= bulkTileLimit<X, Z>();
    DSP_DIAG(BACKEND,
             "WeightOnlyGemm %s rows=%lld columns=%lld depth=%lld split=%d tileBytes=%lld memory w=%s scales=%s",
             bulk ? "bulk" : "direct", static_cast<long long>(rows), static_cast<long long>(columns),
             static_cast<long long>(depth), splitBlocks, static_cast<long long>(tile.bytes), memoryType(w),
             memoryType(blockScales));

    cudaStream_t stream = *context->getCudaStream();
    if (bulk)
      weightOnlyGemmBulkKernel<X, Z, Weights><<<launched, kBlockThreads, static_cast<size_t>(tile.bytes), stream>>>(
          static_cast<const X*>(x->specialBuffer()), weights, static_cast<Z*>(z->specialBuffer()), rows, columns,
          depth, splitBlocks, scratch, tickets, tile);
    else
      weightOnlyGemmKernel<X, Z, Weights><<<launched, kBlockThreads, 0, stream>>>(
          static_cast<const X*>(x->specialBuffer()), weights, static_cast<Z*>(z->specialBuffer()), rows, columns,
          depth, splitBlocks, scratch, tickets);
  } else {
    THROW_EXCEPTION("WeightOnlyGemm: the activation type is not a tensor-core element");
  }
}

template <typename X>
static bool isTensorCoreActivation() {
  return IsTensorCoreElement<X>::value;
}

#endif  // SD_WEIGHT_ONLY_GEMM_AVAILABLE

bool WeightOnlyGemm::isAdmitted(WeightOnlyFormat format, NDArray* x, NDArray* w, NDArray* blockScales,
                                NDArray* z) {
#if SD_WEIGHT_ONLY_GEMM_AVAILABLE
  using Weights = ModelOptNvfp4Weights;
  if (format != WeightOnlyFormat::MODELOPT_NVFP4) return false;
  if (CutlassHelper::getSmVersion(AffinityManager::currentDeviceId()) < 80) return false;
  bool tensorCore = false;
  BUILD_SINGLE_SELECTOR(x->dataType(), tensorCore = isTensorCoreActivation, (), SD_FLOAT_TYPES);
  if (!tensorCore) return false;
  if (w->dataType() != DataTypeUtils::fromT<Weights::Storage>() ||
      blockScales->dataType() != DataTypeUtils::fromT<Weights::ScaleStorage>())
    return false;
  if (x->isEmpty() || w->isEmpty() || blockScales->isEmpty() || z->isEmpty()) return false;
  const LongType depth = x->sizeAt(-1);
  const LongType columns = w->sizeAt(0);
  // A lane owns kLaneDepth K elements (one 16-byte load of codes spanning
  // whole scale blocks); output columns come in whole MMA tiles.
  if (depth % kLaneDepth != 0 || columns % kMmaColumns != 0) return false;
  if (!shape::isDenseRowMajor(x->shapeInfo()) || !shape::isDenseRowMajor(w->shapeInfo()) ||
      !shape::isDenseRowMajor(blockScales->shapeInfo()) || !shape::isDenseRowMajor(z->shapeInfo()))
    return false;
  // A lane's scales are read as one LaneScales.
  return alignedTo16(x) && alignedTo16(w) && alignedTo16(z) &&
         reinterpret_cast<uintptr_t>(blockScales->specialBuffer()) % alignof(Weights::LaneScales) == 0;
#else
  return false;
#endif
}

void WeightOnlyGemm::run(LaunchContext* context, WeightOnlyFormat format, NDArray* x, NDArray* w,
                         NDArray* blockScales, NDArray* globalScale, NDArray* z) {
#if SD_WEIGHT_ONLY_GEMM_AVAILABLE
  if (!isAdmitted(format, x, w, blockScales, z))
    THROW_EXCEPTION("WeightOnlyGemm::run: operands are not admitted to the tensor-core path");
  // The block size is fixed by the algorithm (kSplitK warps); the grid is a
  // cap, and blocks grid-stride over column groups.
  const dim3 dims = getLaunchDims("weight_only_gemm");
  if (dims.y != static_cast<unsigned int>(kBlockThreads))
    THROW_EXCEPTION(("WeightOnlyGemm: BLOCK_SIZE_WEIGHT_ONLY_GEMM must be " + std::to_string(kBlockThreads)).c_str());
  if (dims.x == 0) THROW_EXCEPTION("WeightOnlyGemm: GRID_SIZE_WEIGHT_ONLY_GEMM must be positive");
  // dims.x caps the grid; the launcher sizes it to the shape's work units.
  const unsigned int blocks = dims.x;
  BUILD_DOUBLE_SELECTOR(x->dataType(), z->dataType(), weightOnlyGemmNvfp4_,
                        (context, x, w, blockScales, globalScale, z, blocks), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
#else
  THROW_EXCEPTION("WeightOnlyGemm::run: built without CUTLASS or without modelopt_nvfp4_linear");
#endif
}

}  // namespace sd
