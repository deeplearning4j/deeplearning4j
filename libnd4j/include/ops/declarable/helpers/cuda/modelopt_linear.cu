/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#include <cuda_runtime.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/helpers/modelopt_linear.h>
#include <execution/cuda/LaunchDims.h>
#include <graph/DspDiagnostics.h>
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/WeightOnlyGemm.h>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace sd {
namespace ops {
namespace helpers {

// Runtime tensors cannot be synchronously value-validated during capture. Keep
// the ordered validation kernel for the empty-output corner, where no compute
// kernel runs to carry the fused check above.
template <typename Format>
SD_KERNEL static void modelOptValidateScalesKernel(const typename Format::ScaleStorage* scale,
                                                   const typename Format::Scale* second,
                                                   const LongType* shapeInfo, LongType count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) modelOptScaleChecked(second[0]);
  if (count == 0) return;
  const int rank = shape::rank(shapeInfo);
  const LongType* shapeOf = shape::shapeOf(shapeInfo);
  const LongType* strides = shape::stride(shapeInfo);
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < count; linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linear, rank, shapeOf, coords);
    LongType offset = 0;
    COORDS2INDEX(rank, strides, coords, offset);
    modelOptScaleChecked(static_cast<typename Format::Scale>(scale[offset]));
  }
}

// ─── General path: view-safe thread-per-output dot product ──────────────────
// Handles every stride/view/order combination through INDEX2COORDS/COORDS2INDEX
// with each operand's own strides, AccT accumulation, and fused scale
// validation. Correctness fallback for inputs that fail the fast-path proof.
template <typename X, typename Z, typename Format>
SD_KERNEL static void modelOptLinearKernel(const X* x, const typename Format::Storage* w,
                                          const typename Format::ScaleStorage* scale,
                                          const typename Format::Scale* secondScale, Z* z,
                                          const LongType* xShape, const LongType* wShape,
                                          const LongType* sShape, const LongType* zShape,
                                          LongType length) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const int rank = shape::rank(xShape);
  const LongType kLength = shape::sizeAt(xShape, rank - 1);
  const auto* xs = shape::stride(xShape);
  const auto* ws = shape::stride(wShape);
  const auto* ss = shape::stride(sShape);
  const auto* zs = shape::stride(zShape);
  const auto* zd = shape::shapeOf(zShape);
  // Fused on-stream validation of the global/input scale scalar (previously
  // carried by the standalone pre-flight kernel; see modelOptScaleChecked).
  const typename Format::Scale second = modelOptScaleChecked(secondScale[0]);
  const auto weightScale = Format::weightScale(scale, second);
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < length; linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linear, rank, zd, coords);
    const LongType n = coords[rank - 1];
    LongType xo = 0, zo = 0;
    COORDS2INDEX(rank, zs, coords, zo);
    coords[rank - 1] = 0;
    COORDS2INDEX(rank, xs, coords, xo);
    AccT sum = static_cast<AccT>(0);
    for (LongType k = 0; k < kLength; ++k) {
      const AccT a = Format::template activation<AccT>(static_cast<AccT>(x[xo + k * xs[rank - 1]]), second);
      const AccT b = Format::template weight<X, AccT>(w, ws, scale, ss, weightScale, n, k);
      sum = math::sd_fma<AccT>(a, b, sum);
    }
    z[zo] = static_cast<Z>(sum);
  }
}

// ─── Contiguous fast path: block-tiled dequant GEMV ─────────────────────────
//
// Admitted only after modelOptTiledEligible proves every operand is dense
// row-major and the packed weights are word aligned (no ews() shortcut).
//
// Work decomposition. A block owns kTiledColumnsPerWarp output columns per
// warp and walks K in tiles of kTiledTileWords<AccT> words. Each tile is
// staged into shared memory ONCE per block, already in accumulator form (the
// input converted to AccT; for FP8 also quantized through the ModelOpt
// activation quantizer), and every warp of the block consumes it for all of
// its columns. Activation loads and FP8 activation quantization therefore run
// once per block rather than once per output column, and each weight word is
// loaded and dequantized once per row pass and applied to all rows of that
// pass (up to kTiledRowsPerPass register accumulators per column).
//
// Numerical contract (pinned bit for bit by
// TestModelOptLinear#testTiledRowPassesMatchLaneContract). For every output
// (row, n), lane l accumulates the 8-element words l, l + 32, l + 64, ... in
// ascending order with one fused multiply-add per element, then the 32 lane
// partials are folded in ascending-offset order. A tile holds a whole number
// of 32-word strides, so tiling does not change any lane's word order.
//
// Shared-memory layout: [row][element-in-word][word]. Lane l reads word l of
// each element slot, so a warp's 32 reads hit 32 consecutive accumulators (no
// bank conflicts); the transposition happens once, while staging. The tile is
// a fixed byte budget of static shared memory, so its word count follows from
// sizeof(AccT).
//
// All loops that contain a barrier are block-uniform and no thread returns
// early; out-of-range columns are skipped per warp (the guard derives from
// the warp index, so it is warp-uniform and never splits a shuffle).
static constexpr int kElementsPerWord = 8;  // a lane's unit of K, fixed by the lane contract above
static constexpr int kTiledRowsPerPass = 8;
static constexpr int kTiledColumnsPerWarp = 2;
static constexpr int kTiledTileBytes = 32 * 1024;  // the staged activation tile's static shared memory
template <typename AccT>
static constexpr int kTiledTileWords =
    kTiledTileBytes / (kTiledRowsPerPass * kElementsPerWord * static_cast<int>(sizeof(AccT)));

// The dequantized weights (n, word * kElementsPerWord + e), e < kElementsPerWord,
// of the dense row-major operands the tiled proof admits. The format tag picks
// the word decoder.
//
// NVFP4: a word is one aligned 32-bit load of E2M1 codes inside one scale block.
template <typename X, typename AccT>
SD_DEVICE SD_INLINE static void tiledWord(ModelOptNvfp4, const ModelOptNvfp4::Storage* w,
                                          const ModelOptNvfp4::ScaleStorage* scale,
                                          ModelOptNvfp4::Scale globalScale, LongType n, LongType word,
                                          LongType kLength, AccT (&weights)[kElementsPerWord]) {
  using Format = ModelOptNvfp4;
  static_assert(kElementsPerWord == Format::Weight::codesPer<uint32_t>(), "a word is one 32-bit load of codes");
  static_assert(Format::kBlockLength % kElementsPerWord == 0, "a word lies inside one scale block");
  const Format::Scale blockScale = modelOptScaleChecked(static_cast<Format::Scale>(
      scale[n * (kLength / Format::kBlockLength) + word * kElementsPerWord / Format::kBlockLength]));
  const uint32_t packed = reinterpret_cast<const uint32_t*>(w + n * (kLength / Format::kWeightsPerStorage))[word];
  for (int e = 0; e < kElementsPerWord; ++e)
    weights[e] = static_cast<AccT>(
        modelOptNvfp4Weight<X, Format::Scale>(Format::Weight::unpack(packed, e), blockScale, globalScale));
}

// FP8: a word is kElementsPerWord E4M3 weights, decoded in pairs.
template <typename X, typename AccT>
SD_DEVICE SD_INLINE static void tiledWord(ModelOptFp8, const ModelOptFp8::Storage* w, const ModelOptFp8::ScaleStorage*,
                                          ModelOptFp8::Scale weightScale, LongType n, LongType word, LongType kLength,
                                          AccT (&weights)[kElementsPerWord]) {
  using Format = ModelOptFp8;
  constexpr int kPair = 2;
  static_assert(kElementsPerWord % kPair == 0, "a word is whole weight pairs");
  const Format::Storage* wordWeights = w + n * kLength + word * kElementsPerWord;
  for (int e = 0; e < kElementsPerWord; e += kPair) {
    AccT pair[kPair];
    math::sd_convert_n<Format::Weight, AccT, kPair>(wordWeights + e, pair);
    for (int p = 0; p < kPair; ++p)
      weights[e + p] = reproducible::multiply<AccT>(pair[p], static_cast<AccT>(weightScale));
  }
}

template <typename X, typename Z, typename Format>
SD_KERNEL static void modelOptLinearTiledKernel(
    const X* __restrict__ x, const typename Format::Storage* __restrict__ w,
    const typename Format::ScaleStorage* __restrict__ scale,
    const typename Format::Scale* __restrict__ secondScale, Z* __restrict__ z,
    LongType rows, LongType nColumns, LongType kLength) {
  using AccT = typename simdOps::AggregateType<X>::type;
  using sd::device::WARP_SIZE;
  constexpr int kTileWords = kTiledTileWords<AccT>;
  constexpr int kTileLength = kTileWords * kElementsPerWord;
  static_assert(kTileWords > 0 && kTileWords % WARP_SIZE == 0, "a tile must hold whole lane strides");
  __shared__ AccT activations[kTiledRowsPerPass][kElementsPerWord * kTileWords];

  const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const LongType columnsPerBlock = static_cast<LongType>(blockDim.x / WARP_SIZE) * kTiledColumnsPerWarp;
  const typename Format::Scale second = secondScale[0];
  if (blockIdx.x == 0 && threadIdx.x == 0) modelOptScaleChecked(second);
  const auto weightScale = Format::weightScale(scale, second);

  for (LongType blockColumn = static_cast<LongType>(blockIdx.x) * columnsPerBlock; blockColumn < nColumns;
       blockColumn += static_cast<LongType>(gridDim.x) * columnsPerBlock) {
    const LongType warpColumn = blockColumn + static_cast<LongType>(warp) * kTiledColumnsPerWarp;

    for (LongType rowBase = 0; rowBase < rows; rowBase += kTiledRowsPerPass) {
      const int passRows = static_cast<int>(sd::math::sd_min<LongType>(rows - rowBase, kTiledRowsPerPass));
      AccT sum[kTiledColumnsPerWarp][kTiledRowsPerPass];
      for (int c = 0; c < kTiledColumnsPerWarp; ++c)
        for (int r = 0; r < kTiledRowsPerPass; ++r) sum[c][r] = static_cast<AccT>(0);

      for (LongType tileStart = 0; tileStart < kLength; tileStart += kTileLength) {
        const int tileWords =
            static_cast<int>(sd::math::sd_min<LongType>(kLength - tileStart, kTileLength)) / kElementsPerWord;

        __syncthreads();  // every warp is done reading the previous tile
        for (int i = static_cast<int>(threadIdx.x); i < passRows * kElementsPerWord * tileWords;
             i += static_cast<int>(blockDim.x)) {
          const int tileWord = i % tileWords;
          const int element = (i / tileWords) % kElementsPerWord;
          const int r = i / (tileWords * kElementsPerWord);
          const AccT value = static_cast<AccT>(
              x[(rowBase + r) * kLength + tileStart + tileWord * kElementsPerWord + element]);
          activations[r][element * kTileWords + tileWord] = Format::template activation<AccT>(value, second);
        }
        __syncthreads();

        const LongType tileWordStart = tileStart / kElementsPerWord;
        for (int tileWord = lane; tileWord < tileWords; tileWord += WARP_SIZE) {
          const LongType word = tileWordStart + tileWord;
          for (int c = 0; c < kTiledColumnsPerWarp; ++c) {
            const LongType n = warpColumn + c;
            if (n >= nColumns) break;  // warp-uniform

            AccT weights[kElementsPerWord];
            tiledWord<X, AccT>(Format{}, w, scale, weightScale, n, word, kLength, weights);

            for (int r = 0; r < kTiledRowsPerPass; ++r) {
              if (r >= passRows) break;  // warp-uniform
              for (int e = 0; e < kElementsPerWord; ++e)
                sum[c][r] = sd::math::sd_fma<AccT>(activations[r][e * kTileWords + tileWord], weights[e],
                                                   sum[c][r]);
            }
          }
        }
      }

      // Ordered lane fold. warpReduceSum folds in butterfly (offset-halving)
      // order, which reassociates the partials and was observed to flip
      // one-ULP BF16 storage boundaries against the general path's left fold.
      // Ascending offsets make lane 0 absorb its neighbours first and larger
      // partials enter last, matching the general path's fold shape.
      for (int c = 0; c < kTiledColumnsPerWarp; ++c) {
        const LongType n = warpColumn + c;
        if (n >= nColumns) break;  // warp-uniform
        for (int r = 0; r < kTiledRowsPerPass; ++r) {
          if (r >= passRows) break;  // warp-uniform
          AccT total = sum[c][r];
          for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            const AccT neighbor = __shfl_down_sync(0xffffffff, total, offset);
            if (lane + offset < WARP_SIZE) total += neighbor;
          }
          if (lane == 0) z[(rowBase + r) * nColumns + n] = static_cast<Z>(total);
        }
      }
    }
  }
}

// The content stamp's rule (ContentPredicate::MODELOPT_SCALES_VALID) is "every
// E4M3 block scale is positive and finite". A stamp does not record the element
// type it was read as, so only scale tensors stored as E4M3 are stamped; other
// scale tensors (the FP8 format's one FP32 weight scale) are scanned each call.
template <typename Format>
static constexpr bool kModelOptScalesStamped = std::is_same<typename Format::ScaleStorage, float8_e4m3>::value;

// Host-current scales are validated here, before any launch, so invalid input
// fails with an exception instead of a device trap; the compute kernels also
// validate every scale they read on the stream.
//  - The format's scalar scale (NVFP4 global scale, FP8 input scale) is
//    checked whenever its host copy is current: one comparison per call.
//  - The format's scale tensor is scanned when its host copy is current. NVFP4
//    block-scale tensors (megabytes for a large model) record the result as a
//    content stamp on their DataBuffer, so a model's constant scales are
//    scanned once. The stamp dies with the buffer and with any write to it; the
//    previous cache, keyed on host addresses, vouched for new buffers the
//    allocator placed at a validated buffer's freed address.
template <typename Format>
static void validateHostCurrentScales(NDArray* scale, NDArray* secondScale) {
  if (secondScale->isActualOnHostSide()) modelOptCheckSecondScale<Format>(secondScale);
  if (scale->isEmpty() || !scale->isActualOnHostSide()) return;
  // A stamp describes the whole buffer, so it is only read or written when the
  // array covers every byte of it.
  DataBuffer* buffer = scale->dataBuffer();
  const bool coversBuffer = kModelOptScalesStamped<Format> && buffer != nullptr && scale->offset() == 0 &&
                            shape::isDenseRowMajor(scale->shapeInfo()) &&
                            scale->lengthOf() * static_cast<LongType>(scale->sizeOfT()) == buffer->getLenInBytes();
  if (coversBuffer && buffer->isContentValidated(ContentPredicate::MODELOPT_SCALES_VALID)) return;
  modelOptCheckScaleTensor<Format>(scale);
  if (coversBuffer) buffer->stampContentValidated(ContentPredicate::MODELOPT_SCALES_VALID);
}

// Fast-path proof. Every precondition is explicit; no ews() shortcut. Each
// operand must be dense C-order with exactly the row-major strides its shape
// implies, and the packed weight rows must be whole 32-bit words so the
// lane-strided uint32_t loads stay aligned. Dense-stride offset views pass the
// stride proof but must also carry a word-aligned shifted base; anything else
// falls back to the general view-safe kernel.
static bool modelOptTiledEligible(NDArray* x, NDArray* w, NDArray* scale, NDArray* z) {
  if (x->isEmpty() || w->isEmpty() || scale->isEmpty() || z->isEmpty()) return false;
  const LongType k = x->sizeAt(-1);
  // K must be whole lane words (this also makes every NVFP4 weight row whole
  // 32-bit words, so the lane-strided uint32_t loads stay aligned).
  if (k % kElementsPerWord != 0) return false;
  // X must be a dense [rows, K] row-major block (rank-1 is a single K row);
  // W dense [N, K/2] (NVFP4) or [N, K] (FP8); NVFP4 block scales dense
  // [N, K/16] (the FP8 scale is a rank-0 scalar, trivially dense); Z dense
  // [rows, N] over the flattened leading dims. The kernel addresses every
  // operand as exactly that packed layout.
  if (!shape::isDenseRowMajor(x->shapeInfo()) || !shape::isDenseRowMajor(w->shapeInfo()) ||
      !shape::isDenseRowMajor(scale->shapeInfo()) || !shape::isDenseRowMajor(z->shapeInfo()))
    return false;
  // Word alignment: the shifted view base and every row start must be
  // aligned for the reinterpret_cast<uint32_t*> loads.
  const uintptr_t wBase = reinterpret_cast<uintptr_t>(w->specialBuffer());
  return wBase % alignof(uint32_t) == 0;
}

template <typename X, typename Z, typename Format>
static void modelOptLinear_(Format, LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                            NDArray* secondScale, NDArray* z, dim3 dims) {
  auto* stream = context->getCudaStream();
  const LongType needed = (z->lengthOf() - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  modelOptLinearKernel<X, Z, Format><<<blocks, dims.y, dims.z, *stream>>>(
      x->isEmpty() ? nullptr : static_cast<const X*>(x->specialBuffer()),
      w->isEmpty() ? nullptr : static_cast<const typename Format::Storage*>(w->specialBuffer()),
      scale->isEmpty() ? nullptr : static_cast<const typename Format::ScaleStorage*>(scale->specialBuffer()),
      static_cast<const typename Format::Scale*>(secondScale->specialBuffer()), static_cast<Z*>(z->specialBuffer()),
      x->specialShapeInfo(), w->specialShapeInfo(), scale->specialShapeInfo(), z->specialShapeInfo(),
      z->lengthOf());
}

template <typename X, typename Z, typename Format>
static void modelOptLinearTiled_(Format, LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                 NDArray* secondScale, NDArray* z, dim3 dims) {
  auto* stream = context->getCudaStream();
  const LongType rank = x->rankOf();
  const LongType kLength = x->sizeAt(-1);
  const LongType rows = rank == 1 ? 1 : x->lengthOf() / kLength;
  const LongType nColumns = w->sizeAt(0);
  const LongType columnsPerBlock =
      static_cast<LongType>(dims.y / sd::device::WARP_SIZE) * kTiledColumnsPerWarp;
  const LongType needed = (nColumns + columnsPerBlock - 1) / columnsPerBlock;
  unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  if (blocks == 0) blocks = 1;
  modelOptLinearTiledKernel<X, Z, Format><<<blocks, dims.y, dims.z, *stream>>>(
      static_cast<const X*>(x->specialBuffer()), static_cast<const typename Format::Storage*>(w->specialBuffer()),
      static_cast<const typename Format::ScaleStorage*>(scale->specialBuffer()),
      static_cast<const typename Format::Scale*>(secondScale->specialBuffer()), static_cast<Z*>(z->specialBuffer()),
      rows, nColumns, kLength);
}

// Launch-limit queries are cached per device; cudaGetDeviceProperties costs
// tens of microseconds and the decode path calls this op hundreds of times per
// token. Device properties are immutable for the process lifetime, so the
// first successful query wins.
static const cudaDeviceProp& modelOptDeviceProps(LaunchContext* context) {
  static std::mutex mutex;
  static std::unordered_map<int, cudaDeviceProp> cache;
  const int deviceId = context->getDeviceID();
  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(deviceId);
  if (found != cache.end()) return found->second;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    throw std::runtime_error("ModelOpt linear: unable to query launch limits");
  return cache.emplace(deviceId, prop).first->second;
}

// Named launch dimensions are environment-overridable; reject anything the
// device cannot launch before it reaches a kernel.
static dim3 modelOptLaunchDims(const char* name, const cudaDeviceProp& prop) {
  const dim3 dims = getLaunchDims(name);
  if (dims.x == 0 || dims.y == 0 || dims.x > static_cast<unsigned int>(prop.maxGridSize[0]) ||
      dims.y > static_cast<unsigned int>(prop.maxThreadsPerBlock) ||
      dims.y > static_cast<unsigned int>(prop.maxThreadsDim[0]) || dims.z > prop.sharedMemPerBlock)
    throw std::invalid_argument(std::string("ModelOpt linear: invalid launch dimensions for ") + name);
  return dims;
}

// ─── Scaled GEMM on tensor cores: quantize once, then one cuBLASLt GEMM ─────
//
// Produces the dense row-major activation operand q = modelOptQuantize(x,
// inputScale) — the quantizer the native kernels apply per element — from an
// X of any layout. It also carries the on-stream validation of both scalar
// scales, which the library GEMM cannot perform.
template <typename X, typename Format>
SD_KERNEL static void modelOptQuantizeKernel(const X* x, const LongType* xShapeInfo,
                                             typename Format::Activation* quantized, LongType length,
                                             const typename Format::Scale* inputScale,
                                             const typename Format::ScaleStorage* weightScale) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const AccT scale = static_cast<AccT>(inputScale[0]);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    modelOptScaleChecked(inputScale[0]);
    Format::weightScale(weightScale, inputScale[0]);  // the per-tensor weight factor, checked
  }
  const int rank = shape::rank(xShapeInfo);
  const LongType* xShape = shape::shapeOf(xShapeInfo);
  const LongType* xStrides = shape::stride(xShapeInfo);
  for (LongType linearIndex = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; linearIndex < length;
       linearIndex += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linearIndex, rank, xShape, coords);
    LongType xOffset = 0;
    COORDS2INDEX(rank, xStrides, coords, xOffset);
    quantized[linearIndex] = modelOptQuantize<typename Format::Activation, AccT>(static_cast<AccT>(x[xOffset]), scale);
  }
}

template <typename X, typename Format>
static void modelOptQuantize_(Format, LaunchContext* context, NDArray* x, typename Format::Activation* quantized,
                              NDArray* scale, NDArray* secondScale, dim3 dims) {
  const LongType length = x->lengthOf();
  const LongType needed = (length - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  modelOptQuantizeKernel<X, Format><<<blocks, dims.y, dims.z, *context->getCudaStream()>>>(
      static_cast<const X*>(x->specialBuffer()), x->specialShapeInfo(), quantized, length,
      static_cast<const typename Format::Scale*>(secondScale->specialBuffer()),
      static_cast<const typename Format::ScaleStorage*>(scale->specialBuffer()));
}

// cuBLASLt scaled GEMMs need 16-byte aligned operand bases and leading
// dimensions. W and Z are addressed as dense row-major matrices; X needs no
// layout proof because the quantizer writes a dense copy.
static constexpr LongType kModelOptLtAlignment = 16;

// The scaled GEMM accumulates in FP32 (CUBLAS_COMPUTE_32F, see
// MmulHelper::ltMatmulScaled), so it takes only activations whose aggregate
// type is that accumulator; a wider aggregate keeps the native kernels.
using ModelOptLtAccumulator = float;

template <typename X>
static bool modelOptLtAccumulates() {
  return std::is_same<typename simdOps::AggregateType<X>::type, ModelOptLtAccumulator>::value;
}

template <typename Format>
static bool modelOptLtEligible(NDArray* x, NDArray* w, NDArray* z) {
  bool accumulates = false;
  BUILD_SINGLE_SELECTOR(x->dataType(), accumulates = modelOptLtAccumulates, (), SD_FLOAT_TYPES);
  if (!accumulates) return false;
  const LongType depth = x->sizeAt(-1);
  const LongType columns = w->sizeAt(0);
  const LongType depthBytes = depth * static_cast<LongType>(sizeof(typename Format::Activation));
  if (depth <= 0 || depthBytes % kModelOptLtAlignment != 0) return false;
  if ((columns * static_cast<LongType>(z->sizeOfT())) % kModelOptLtAlignment != 0) return false;
  if (!shape::isDenseRowMajor(w->shapeInfo()) || !shape::isDenseRowMajor(z->shapeInfo())) return false;
  return reinterpret_cast<uintptr_t>(w->specialBuffer()) % kModelOptLtAlignment == 0 &&
         reinterpret_cast<uintptr_t>(z->specialBuffer()) % kModelOptLtAlignment == 0;
}

// Z = (inputScale * weightScale) * Q(X) . W^T. Each Q(x) * w product is exact
// in FP32 and the scale product multiplies each output's sum once; the
// library owns the FP32 accumulation order. The native kernels multiply the
// dequantized operands (Q(x) * inputScale) * (w * weightScale), so the two
// paths agree to an FP32 accumulation-error bound, not bit for bit. Returns
// false when cuBLASLt has no algorithm for the problem.
template <typename Format>
static bool modelOptLinearLt(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                             NDArray* secondScale, NDArray* z, const cudaDeviceProp& prop) {
  using Operand = typename Format::Activation;
  static_assert(std::is_same<Operand, typename Format::Weight>::value, "cuBLASLt multiplies operands of one type");
  static_assert(std::is_same<typename Format::Scale, float>::value &&
                    std::is_same<typename Format::ScaleStorage, float>::value,
                "cuBLASLt takes FP32 scale pointers");
  const LongType depth = x->sizeAt(-1);
  const LongType rows = x->lengthOf() / depth;
  PointersManager manager(context, "modelOptLinearLt");
  auto* quantized = static_cast<Operand*>(manager.allocateDevMem(rows * depth * sizeof(Operand)));
  const dim3 dims = modelOptLaunchDims("modelopt_fp8_quantize", prop);
  BUILD_SINGLE_SELECTOR(x->dataType(), modelOptQuantize_, (Format{}, context, x, quantized, scale, secondScale, dims),
                        SD_FLOAT_TYPES);
  return MmulHelper::ltMatmulScaled(context, quantized, w->specialBuffer(), z->specialBuffer(), rows, w->sizeAt(0),
                                    depth, DataTypeUtils::fromT<Operand>(), z->dataType(),
                                    static_cast<const typename Format::Scale*>(secondScale->specialBuffer()),
                                    static_cast<const typename Format::ScaleStorage*>(scale->specialBuffer()));
}

// The tensor-core routes each format tries ahead of the native kernels.
template <typename Format>
struct ModelOptTensorCores;

template <>
struct ModelOptTensorCores<ModelOptNvfp4> {
  static constexpr WeightOnlyFormat kWeightOnly = WeightOnlyFormat::MODELOPT_NVFP4;
  static constexpr bool kScaledGemm = false;  // E2M1 weights dequantize inside the MMA kernel
};

template <>
struct ModelOptTensorCores<ModelOptFp8> {
  static constexpr WeightOnlyFormat kWeightOnly = WeightOnlyFormat::MODELOPT_FP8;
  static constexpr bool kScaledGemm = true;  // quantize once, then one cuBLASLt scaled GEMM
};

template <typename Format>
void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale,
                    NDArray* z) {
  using Route = ModelOptTensorCores<Format>;
  auto* stream = context->getCudaStream();
  const bool capturing = DebugHelper::inGraphCapture(stream);
  if (!capturing) validateHostCurrentScales<Format>(scale, secondScale);
  const cudaDeviceProp& prop = modelOptDeviceProps(context);
  const bool tiled = modelOptTiledEligible(x, w, scale, z);
  const dim3 dims = modelOptLaunchDims(tiled ? "modelopt_linear_tiled" : "modelopt_linear", prop);
  if (tiled && dims.y % sd::device::WARP_SIZE != 0)
    throw std::invalid_argument("ModelOpt linear: tiled block size must be a warp multiple");

  if (z->isEmpty()) {
    // At least one thread must validate the scalar even if the scale tensor is
    // empty. This is validation work, not an empty-output compute launch.
    NDArray::prepareSpecialUse({}, {scale, secondScale});
    const LongType count = scale->lengthOf();
    const LongType needed = count == 0 ? 1 : (count - 1) / dims.y + 1;
    const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
    modelOptValidateScalesKernel<Format><<<blocks, dims.y, 0, *stream>>>(
        scale->isEmpty() ? nullptr : static_cast<const typename Format::ScaleStorage*>(scale->specialBuffer()),
        static_cast<const typename Format::Scale*>(secondScale->specialBuffer()), scale->specialShapeInfo(), count);
    NDArray::registerSpecialUse({}, {scale, secondScale});
    if (!capturing) DebugHelper::checkGlobalErrorCode("ModelOpt linear launch failed");
    return;
  }

  NDArray::prepareSpecialUse({z}, {x, w, scale, secondScale});
  const LongType depth = x->sizeAt(-1);
  const LongType rows = depth > 0 ? x->lengthOf() / depth : 0;
  const bool weightOnly = WeightOnlyGemm::isAdmitted(Route::kWeightOnly, x, w, scale, z);
  bool scaledGemm = false;
  if constexpr (Route::kScaledGemm) {
    if (!weightOnly) {
      const bool eligible = modelOptLtEligible<Format>(x, w, z);
      scaledGemm = eligible && modelOptLinearLt<Format>(context, x, w, scale, secondScale, z, prop);
      if (!scaledGemm) {
        const LongType depthBytes = depth * static_cast<LongType>(sizeof(typename Format::Activation));
        DSP_DIAG(FALLBACK,
                 "ModelOpt %s linear not on cuBLASLt: rows=%lld columns=%lld depth=%lld eligible=%d "
                 "(x=%s depthBytes%%16=%lld zRowBytes%%16=%lld wDense=%d zDense=%d wAlign16=%d zAlign16=%d)",
                 Format::kName, static_cast<long long>(rows), static_cast<long long>(w->sizeAt(0)),
                 static_cast<long long>(depth), eligible ? 1 : 0, DataTypeUtils::asString(x->dataType()).c_str(),
                 static_cast<long long>(depthBytes % kModelOptLtAlignment),
                 static_cast<long long>((w->sizeAt(0) * static_cast<LongType>(z->sizeOfT())) % kModelOptLtAlignment),
                 shape::isDenseRowMajor(w->shapeInfo()) ? 1 : 0, shape::isDenseRowMajor(z->shapeInfo()) ? 1 : 0,
                 reinterpret_cast<uintptr_t>(w->specialBuffer()) % kModelOptLtAlignment == 0 ? 1 : 0,
                 reinterpret_cast<uintptr_t>(z->specialBuffer()) % kModelOptLtAlignment == 0 ? 1 : 0);
      }
    }
  }
  DSP_DIAG(BACKEND, "ModelOpt %s linear path=%s rows=%lld columns=%lld depth=%lld", Format::kName,
           weightOnly ? "weight-only-mma" : scaledGemm ? "cublaslt" : tiled ? "tiled" : "general",
           static_cast<long long>(rows), static_cast<long long>(w->sizeAt(0)), static_cast<long long>(depth));
  if (weightOnly) {
    WeightOnlyGemm::run(context, Route::kWeightOnly, x, w, scale, secondScale, z);
  } else if (!scaledGemm) {
    if (tiled) {
      BUILD_DOUBLE_SELECTOR(x->dataType(), z->dataType(), modelOptLinearTiled_,
                            (Format{}, context, x, w, scale, secondScale, z, dims), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    } else {
      BUILD_DOUBLE_SELECTOR(x->dataType(), z->dataType(), modelOptLinear_,
                            (Format{}, context, x, w, scale, secondScale, z, dims), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    }
  }
  NDArray::registerSpecialUse({z}, {x, w, scale, secondScale});
  if (!capturing) DebugHelper::checkGlobalErrorCode("ModelOpt linear launch failed");
}

template void modelOptLinear<ModelOptNvfp4>(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                            NDArray* secondScale, NDArray* z);
template void modelOptLinear<ModelOptFp8>(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                          NDArray* secondScale, NDArray* z);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
