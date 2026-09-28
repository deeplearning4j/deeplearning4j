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
#include <unordered_map>

namespace sd {
namespace ops {
namespace helpers {

// Runtime tensors cannot be synchronously value-validated during capture. Keep
// the ordered validation kernel for the empty-output corner, where no compute
// kernel runs to carry the fused check above.
SD_KERNEL static void modelOptValidateScalesKernel(const void* scale, const float* second,
                                                   const LongType* shapeInfo,
                                                   LongType count, bool nvfp4) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && !modelOptValidScale(second[0]))
    asm("trap;");
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < count; linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    float value;
    if (nvfp4) {
      const LongType cols = shape::sizeAt(shapeInfo, 1);
      const auto* strides = shape::stride(shapeInfo);
      value = static_cast<float>(static_cast<const float8*>(scale)[
          (linear / cols) * strides[0] + (linear % cols) * strides[1]]);
    } else {
      value = static_cast<const float*>(scale)[0];
    }
    if (!modelOptValidScale(value))
      asm("trap;");
  }
}

// ─── General path: view-safe thread-per-output dot product ──────────────────
// Handles every stride/view/order combination through INDEX2COORDS/COORDS2INDEX
// with each operand's own strides, FP32 AccT accumulation, and fused scale
// validation. Correctness fallback for inputs that fail the fast-path proof.
template <typename X>
SD_KERNEL static void modelOptLinearKernel(const X* x, const void* w, const void* scale,
                                          const float* secondScale, void* z,
                                          const LongType* xShape, const LongType* wShape,
                                          const LongType* sShape, const LongType* zShape,
                                          LongType length, bool nvfp4, bool floatOutput) {
  const int rank = shape::rank(xShape);
  const LongType kLength = shape::sizeAt(xShape, rank - 1);
  const auto* xs = shape::stride(xShape);
  const auto* ws = shape::stride(wShape);
  const auto* ss = shape::stride(sShape);
  const auto* zs = shape::stride(zShape);
  const auto* zd = shape::shapeOf(zShape);
  const float second = secondScale[0];
  // Fused on-stream validation of the global/input scale scalar (previously
  // carried by the standalone pre-flight kernel; see modelOptScaleChecked).
  if (!modelOptValidScale(second)) asm("trap;");
  const float weightScale = nvfp4 ? 1.0f : modelOptScaleChecked(
      static_cast<const float*>(scale)[0]);
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < length; linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linear, rank, zd, coords);
    const LongType n = coords[rank - 1];
    LongType xo = 0, zo = 0;
    COORDS2INDEX(rank, zs, coords, zo);
    coords[rank - 1] = 0;
    COORDS2INDEX(rank, xs, coords, xo);
    using AccT = float;
    AccT sum = 0.0f;
    for (LongType k = 0; k < kLength; ++k) {
      float a = static_cast<float>(x[xo + k * xs[rank - 1]]);
      float b;
      if (nvfp4) {
        const auto byte = static_cast<const uint8_t*>(w)[n * ws[0] + (k / 2) * ws[1]];
        const auto nibble = static_cast<uint8_t>((byte >> ((k & 1) * 4)) & 15);
        b = modelOptNvfp4Weight<X>(nibble,
            modelOptScaleChecked(static_cast<float>(static_cast<const float8*>(scale)[
                n * ss[0] + (k / 16) * ss[1]])), second);
      } else {
        a = modelOptFp8Activation(a, second) * second;
        b = static_cast<float>(static_cast<const float8*>(w)[n * ws[0] + k * ws[1]]) * weightScale;
      }
      sum = fmaf(a, b, sum);
    }
    if (floatOutput) static_cast<float*>(z)[zo] = sum;
    else static_cast<X*>(z)[zo] = static_cast<X>(sum);
  }
}

// ─── Contiguous fast path: block-tiled dequant GEMV ─────────────────────────
//
// Admitted only after modelOptTiledEligible proves every operand is dense
// row-major and the packed weights are word aligned (no ews() shortcut).
//
// Work decomposition. A block owns kTiledColumnsPerWarp output columns per
// warp and walks K in tiles of kTiledTileLength activations. Each tile is
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
// each element slot, so a warp's 32 reads hit 32 consecutive floats (no bank
// conflicts); the transposition happens once, while staging.
//
// All loops that contain a barrier are block-uniform and no thread returns
// early; out-of-range columns are skipped per warp (the guard derives from
// the warp index, so it is warp-uniform and never splits a shuffle).
static constexpr int kElementsPerWord = 8;  // one 32-bit word: 8 FP4 nibbles or 8 FP8 bytes
static constexpr int kTiledRowsPerPass = 8;
static constexpr int kTiledColumnsPerWarp = 2;
static constexpr int kTiledTileWords = 128;  // a multiple of the warp size
static constexpr int kTiledTileLength = kTiledTileWords * kElementsPerWord;

template <typename X>
SD_KERNEL static void modelOptLinearTiledKernel(
    const X* __restrict__ x, const void* __restrict__ w, const void* __restrict__ scale,
    const float* __restrict__ secondScale, void* __restrict__ z,
    LongType rows, LongType nColumns, LongType kLength, bool nvfp4, bool floatOutput) {
  using AccT = typename simdOps::AggregateType<X>::type;
  using sd::device::WARP_SIZE;
  static_assert(kTiledTileWords % WARP_SIZE == 0, "a tile must hold whole lane strides");
  __shared__ AccT activations[kTiledRowsPerPass][kElementsPerWord * kTiledTileWords];

  const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const LongType columnsPerBlock = static_cast<LongType>(blockDim.x / WARP_SIZE) * kTiledColumnsPerWarp;
  const float second = secondScale[0];
  if (blockIdx.x == 0 && threadIdx.x == 0 && !modelOptValidScale(second)) asm("trap;");
  const AccT weightScale = nvfp4 ? static_cast<AccT>(1)
                                 : static_cast<AccT>(modelOptScaleChecked(static_cast<const float*>(scale)[0]));
  const auto* packedWeights = static_cast<const uint8_t*>(w);
  const auto* fp8Weights = static_cast<const float8*>(w);
  const auto* blockScales = static_cast<const float8*>(scale);

  for (LongType blockColumn = static_cast<LongType>(blockIdx.x) * columnsPerBlock; blockColumn < nColumns;
       blockColumn += static_cast<LongType>(gridDim.x) * columnsPerBlock) {
    const LongType warpColumn = blockColumn + static_cast<LongType>(warp) * kTiledColumnsPerWarp;

    for (LongType rowBase = 0; rowBase < rows; rowBase += kTiledRowsPerPass) {
      const int passRows = static_cast<int>(sd::math::sd_min<LongType>(rows - rowBase, kTiledRowsPerPass));
      AccT sum[kTiledColumnsPerWarp][kTiledRowsPerPass];
      for (int c = 0; c < kTiledColumnsPerWarp; ++c)
        for (int r = 0; r < kTiledRowsPerPass; ++r) sum[c][r] = static_cast<AccT>(0);

      for (LongType tileStart = 0; tileStart < kLength; tileStart += kTiledTileLength) {
        const int tileWords =
            static_cast<int>(sd::math::sd_min<LongType>(kLength - tileStart, kTiledTileLength)) / kElementsPerWord;

        __syncthreads();  // every warp is done reading the previous tile
        for (int i = static_cast<int>(threadIdx.x); i < passRows * kElementsPerWord * tileWords;
             i += static_cast<int>(blockDim.x)) {
          const int tileWord = i % tileWords;
          const int element = (i / tileWords) % kElementsPerWord;
          const int r = i / (tileWords * kElementsPerWord);
          const AccT value = static_cast<AccT>(
              x[(rowBase + r) * kLength + tileStart + tileWord * kElementsPerWord + element]);
          activations[r][element * kTiledTileWords + tileWord] =
              nvfp4 ? value : static_cast<AccT>(modelOptFp8Activation(value, second) * second);
        }
        __syncthreads();

        const LongType tileWordStart = tileStart / kElementsPerWord;
        for (int tileWord = lane; tileWord < tileWords; tileWord += WARP_SIZE) {
          const LongType word = tileWordStart + tileWord;
          for (int c = 0; c < kTiledColumnsPerWarp; ++c) {
            const LongType n = warpColumn + c;
            if (n >= nColumns) break;  // warp-uniform

            AccT weights[kElementsPerWord];
            if (nvfp4) {
              // A word is half of one 16-element scale block.
              const float blockScale =
                  modelOptScaleChecked(static_cast<float>(blockScales[n * (kLength / 16) + word / 2]));
              const uint32_t packed = reinterpret_cast<const uint32_t*>(packedWeights + n * (kLength / 2))[word];
              for (int e = 0; e < kElementsPerWord; ++e)
                weights[e] = static_cast<AccT>(modelOptNvfp4Weight<X>(
                    static_cast<unsigned char>((packed >> (4 * e)) & 15), blockScale, second));
            } else {
              const float8* wordWeights = fp8Weights + n * kLength + word * kElementsPerWord;
              for (int e = 0; e < kElementsPerWord; ++e)
                weights[e] = static_cast<AccT>(wordWeights[e]) * weightScale;
            }

            for (int r = 0; r < kTiledRowsPerPass; ++r) {
              if (r >= passRows) break;  // warp-uniform
              for (int e = 0; e < kElementsPerWord; ++e)
                sum[c][r] = sd::math::sd_fma<AccT>(activations[r][e * kTiledTileWords + tileWord], weights[e],
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
          if (lane == 0) {
            const LongType zOffset = (rowBase + r) * nColumns + n;
            if (floatOutput)
              static_cast<float*>(z)[zOffset] = static_cast<float>(total);
            else
              static_cast<X*>(z)[zOffset] = static_cast<X>(total);
          }
        }
      }
    }
  }
}

// Host-current scales are validated here, before any launch, so invalid input
// fails with an exception instead of a device trap; the compute kernels also
// validate every scale they read on the stream.
//  - Scalar scales (global/input scale, FP8 weight scale) are checked whenever
//    their host copy is current: one comparison per call.
//  - NVFP4 block-scale tensors (megabytes for a large model) are scanned when
//    their host copy is current and the result is recorded as a content stamp
//    on their DataBuffer, so a model's constant scales are scanned once. The
//    stamp dies with the buffer and with any write to it; the previous cache,
//    keyed on host addresses, vouched for new buffers the allocator placed at
//    a validated buffer's freed address.
static void validateHostCurrentScales(NDArray* scale, NDArray* secondScale, bool nvfp4) {
  if (secondScale->isActualOnHostSide() && !modelOptValidScale(secondScale->bufferAsT<float>()[0]))
    throw std::invalid_argument("ModelOpt linear: global/input scale must be positive and finite");
  if (scale->isEmpty()) return;
  if (!nvfp4) {
    if (scale->isActualOnHostSide() && !modelOptValidScale(scale->bufferAsT<float>()[0]))
      throw std::invalid_argument("ModelOpt FP8 linear: weight scale must be positive and finite");
    return;
  }
  if (!scale->isActualOnHostSide()) return;
  // A stamp describes the whole buffer, so it is only read or written when the
  // array covers every byte of it.
  DataBuffer* buffer = scale->dataBuffer();
  const bool coversBuffer = buffer != nullptr && scale->offset() == 0 &&
                            shape::isDenseRowMajor(scale->shapeInfo()) &&
                            scale->lengthOf() * static_cast<LongType>(scale->sizeOfT()) == buffer->getLenInBytes();
  if (coversBuffer && buffer->isContentValidated(ContentPredicate::MODELOPT_SCALES_VALID)) return;
  const auto* data = scale->bufferAsT<float8>();
  const auto* strides = scale->stridesOf();
  const LongType rows = scale->sizeAt(0), cols = scale->sizeAt(1);
  for (LongType row = 0; row < rows; ++row)
    for (LongType col = 0; col < cols; ++col)
      if (!modelOptValidScale(static_cast<float>(data[row * strides[0] + col * strides[1]])))
        throw std::invalid_argument("ModelOpt NVFP4 linear: every block scale must be positive and finite");
  if (coversBuffer) buffer->stampContentValidated(ContentPredicate::MODELOPT_SCALES_VALID);
}

// Fast-path proof. Every precondition is explicit; no ews() shortcut. Each
// operand must be dense C-order with exactly the row-major strides its shape
// implies, and the packed weight rows must be whole 32-bit words so the
// lane-strided uint32_t loads stay aligned. Dense-stride offset views pass the
// stride proof but must also carry a word-aligned shifted base; anything else
// falls back to the general view-safe kernel.
static bool modelOptTiledEligible(NDArray* x, NDArray* w, NDArray* scale,
                                 NDArray* z, bool nvfp4) {
  if (x->isEmpty() || w->isEmpty() || scale->isEmpty() || z->isEmpty()) return false;
  const LongType k = x->sizeAt(-1);
  // K must be whole 32-bit words so the lane-strided uint32_t loads stay
  // aligned (this also implies the NVFP4 row length K/2 is a word multiple).
  if (k % 8 != 0) return false;
  // X must be a dense [rows, K] row-major block (rank-1 is a single K row);
  // W dense [N, K/2] (NVFP4) or [N, K] (FP8); NVFP4 block scales dense
  // [N, K/16] (the FP8 scale is a rank-0 scalar with no stride to prove); Z
  // dense [rows, N] over the flattened leading dims. The kernel addresses
  // every operand as exactly that packed layout.
  if (!shape::isDenseRowMajor(x->shapeInfo()) || !shape::isDenseRowMajor(w->shapeInfo()) ||
      !shape::isDenseRowMajor(z->shapeInfo()))
    return false;
  if (nvfp4 && !shape::isDenseRowMajor(scale->shapeInfo())) return false;
  // Word alignment: the shifted view base and every row start must be
  // 4-byte aligned for the reinterpret_cast<uint32_t*> loads.
  const uintptr_t wBase = reinterpret_cast<uintptr_t>(w->specialBuffer());
  return wBase % 4 == 0;
}

template <typename X>
static void modelOptLinear_(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                            NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput,
                            dim3 dims) {
  auto* stream = context->getCudaStream();
  const auto* xShape = x->specialShapeInfo();
  const auto* wShape = w->specialShapeInfo();
  const auto* sShape = scale->specialShapeInfo();
  const auto* zShape = z->specialShapeInfo();
  const LongType needed = (z->lengthOf() - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  modelOptLinearKernel<X><<<blocks, dims.y, dims.z, *stream>>>(
      x->isEmpty() ? nullptr : static_cast<const X*>(x->specialBuffer()),
      w->isEmpty() ? nullptr : w->specialBuffer(), scale->isEmpty() ? nullptr : scale->specialBuffer(),
      static_cast<const float*>(secondScale->specialBuffer()), z->specialBuffer(),
      xShape, wShape, sShape, zShape, z->lengthOf(), nvfp4, floatOutput);
}

BUILD_SINGLE_TEMPLATE(void modelOptLinear_,
    (LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale,
     NDArray* z, bool nvfp4, bool floatOutput, dim3 dims), SD_MODELOPT_LINEAR_TYPES);

template <typename X>
static void modelOptLinearTiled_(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                 NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput,
                                 dim3 dims) {
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
  modelOptLinearTiledKernel<X><<<blocks, dims.y, dims.z, *stream>>>(
      static_cast<const X*>(x->specialBuffer()), w->specialBuffer(), scale->specialBuffer(),
      static_cast<const float*>(secondScale->specialBuffer()), z->specialBuffer(),
      rows, nColumns, kLength, nvfp4, floatOutput);
}

BUILD_SINGLE_TEMPLATE(void modelOptLinearTiled_,
    (LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale,
     NDArray* z, bool nvfp4, bool floatOutput, dim3 dims), SD_MODELOPT_LINEAR_TYPES);

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

// ─── FP8 on tensor cores: quantize once, then one scaled cuBLASLt GEMM ──────
//
// Produces the dense row-major E4M3 activation operand q = E4M3(clamp(x /
// inputScale)) — the quantizer the native kernels apply per element — from an
// X of any layout. It also carries the on-stream validation of both scalar
// scales, which the library GEMM cannot perform.
template <typename X>
SD_KERNEL static void modelOptFp8QuantizeKernel(const X* x, const LongType* xShapeInfo, float8* quantized,
                                                LongType length, const float* inputScale,
                                                const float* weightScale) {
  const float scale = inputScale[0];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    modelOptScaleChecked(scale);
    modelOptScaleChecked(weightScale[0]);
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
    quantized[linearIndex] = modelOptFp8Quantize(static_cast<float>(x[xOffset]), scale);
  }
}

template <typename X>
static void modelOptFp8Quantize_(LaunchContext* context, NDArray* x, float8* quantized, NDArray* scale,
                                 NDArray* secondScale, dim3 dims) {
  const LongType length = x->lengthOf();
  const LongType needed = (length - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  modelOptFp8QuantizeKernel<X><<<blocks, dims.y, dims.z, *context->getCudaStream()>>>(
      static_cast<const X*>(x->specialBuffer()), x->specialShapeInfo(), quantized, length,
      static_cast<const float*>(secondScale->specialBuffer()), static_cast<const float*>(scale->specialBuffer()));
}

BUILD_SINGLE_TEMPLATE(void modelOptFp8Quantize_,
    (LaunchContext* context, NDArray* x, float8* quantized, NDArray* scale, NDArray* secondScale, dim3 dims),
    SD_MODELOPT_LINEAR_TYPES);

// cuBLASLt FP8 GEMMs need 16-byte aligned operand bases and leading
// dimensions. W and Z are addressed as dense row-major matrices; X needs no
// layout proof because the quantizer writes a dense copy.
static bool modelOptFp8LtEligible(NDArray* x, NDArray* w, NDArray* z) {
  const LongType depth = x->sizeAt(-1);
  const LongType columns = w->sizeAt(0);
  if (depth <= 0 || depth % 16 != 0) return false;
  if ((columns * static_cast<LongType>(z->sizeOfT())) % 16 != 0) return false;
  if (!shape::isDenseRowMajor(w->shapeInfo()) || !shape::isDenseRowMajor(z->shapeInfo())) return false;
  return reinterpret_cast<uintptr_t>(w->specialBuffer()) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(z->specialBuffer()) % 16 == 0;
}

// Z = (inputScale * weightScale) * E4M3(X) . E4M3(W)^T: every product is the
// same exact FP32 product the native kernels form; only the order of the FP32
// accumulation belongs to the library. Returns false when cuBLASLt has no
// algorithm for the problem.
static bool modelOptFp8LinearLt(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                NDArray* secondScale, NDArray* z, const cudaDeviceProp& prop) {
  const LongType depth = x->sizeAt(-1);
  const LongType rows = x->lengthOf() / depth;
  PointersManager manager(context, "modelOptFp8LinearLt");
  auto* quantized = static_cast<float8*>(manager.allocateDevMem(rows * depth * sizeof(float8)));
  const dim3 dims = modelOptLaunchDims("modelopt_fp8_quantize", prop);
  BUILD_SINGLE_SELECTOR(x->dataType(), modelOptFp8Quantize_, (context, x, quantized, scale, secondScale, dims),
                        SD_MODELOPT_LINEAR_TYPES);
  return MmulHelper::ltMatmulScaled(context, quantized, w->specialBuffer(), z->specialBuffer(), rows, w->sizeAt(0),
                                    depth, FLOAT8, z->dataType(),
                                    static_cast<const float*>(secondScale->specialBuffer()),
                                    static_cast<const float*>(scale->specialBuffer()));
}

void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                    NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput) {
  auto* stream = context->getCudaStream();
  const bool capturing = DebugHelper::inGraphCapture(stream);
  if (!capturing) validateHostCurrentScales(scale, secondScale, nvfp4);
  const cudaDeviceProp& prop = modelOptDeviceProps(context);
  const bool tiled = modelOptTiledEligible(x, w, scale, z, nvfp4);
  const dim3 dims = modelOptLaunchDims(tiled ? "modelopt_linear_tiled" : "modelopt_linear", prop);
  if (tiled && dims.y % sd::device::WARP_SIZE != 0)
    throw std::invalid_argument("ModelOpt linear: tiled block size must be a warp multiple");

  if (z->isEmpty()) {
    // At least one thread must validate the scalar even if the block-scale
    // tensor is empty. This is validation work, not an empty-output compute launch.
    NDArray::prepareSpecialUse({}, {scale, secondScale});
    const auto* sShape = scale->specialShapeInfo();
    const LongType count = scale->lengthOf();
    const LongType needed = count == 0 ? 1 : (count - 1) / dims.y + 1;
    const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
    modelOptValidateScalesKernel<<<blocks, dims.y, 0, *stream>>>(
        scale->isEmpty() ? nullptr : scale->specialBuffer(),
        static_cast<const float*>(secondScale->specialBuffer()), sShape, count, nvfp4);
    NDArray::registerSpecialUse({}, {scale, secondScale});
    if (!capturing) DebugHelper::checkGlobalErrorCode("ModelOpt linear launch failed");
    return;
  }

  NDArray::prepareSpecialUse({z}, {x, w, scale, secondScale});
  const LongType depth = x->sizeAt(-1);
  const LongType rows = depth > 0 ? x->lengthOf() / depth : 0;
  const bool fp8Eligible = !nvfp4 && modelOptFp8LtEligible(x, w, z);
  const bool fp8OnTensorCores =
      fp8Eligible && modelOptFp8LinearLt(context, x, w, scale, secondScale, z, prop);
  if (!nvfp4 && !fp8OnTensorCores) {
    DSP_DIAG(FALLBACK,
             "ModelOpt FP8 linear not on cuBLASLt: rows=%lld columns=%lld depth=%lld eligible=%d "
             "(depth%%16=%lld zRowBytes%%16=%lld wDense=%d zDense=%d wAlign16=%d zAlign16=%d)",
             static_cast<long long>(rows), static_cast<long long>(w->sizeAt(0)), static_cast<long long>(depth),
             fp8Eligible ? 1 : 0, static_cast<long long>(depth % 16),
             static_cast<long long>((w->sizeAt(0) * static_cast<LongType>(z->sizeOfT())) % 16),
             shape::isDenseRowMajor(w->shapeInfo()) ? 1 : 0, shape::isDenseRowMajor(z->shapeInfo()) ? 1 : 0,
             reinterpret_cast<uintptr_t>(w->specialBuffer()) % 16 == 0 ? 1 : 0,
             reinterpret_cast<uintptr_t>(z->specialBuffer()) % 16 == 0 ? 1 : 0);
  }
  const bool nvfp4OnTensorCores =
      nvfp4 && WeightOnlyGemm::isAdmitted(WeightOnlyFormat::MODELOPT_NVFP4, x, w, scale, z);
  DSP_DIAG(BACKEND, "ModelOpt %s linear path=%s rows=%lld columns=%lld depth=%lld", nvfp4 ? "NVFP4" : "FP8",
           nvfp4OnTensorCores ? "weight-only-mma" : fp8OnTensorCores ? "cublaslt" : tiled ? "tiled" : "general",
           static_cast<long long>(rows), static_cast<long long>(w->sizeAt(0)), static_cast<long long>(depth));
  if (nvfp4OnTensorCores) {
    WeightOnlyGemm::run(context, WeightOnlyFormat::MODELOPT_NVFP4, x, w, scale, secondScale, z, floatOutput);
  } else if (!fp8OnTensorCores) {
    if (tiled) {
      BUILD_SINGLE_SELECTOR(x->dataType(), modelOptLinearTiled_,
          (context, x, w, scale, secondScale, z, nvfp4, floatOutput, dims), SD_MODELOPT_LINEAR_TYPES);
    } else {
      BUILD_SINGLE_SELECTOR(x->dataType(), modelOptLinear_,
          (context, x, w, scale, secondScale, z, nvfp4, floatOutput, dims), SD_MODELOPT_LINEAR_TYPES);
    }
  }
  NDArray::registerSpecialUse({z}, {x, w, scale, secondScale});
  if (!capturing) DebugHelper::checkGlobalErrorCode("ModelOpt linear launch failed");
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
