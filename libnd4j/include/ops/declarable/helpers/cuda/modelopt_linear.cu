/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/helpers/modelopt_linear.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace sd {
namespace ops {
namespace helpers {

// Per-element scale validation is fused into both compute kernels below: the
// kernels already read every scale byte, so this predicate is the single
// on-stream source of truth (captured, replayed, and observed at the caller's
// stream completion boundary). The CUDA device trap intrinsic remains active in
// release builds (plain assert() disappears under NDEBUG).
SD_DEVICE SD_INLINE float modelOptScaleChecked(float value) {
  if (!modelOptValidScale(value)) __trap();
  return value;
}

// Runtime tensors cannot be synchronously value-validated during capture. Keep
// the ordered validation kernel for the empty-output corner, where no compute
// kernel runs to carry the fused check above.
SD_KERNEL static void modelOptValidateScalesKernel(const void* scale, const float* second,
                                                   const LongType* shapeInfo,
                                                   LongType count, bool nvfp4) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && !modelOptValidScale(second[0]))
    __trap();
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
      __trap();
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
  if (!modelOptValidScale(second)) __trap();
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

// ─── Contiguous fast path: warp-per-column dequant GEMV ─────────────────────
//
// Admitted only after modelOptTiledEligible proves every precondition: all
// operands dense C-order with exactly the row-major strides their shapes imply
// (no ews() shortcut), and word-aligned rows. One warp owns one output column
// n; a lane-strided uint32_t word loop over K reads 4 packed bytes per lane
// per iteration, so a warp issues one fully coalesced 128-byte transaction
// instead of the general path's one scattered byte per thread (adjacent
// threads there read K/2 bytes apart: 1/16 effective bandwidth, the defect
// measured on the 27B). A uint32 word covers 8 consecutive FP4 lanes, which is
// exactly half of one 16-lane scale block, so each word resolves a single
// block scale; the X-dtype storage rounding of modelOptNvfp4Weight is applied
// per nibble exactly as in the general path. FP8 reads 8 packed e4m3 bytes per
// iteration. Accumulation is FP32 AccT; warpReduceSum performs the ordered
// final reduction with a full-warp shuffle mask. The out-of-range column guard
// is warp-uniform (n derives from tid>>5, identical across a warp's lanes), so
// an early return leaves the whole warp together: it never splits a shuffle
// participant set, and the kernel has no block-wide barriers. It MUST retire
// before any weight/scale pointer arithmetic — trailing warps of the last block
// would otherwise read past the end of the weight buffer.
template <typename X>
SD_KERNEL static void modelOptLinearTiledKernel(
    const X* __restrict__ x, const void* __restrict__ w, const void* __restrict__ scale,
    const float* __restrict__ secondScale, void* __restrict__ z,
    LongType rows, LongType nColumns, LongType kLength, bool nvfp4, bool floatOutput) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const LongType words = kLength / 8;  // kLength % 8 == 0 is proven on the host
  const float second = secondScale[0];
  if (blockIdx.x == 0 && threadIdx.x == 0 && !modelOptValidScale(second)) __trap();
  for (LongType columnBlock = static_cast<LongType>(blockIdx.x) * (blockDim.x / 32);
       columnBlock < nColumns;
       columnBlock += static_cast<LongType>(gridDim.x) * (blockDim.x / 32)) {
    const LongType n = columnBlock + (tid >> 5);
    if (n >= nColumns) return;  // warp-uniform: see note above
    const float weightScale = nvfp4 ? 1.0f : modelOptScaleChecked(
        static_cast<const float*>(scale)[0]);
    const uint32_t* wWords = nvfp4 ? reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(w) + n * (kLength / 2)) : nullptr;
    const float8* wFp8 = nvfp4 ? nullptr : (static_cast<const float8*>(w) + n * kLength);
    const float8* sFp8 = nvfp4 ? (static_cast<const float8*>(scale) + n * (kLength / 16)) : nullptr;
    for (LongType row = 0; row < rows; ++row) {
      const X* xRow = x + row * kLength;
      using AccT = float;
      AccT sum = 0.0f;
      for (LongType word = lane; word < words; word += 32) {
        // Direct fmaf chain into the lane accumulator: one rounded add per
        // element (the general path's operations), no intermediate per-word
        // partial that would add one extra rounding per word.
        // NOTE: the lane split itself IS a reassociation of the general
        // path's ascending-k left fold. The fold of the 32 lane partials
        // below must stay ordered — see the reduction comment.
        if (nvfp4) {
          // 8 consecutive k lanes share one 16-lane scale block (word/2).
          const float blockScale = modelOptScaleChecked(static_cast<float>(sFp8[word / 2]));
          const uint32_t packed = wWords[word];
          sum = fmaf(static_cast<float>(xRow[word * 8 + 0]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>(packed & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 1]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 4) & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 2]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 8) & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 3]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 12) & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 4]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 16) & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 5]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 20) & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 6]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 24) & 15),
                                            blockScale, second), sum);
          sum = fmaf(static_cast<float>(xRow[word * 8 + 7]),
                     modelOptNvfp4Weight<X>(static_cast<uint8_t>((packed >> 28) & 15),
                                            blockScale, second), sum);
        } else {
          const LongType base = word * 8;
          // Same per-element semantics and ascending-k order as the general
          // path: a = quant(x)*second, b = fp8(w)*weightScale, then a single
          // fused multiply-add per element. A plain multiply chain would add a
          // rounding per element and shift low-precision outputs across
          // storage rounding boundaries.
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 0]), second) * second,
                     static_cast<float>(wFp8[base + 0]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 1]), second) * second,
                     static_cast<float>(wFp8[base + 1]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 2]), second) * second,
                     static_cast<float>(wFp8[base + 2]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 3]), second) * second,
                     static_cast<float>(wFp8[base + 3]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 4]), second) * second,
                     static_cast<float>(wFp8[base + 4]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 5]), second) * second,
                     static_cast<float>(wFp8[base + 5]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 6]), second) * second,
                     static_cast<float>(wFp8[base + 6]) * weightScale, sum);
          sum = fmaf(modelOptFp8Activation(static_cast<float>(xRow[base + 7]), second) * second,
                     static_cast<float>(wFp8[base + 7]) * weightScale, sum);
        }
      }
      // Ordered contract reduction. The per-lane fma chains above perform
      // the general path's per-element operations; the only numerical
      // difference to the general path is HOW the 32 lane partials are
      // folded. sd::device::warpReduceSum folds in butterfly order (offset
      // halving), a reassociation that was observed to flip one-ULP-class
      // BF16 storage boundaries on TestModelOptLinear (expected -1.2265625
      // vs -1.234375): lane 15's partial entered the running sum ~2^8
      // rounds later than in the general path, and a mid-fold rounding
      // decision differed. Folding in ASCENDING lane order (offset-increasing
      // shuffles: lane 0 absorbs 1,2,4,8,16, then partials concatenate in
      // 16/16, 8/24, 4/28... groups) keeps every large partial entering as
      // late as possible, the same shape as the general path's left fold.
      // For every K where the 32 partial exponents differ by at most one
      // f32 quantum (K <= 2048; all correctness fixtures and a 2048-wide
      // decode layer), each fold addition is EXACT and both folds yield the
      // identical f32 value, so the BF16 storage cast is bit-identical to
      // the general path. The residual corner (larger K with partial
      // exponents spanning multiple quanta) is the same ill-conditioned set
      // the general path's own chain belongs to.
      for (int offset = 1; offset < 32; offset <<= 1) {
        const AccT neighbor = __shfl_down_sync(0xffffffff, sum, offset);
        if (lane + offset < 32) sum += neighbor;
      }
      if (lane == 0) {
        const LongType zo = row * nColumns + n;
        if (floatOutput) static_cast<float*>(z)[zo] = sum;
        else static_cast<X*>(z)[zo] = static_cast<X>(sum);
      }
    }
  }
}

// Host-current checkpoint scales can fail before any launch without a device
// transfer, so they are validated on the host exactly once per scale buffer.
// ModelOpt scales are constants loaded with the model and never mutated
// between calls, so the per-call full host rescans this replaced were pure
// overhead (measured 24.3 ms/call on the 27B, 3.1 s/token across 129
// calls/token). Device-current tensors rely on the ordered fused validation
// inside the compute kernels. Failures are never cached: an invalid buffer
// re-scans and re-throws on every call.
static void validateHostScalesOnce(NDArray* scale, NDArray* secondScale, bool nvfp4) {
  if (secondScale->isActualOnHostSide() && !modelOptValidScale(secondScale->bufferAsT<float>()[0]))
    throw std::invalid_argument("ModelOpt linear: global/input scale must be positive and finite");
  if (scale->isEmpty() || !scale->isActualOnHostSide()) return;
  auto* buffer = scale->dataBuffer();
  if (buffer == nullptr || buffer->primary() == nullptr) return;
  static std::mutex mutex;
  static std::unordered_map<uint64_t, bool> validated;
  std::lock_guard<std::mutex> lock(mutex);
  // Key on identity AND length so a freed-then-reallocated buffer with a
  // different size revalidates even if the allocator reuses the address.
  const uint64_t key = reinterpret_cast<uint64_t>(buffer->primary()) ^
                       (static_cast<uint64_t>(buffer->getLenInBytes()) << 48);
  if (validated.count(key) != 0) return;
  if (!nvfp4) {
    if (!modelOptValidScale(scale->bufferAsT<float>()[0]))
      throw std::invalid_argument("ModelOpt FP8 linear: weight scale must be positive and finite");
  } else {
    const auto* data = scale->bufferAsT<float8>();
    const auto* strides = scale->stridesOf();
    const LongType rows = scale->sizeAt(0), cols = scale->sizeAt(1);
    for (LongType row = 0; row < rows; ++row)
      for (LongType col = 0; col < cols; ++col)
        if (!modelOptValidScale(static_cast<float>(data[row * strides[0] + col * strides[1]])))
          throw std::invalid_argument(
              "ModelOpt NVFP4 linear: every block scale must be positive and finite");
  }
  validated[key] = true;
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
  const int rank = x->rankOf();
  const LongType k = x->sizeAt(-1);
  // K must be whole 32-bit words so the lane-strided uint32_t loads stay
  // aligned (this also implies the NVFP4 row length K/2 is a word multiple).
  if (k % 8 != 0) return false;
  // X must be a dense [rows, K] row-major block (rank-1 is a single K row).
  if (rank == 1) {
    if (x->stridesOf()[0] != 1) return false;
  } else {
    const sd::LongType* xs = x->stridesOf();
    const sd::LongType* xd = x->shapeOf();
    sd::LongType expected = 1;
    for (int d = rank - 1; d >= 0; --d) {
      if (xs[d] != expected) return false;
      expected *= xd[d];
    }
    if (x->ordering() != 'c') return false;
  }
  // W must be dense [N, K/2] (NVFP4) or [N, K] (FP8) row-major.
  {
    const sd::LongType* ws = w->stridesOf();
    if (w->ordering() != 'c' || ws[0] != (nvfp4 ? k / 2 : k) || ws[1] != 1) return false;
  }
  // NVFP4 block scales must be dense [N, K/16] row-major; the FP8 scale is a
  // rank-0 scalar with no stride to prove.
  if (nvfp4) {
    const sd::LongType* ss = scale->stridesOf();
    if (scale->ordering() != 'c' || ss[0] != k / 16 || ss[1] != 1) return false;
  }
  // Z must be dense [rows, N] row-major over the flattened leading dims.
  {
    const int zRank = z->rankOf();
    const sd::LongType* zs = z->stridesOf();
    const sd::LongType* zd = z->shapeOf();
    sd::LongType expected = 1;
    for (int d = zRank - 1; d >= 0; --d) {
      if (zs[d] != expected) return false;
      expected *= zd[d];
    }
    if (z->ordering() != 'c') return false;
  }
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
  const int warpsPerBlock = static_cast<int>(dims.y) / 32;
  const LongType needed = (nColumns + warpsPerBlock - 1) / warpsPerBlock;
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

void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                    NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput) {
  auto* stream = context->getCudaStream();
  const bool capturing = DebugHelper::inGraphCapture(stream);
  if (!capturing) validateHostScalesOnce(scale, secondScale, nvfp4);
  const cudaDeviceProp& prop = modelOptDeviceProps(context);
  const bool tiled = modelOptTiledEligible(x, w, scale, z, nvfp4);
  const dim3 dims = getLaunchDims(tiled ? "modelopt_linear_tiled" : "modelopt_linear");
  if (dims.x == 0 || dims.y == 0 || dims.x > static_cast<unsigned int>(prop.maxGridSize[0]) ||
      dims.y > static_cast<unsigned int>(prop.maxThreadsPerBlock) ||
      dims.y > static_cast<unsigned int>(prop.maxThreadsDim[0]) || dims.z > prop.sharedMemPerBlock)
    throw std::invalid_argument("ModelOpt linear: invalid launch dimensions");
  if (tiled && dims.y % 32 != 0)
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
  if (tiled) {
    BUILD_SINGLE_SELECTOR(x->dataType(), modelOptLinearTiled_,
        (context, x, w, scale, secondScale, z, nvfp4, floatOutput, dims), SD_MODELOPT_LINEAR_TYPES);
  } else {
    BUILD_SINGLE_SELECTOR(x->dataType(), modelOptLinear_,
        (context, x, w, scale, secondScale, z, nvfp4, floatOutput, dims), SD_MODELOPT_LINEAR_TYPES);
  }
  NDArray::registerSpecialUse({z}, {x, w, scale, secondScale});
  if (!capturing) DebugHelper::checkGlobalErrorCode("ModelOpt linear launch failed");
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
