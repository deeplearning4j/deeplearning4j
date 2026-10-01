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

#include <ops/declarable/helpers/token_sample.h>
#include <ops/declarable/helpers/cuda/argmax_row_scan.cuh>
#include <ops/declarable/helpers/reproducible_math.h>
#include <ops/declarable/helpers/sampling_penalties.h>
#include <array/NDArray.h>
#include <array/DataTypeUtils.h>
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
#include <cuda_runtime.h>

#include <type_traits>
#include <vector>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

/**
 * SHARED TIE + ABSENT-CANDIDATE CONTRACT (review rounds 6+7): the scalar
 * GREEDY selector must agree with the CPU argmax and the autoregressive
 * decode kernels — larger value wins, EXACT ties resolve to the SMALLER
 * vocabulary index, and a thread that saw NO element carries an ABSENT
 * candidate (index = vocabSize) that can never win. The old finite
 * -DataTypeUtils::max<AccT>() init at index 0 (a) beat very-negative real
 * logits, (b) let an unused thread's synthetic candidate win a masked /
 * extreme-value row ([-inf, -FLT_MAX] selected 0 instead of 1), and (c)
 * kept the left entry on ties. The value of an absent candidate is -inf:
 * the floating maximum-reduction identity (NVIDIA convention).
 */
template <typename AccT>
static SD_DEVICE inline bool greedyTakeOther(AccT currentVal, LongType currentIdx,
                                             AccT otherVal, LongType otherIdx,
                                             LongType vocabSize) {
    const bool currentValid = currentIdx < vocabSize;
    const bool otherValid = otherIdx < vocabSize;
    if (!otherValid) return false;             // absent candidate never wins
    if (!currentValid) return true;            // a real candidate beats absent
    if (otherVal > currentVal) return true;
    // Exact tie: smaller vocabulary index wins (CPU lowest-index contract).
    return otherVal == currentVal && otherIdx < currentIdx;
}

// Dynamic shared memory of the sampling kernels: an 8-byte entry per thread (an index or a
// measure), then two accumulator entries per thread.
template <typename AccT>
struct TokenSampleShared {
  LongType* index;
  AccT* first;
  AccT* second;

  SD_DEVICE explicit TokenSampleShared(unsigned char* base)
      : index(reinterpret_cast<LongType*>(base)),
        first(reinterpret_cast<AccT*>(index + blockDim.x)),
        second(first + blockDim.x) {}

  static size_t bytes(unsigned int threads) {
    return static_cast<size_t>(threads) * (sizeof(LongType) + 2 * sizeof(AccT));
  }
};

// Folds the per-thread entries of the block onto entry 0 for any block size (the launch
// dimensions can be overridden to any value): each step merges the upper part of the active
// entries onto the lower part. merge(into, from) combines entry from into entry into. Every
// thread of the block calls it once its entries are written; entry 0 is final on return.
template <typename Merge>
static SD_DEVICE void tokenSampleFold(Merge merge) {
  __syncthreads();
  for (unsigned int active = blockDim.x; active > 1;) {
    const unsigned int kept = (active + 1) / 2;
    if (threadIdx.x < active - kept) merge(threadIdx.x, threadIdx.x + kept);
    __syncthreads();
    active = kept;
  }
}

// Greedy selection, one block per row. Per-thread scan and candidate contract:
// cuda/argmax_row_scan.cuh (the CPU argmax: NaN is never selected, lowest index wins ties, a row
// with no value above -inf yields index 0 at probability 0); the block merge applies the same
// rule (greedyTakeOther).
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void tokenSampleGreedyKernel(
    const T* logits, LongType* output, LongType outputStride,
    typename simdOps::AggregateType<T>::type* probabilities, LongType probabilityStride, LongType vocabSize,
    LongType rowStride, LongType elemStride, LongType rowOffset) {
  using AccT = typename simdOps::AggregateType<T>::type;
  extern __shared__ unsigned char tokenSampleSharedMemory[];
  const TokenSampleShared<AccT> shared(tokenSampleSharedMemory);
  const LongType b = blockIdx.x;

  AccT localMax;
  LongType localIndex;
  bool unusedNan = false;
  argmaxScanRow<T, AccT>(logits + b * rowStride + rowOffset, vocabSize, elemStride, localMax, localIndex, false,
                         unusedNan);
  shared.first[threadIdx.x] = localMax;
  shared.index[threadIdx.x] = localIndex;
  tokenSampleFold([&](unsigned int into, unsigned int from) {
    if (greedyTakeOther(shared.first[into], shared.index[into], shared.first[from], shared.index[from], vocabSize)) {
      shared.first[into] = shared.first[from];
      shared.index[into] = shared.index[from];
    }
  });

  if (threadIdx.x == 0) {
    const bool found = shared.index[0] < vocabSize;
    output[b * outputStride] = found ? shared.index[0] : 0;
    if (probabilities != nullptr) probabilities[b * probabilityStride] = static_cast<AccT>(found ? 1 : 0);
  }
}

// A row of logits read as softmax weights (tokenSampleWeight): every pass recomputes a weight
// bit for bit, so the passes agree on which tokens a threshold keeps.
template <typename T, typename AccT>
struct TokenSampleRow {
  const T* logits;
  LongType vocabSize;
  LongType elemStride;
  AccT invTemp;
  AccT rowMax;

  SD_DEVICE AccT weight(LongType v) const {
    return tokenSampleWeight<AccT>(tokenSampleScaled<AccT>(static_cast<AccT>(logits[v * elemStride]), invTemp),
                                   rowMax);
  }
};

// What a threshold keeps of a row: the measure of the kept tokens (their count when M is LongType,
// their weight when M is the accumulator type), the smallest kept weight and the largest positive
// weight below the threshold (-1 when there is none).
template <typename M, typename AccT>
struct TokenSampleKept {
  M measure;
  AccT up;
  AccT down;
};

// The measure a kept weight adds: the weight itself for a mass, one for a count.
template <typename M, typename AccT>
SD_DEVICE SD_INLINE M tokenSampleMeasureOf(AccT weight) {
  if constexpr (std::is_same<M, AccT>::value) {
    return weight;
  } else {
    return static_cast<M>(1);
  }
}

template <typename M, typename AccT>
SD_DEVICE SD_INLINE M tokenSampleMeasureAdd(M measure, M addend) {
  if constexpr (std::is_same<M, AccT>::value) {
    return reproducible::add<AccT>(measure, addend);
  } else {
    return measure + addend;
  }
}

// Every thread returns the same result: the measure sums per thread in stride order, then over
// the block in a fixed fold order, so it is the same function of the threshold in every pass and
// never grows with the threshold (rounded sums of nonnegative terms are monotone).
template <typename M, typename T, typename AccT>
static SD_DEVICE TokenSampleKept<M, AccT> tokenSampleKeep(const TokenSampleRow<T, AccT>& row, AccT threshold,
                                                         const TokenSampleShared<AccT>& shared) {
  const AccT zero = static_cast<AccT>(0);
  M measure = static_cast<M>(0);
  AccT up = DataTypeUtils::infOrMax<AccT>();
  AccT down = static_cast<AccT>(-1);
  for (LongType v = threadIdx.x; v < row.vocabSize; v += blockDim.x) {
    const AccT w = row.weight(v);
    if (!(w > zero)) continue;
    if (w >= threshold) {
      measure = tokenSampleMeasureAdd<M, AccT>(measure, tokenSampleMeasureOf<M, AccT>(w));
      if (w < up) up = w;
    } else if (w > down) {
      down = w;
    }
  }

  M* measures = reinterpret_cast<M*>(shared.index);
  measures[threadIdx.x] = measure;
  shared.first[threadIdx.x] = up;
  shared.second[threadIdx.x] = down;
  tokenSampleFold([&](unsigned int into, unsigned int from) {
    measures[into] = tokenSampleMeasureAdd<M, AccT>(measures[into], measures[from]);
    if (shared.first[from] < shared.first[into]) shared.first[into] = shared.first[from];
    if (shared.second[from] > shared.second[into]) shared.second[into] = shared.second[from];
  });
  const TokenSampleKept<M, AccT> kept{measures[0], shared.first[0], shared.second[0]};
  __syncthreads();
  return kept;
}

// The largest threshold of at least floor whose kept set is not empty and measures at least need,
// or floor when there is none. The search keeps lo at a threshold that qualifies (floor aside)
// and top at or above the answer, both on weight values: a qualifying midpoint lifts lo to the
// smallest weight it keeps (which keeps the same tokens), any other lowers top to the largest
// weight below it. Each pass halves the interval or closes it, and the answer is exact.
template <typename M, typename T, typename AccT>
static SD_DEVICE AccT tokenSampleThreshold(const TokenSampleRow<T, AccT>& row, AccT floor, M need,
                                           const TokenSampleShared<AccT>& shared) {
  AccT lo = floor;
  AccT top = static_cast<AccT>(1);
  while (lo < top) {
    AccT mid = lo + (top - lo) * static_cast<AccT>(0.5);
    if (!(mid > lo)) mid = top;
    const TokenSampleKept<M, AccT> kept = tokenSampleKeep<M>(row, mid, shared);
    if (kept.measure > static_cast<M>(0) && kept.measure >= need) {
      lo = kept.up;
    } else {
      top = kept.down;
    }
  }
  return lo;
}

// Temperature, top-k and top-p truncated draw, one block per row (see tokenSampleDraw). The
// thresholds are the CPU's: top-k keeps the topK-th largest positive weight (a count search),
// top-p the largest threshold above it still holding topP of the top-k mass (a mass search).
// The draw sums the kept weights of contiguous chunks of the row, one chunk per thread, and one
// thread walks the chunk sums to the chunk holding u_b times the total, then that chunk's tokens.
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void tokenSampleDrawKernel(
    const T* logits, LongType* output, LongType outputStride,
    typename simdOps::AggregateType<T>::type* probabilities, LongType probabilityStride,
    const typename simdOps::AggregateType<T>::type* uniforms, LongType uniformStride, graph::RandomGenerator rng,
    LongType vocabSize, LongType rowStride, LongType elemStride, LongType rowOffset,
    typename simdOps::AggregateType<T>::type invTemp, int topK, typename simdOps::AggregateType<T>::type topP) {
  using AccT = typename simdOps::AggregateType<T>::type;
  extern __shared__ unsigned char tokenSampleSharedMemory[];
  const TokenSampleShared<AccT> shared(tokenSampleSharedMemory);
  const LongType b = blockIdx.x;
  const AccT zero = static_cast<AccT>(0);
  TokenSampleRow<T, AccT> row{logits + b * rowStride + rowOffset, vocabSize, elemStride, invTemp, zero};

  // The maximum starts at the lowest finite value, so a row of -inf weighs 0 throughout.
  AccT localMax = -DataTypeUtils::max<AccT>();
  for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
    const AccT scaled = tokenSampleScaled<AccT>(static_cast<AccT>(row.logits[v * elemStride]), invTemp);
    if (scaled > localMax) localMax = scaled;
  }
  shared.first[threadIdx.x] = localMax;
  tokenSampleFold([&](unsigned int into, unsigned int from) {
    if (shared.first[from] > shared.first[into]) shared.first[into] = shared.first[from];
  });
  row.rowMax = shared.first[0];
  __syncthreads();

  AccT threshold = zero;
  if (topK > 0 && topK < vocabSize) {
    threshold = tokenSampleThreshold<LongType>(row, zero, static_cast<LongType>(topK), shared);
  }
  if (topP > zero && topP < static_cast<AccT>(1)) {
    const AccT massK = tokenSampleKeep<AccT>(row, threshold, shared).measure;
    if (massK > zero) {
      threshold = tokenSampleThreshold<AccT>(row, threshold, reproducible::multiply<AccT>(topP, massK), shared);
    }
  }

  const LongType chunk = (vocabSize + blockDim.x - 1) / blockDim.x;
  const LongType chunkOrigin = static_cast<LongType>(threadIdx.x) * chunk;
  const LongType chunkStart = chunkOrigin < vocabSize ? chunkOrigin : vocabSize;
  const LongType chunkEnd = chunkStart + chunk < vocabSize ? chunkStart + chunk : vocabSize;
  AccT mass = zero;
  LongType lastKept = -1;
  for (LongType v = chunkStart; v < chunkEnd; v++) {
    const AccT w = row.weight(v);
    if (w > zero && w >= threshold) {
      mass = reproducible::add<AccT>(mass, w);
      lastKept = v;
    }
  }
  shared.index[threadIdx.x] = lastKept;
  shared.first[threadIdx.x] = mass;
  __syncthreads();

  if (threadIdx.x != 0) return;
  AccT total = zero;
  LongType last = -1;
  for (unsigned int c = 0; c < blockDim.x; c++) {
    if (shared.index[c] < 0) continue;
    total = reproducible::add<AccT>(total, shared.first[c]);
    last = shared.index[c];
  }
  LongType token = 0;
  AccT probability = zero;
  if (last >= 0) {
    graph::RandomGenerator generator = rng;
    const AccT u = uniforms != nullptr ? uniforms[b * uniformStride] : generator.relativeT<AccT>(b);
    const AccT target = reproducible::multiply<AccT>(u, total);
    // The kept token whose cumulative weight first exceeds the target; the last kept token of the
    // chunk that crosses it, or of the row, when rounding leaves none.
    token = last;
    AccT cumulative = zero;
    for (unsigned int c = 0; c < blockDim.x; c++) {
      if (shared.index[c] < 0) continue;
      const AccT next = reproducible::add<AccT>(cumulative, shared.first[c]);
      if (next > target) {
        const LongType chunkLast = shared.index[c];
        token = chunkLast;
        for (LongType v = static_cast<LongType>(c) * chunk; v < chunkLast; v++) {
          const AccT w = row.weight(v);
          if (w > zero && w >= threshold) {
            cumulative = reproducible::add<AccT>(cumulative, w);
            if (cumulative > target) {
              token = v;
              break;
            }
          }
        }
        break;
      }
      cumulative = next;
    }
    probability = reproducible::divide<AccT>(row.weight(token), total);
  }
  output[b * outputStride] = token;
  if (probabilities != nullptr) probabilities[b * probabilityStride] = probability;
}

// Per-row operands the kernels address through a stride in the kernels' data types are used in
// place; others stage through arrays of those types, converted by the central assign (FP8
// included). Every staged array retires behind its last consumer on the stream.
template <typename T>
static void tokenSampleDraw_(NDArray* logits, NDArray* output, NDArray* probabilities, NDArray* uniforms,
                             graph::RandomGenerator rng, double temperature, int topK, double topP,
                             LaunchContext* context) {
  using AccT = typename simdOps::AggregateType<T>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();
  const TokenSampleRows rows = tokenSampleRows(logits);
  const bool greedy = tokenSampleIsGreedy(temperature, topK, topP);
  const AccT invTemp = static_cast<AccT>(temperature > 0.0 ? 1.0 / temperature : 1.0);
  const AccT keepMass = static_cast<AccT>(topP);
  std::vector<LongType> batchShape = {rows.batch};

  LongType tokenStride = tokenSampleRowStride(output);
  NDArray* tokens = output;
  if (tokenStride < 0) {
    tokens = new NDArray('c', batchShape, output->dataType(), context);
    tokenStride = 1;
  }

  NDArray* chosen = nullptr;
  LongType chosenStride = 0;
  if (probabilities != nullptr) {
    chosenStride = tokenSampleRowStride(probabilities);
    chosen = probabilities;
    if (probabilities->dataType() != accType || chosenStride < 0) {
      chosen = new NDArray('c', batchShape, accType, context);
      chosenStride = 1;
    }
  }

  NDArray* draws = nullptr;
  LongType drawStride = 0;
  if (!greedy && uniforms != nullptr) {
    drawStride = tokenSampleRowStride(uniforms);
    draws = uniforms;
    if (uniforms->dataType() != accType || drawStride < 0) {
      draws = new NDArray('c', batchShape, accType, context);
      draws->assign(uniforms);
      drawStride = 1;
    }
  }

  auto* stream = context->getCudaStream();
  const dim3 dims = getLaunchDims("token_sample");
  const size_t sharedBytes = TokenSampleShared<AccT>::bytes(dims.y);
  auto* tokenBuffer = static_cast<LongType*>(tokens->specialBuffer());
  auto* chosenBuffer = chosen != nullptr ? static_cast<AccT*>(chosen->specialBuffer()) : nullptr;
  NDArray::prepareSpecialUse({tokens, chosen}, {logits, draws});
  if (greedy) {
    tokenSampleGreedyKernel<T><<<rows.batch, dims.y, sharedBytes, *stream>>>(
        static_cast<const T*>(logits->specialBuffer()), tokenBuffer, tokenStride, chosenBuffer, chosenStride,
        rows.vocabSize, rows.rowStride, rows.elemStride, rows.rowOffset);
  } else {
    const AccT* drawBuffer = draws != nullptr ? static_cast<const AccT*>(draws->specialBuffer()) : nullptr;
    tokenSampleDrawKernel<T><<<rows.batch, dims.y, sharedBytes, *stream>>>(
        static_cast<const T*>(logits->specialBuffer()), tokenBuffer, tokenStride, chosenBuffer, chosenStride,
        drawBuffer, drawStride, rng, rows.vocabSize, rows.rowStride, rows.elemStride, rows.rowOffset, invTemp, topK,
        keepMass);
  }
  NDArray::registerSpecialUse({tokens, chosen}, {logits, draws});
  DebugHelper::checkGlobalErrorCode("tokenSampleDraw failed");

  if (tokens != output) {
    output->assign(tokens);
    MmulHelper::deleteTemporary(tokens);
  }
  if (chosen != probabilities) {
    probabilities->assign(chosen);
    MmulHelper::deleteTemporary(chosen);
  }
  if (draws != nullptr && draws != uniforms) MmulHelper::deleteTemporary(draws);
}

void tokenSampleDraw(NDArray* logits, NDArray* output, NDArray* probabilities, NDArray* uniforms,
                     graph::RandomGenerator rng, double temperature, int topK, double topP,
                     LaunchContext* context) {
  tokenSampleDrawCheck(logits, output, probabilities, uniforms);
  if (logits->lengthOf() == 0) return;
  BUILD_SINGLE_SELECTOR(logits->dataType(), tokenSampleDraw_,
                        (logits, output, probabilities, uniforms, rng, temperature, topK, topP, context),
                        SD_FLOAT_TYPES);
}

void tokenSample(NDArray* logits, NDArray* output,
                     double temperature, int topK, double topP,
                     LongType seed, LaunchContext* context) {
  // A greedy selection draws nothing, so it takes no entropy.
  const bool greedy = tokenSampleIsGreedy(temperature, topK, topP);
  tokenSampleDraw(logits, output, nullptr, nullptr,
                  greedy ? graph::RandomGenerator(1, 1) : tokenSampleGenerator(seed),
                  temperature, topK, topP, context);
}

void tokenSampleWithPenalties(NDArray* logits, NDArray* output,
                                   NDArray* inputIds,
                                   double temperature, int topK,
                                   double topP, double minP,
                                   double repPenalty, double freqPenalty,
                                   double presPenalty,
                                   double typicalP,
                                   double xtcProbability, double xtcThreshold,
                                   LongType seed, LaunchContext* context) {
    // Sampling pipeline order (see token_sample.h for canonical documentation):
    // 1. Penalties (repetition / frequency / presence)
    // 2. Min-p filter
    // 3. Typical-p filter
    // 4. XTC filter
    // 5. Temperature + top-k + top-p + sample (tokenSample_)

    // Step 1: Apply penalties (in-place on device)
    if (inputIds != nullptr && (repPenalty != 1.0 || freqPenalty != 0.0 || presPenalty != 0.0)) {
        applyLogitPenalties(logits, inputIds, repPenalty, freqPenalty, presPenalty, context);
    }

    // Step 2: Apply min-p filtering (in-place on device)
    if (minP > 0.0) {
        applyMinPFilter(logits, minP, context);
    }

    // Step 3: Apply typical-p filtering (in-place on device)
    if (typicalP > 0.0 && typicalP < 1.0) {
        applyTypicalPFilter(logits, typicalP, context);
    }

    // Step 4: Apply XTC filter (in-place on device, stochastic)
    if (xtcProbability > 0.0) {
        applyXtcFilter(logits, xtcProbability, xtcThreshold, seed, context);
    }

    // Step 5: Standard sampling (temperature, topK, topP)
    tokenSample(logits, output, temperature, topK, topP, seed, context);
}

static bool shouldSuppressStopTokens(const TokenSampleConfig& config) {
    return config.minNewTokens > 0
        && config.generatedTokenOffset < config.minNewTokens
        && config.stopTokenIds != nullptr
        && config.stopTokenCount > 0;
}

template <typename T>
static SD_KERNEL void suppressOneStopTokenKernel(void* vlogits,
                                                 const int stopId,
                                                 const LongType batch,
                                                 const LongType vocabSize,
                                                 const LongType rowStride,
                                                 const LongType elemStride,
                                                 const LongType rowOffset) {
    if (stopId < 0 || stopId >= vocabSize) return;
    auto logits = reinterpret_cast<T*>(vlogits);
    LongType b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    LongType base = b * rowStride + rowOffset;
    // -inf sentinel (infOrMax = +inf for float/double, +max for half/bf16) to match the
    // generation subsystem's -infinity masking convention.
    logits[base + static_cast<LongType>(stopId) * elemStride] = static_cast<T>(-sd::DataTypeUtils::infOrMax<T>());
}

template <typename T>
static void suppressStopTokensLauncher(NDArray* logits, const TokenSampleConfig& config,
                                       LaunchContext* context) {
    if (!shouldSuppressStopTokens(config)) return;

    const int rank = logits->rankOf();
    LongType batch = 1;
    LongType vocabSize;
    LongType seqLen = 1;
    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else if (rank == 2) {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    } else {
        batch = logits->sizeAt(0);
        seqLen = logits->sizeAt(1);
        vocabSize = logits->sizeAt(2);
    }

    auto strides = logits->stridesOf();
    LongType rowStride = 0;
    LongType elemStride;
    LongType rowOffset = 0;
    if (rank == 1) {
        elemStride = strides[0];
    } else if (rank == 2) {
        rowStride = strides[0];
        elemStride = strides[1];
    } else {
        rowStride = strides[0];
        elemStride = strides[2];
        rowOffset = (seqLen - 1) * strides[1];
    }

    auto stream = context->getCudaStream();
    dim3 launchDims = getLaunchDims("token_sample");
    int threads = static_cast<int>(launchDims.y);
    int blocks = static_cast<int>((batch + threads - 1) / threads);
    for (int i = 0; i < config.stopTokenCount; i++) {
        suppressOneStopTokenKernel<T><<<blocks, threads, 0, *stream>>>(
            logits->specialBuffer(), config.stopTokenIds[i], batch, vocabSize,
            rowStride, elemStride, rowOffset);
        DebugHelper::checkGlobalErrorCode("suppressOneStopToken failed");
    }
}

static void suppressStopTokens(NDArray* logits, const TokenSampleConfig& config,
                               LaunchContext* context) {
    if (!shouldSuppressStopTokens(config)) return;
    NDArray::prepareSpecialUse({logits}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), suppressStopTokensLauncher,
                          (logits, config, context), SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits});
}

static int resolveScalarStrategy(const TokenSampleConfig& config) {
    if (config.strategy == TOKEN_SAMPLE_AUTO) {
        return (config.temperature <= 0.0 || (config.topK <= 1 && config.topP <= 0.0))
               ? TOKEN_SAMPLE_GREEDY : TOKEN_SAMPLE_SAMPLE;
    }
    return config.strategy;
}

void tokenSamplePolicy(NDArray* logits, NDArray* output,
                       NDArray* inputIds,
                       const TokenSampleConfig& config,
                       TokenSampleResult* result,
                       LaunchContext* context) {
    const int strategy = resolveScalarStrategy(config);
    const bool scalar = config.batchMax == 1 && config.windowMax == 1
                        && config.activeBatch == 1 && config.activeWindow == 1;
    if (!scalar || (strategy != TOKEN_SAMPLE_GREEDY && strategy != TOKEN_SAMPLE_SAMPLE)) {
        THROW_EXCEPTION("tokenSamplePolicy: only scalar GREEDY/SAMPLE is implemented until "
                        "autoregressive_decode exposes the ADR 0106 B/W substrate");
    }

    if (result != nullptr) {
        result->selectedToken = -1;
        result->selectedBatch = 0;
        result->selectedWindow = 0;
        result->acceptedCount = 1;
        result->parentBeam = 0;
        result->score = 0.0;
    }

    suppressStopTokens(logits, config, context);

    if (strategy == TOKEN_SAMPLE_GREEDY) {
        tokenSample(logits, output, 0.0, 0, 0.0, config.seed, context);
    } else {
        const bool hasPenalties = inputIds != nullptr
            && (config.repPenalty != 1.0 || config.freqPenalty != 0.0 || config.presPenalty != 0.0);
        const bool hasExtendedSamplers = config.minP > 0.0
            || (config.typicalP > 0.0 && config.typicalP < 1.0)
            || config.xtcProbability > 0.0;
        if (hasPenalties || hasExtendedSamplers) {
            tokenSampleWithPenalties(logits, output, inputIds,
                                     config.temperature, config.topK, config.topP, config.minP,
                                     config.repPenalty, config.freqPenalty, config.presPenalty,
                                     config.typicalP, config.xtcProbability, config.xtcThreshold,
                                     config.seed, context);
        } else {
            tokenSample(logits, output, config.temperature, config.topK, config.topP,
                        config.seed, context);
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
