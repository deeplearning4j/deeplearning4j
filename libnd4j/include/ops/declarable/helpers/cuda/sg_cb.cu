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
// @author raver119@gmail.com
//
#include <array/NDArrayFactory.h>
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/sg_cb.h>

#include "helpers/DebugHelper.h"


#define HS_MAX_EXP 6.0f

namespace sd {
namespace ops {
namespace helpers {





template <typename T>
SD_KERNEL SD_INLINE void hSoftmaxKernel(void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength,
                              int code, int expLength, bool isInference) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);

  T dot(0.0f);
  T g(0.0f);
  T f(0.0f);

  // dot
  for (int e = 0; e < vectorLength; e++) {
    dot += syn0[e] * syn1[e];
  }

  // gradient
  if (dot < (T)-HS_MAX_EXP || dot >= (T)HS_MAX_EXP) return;

  int idx = static_cast<int>((dot + HS_MAX_EXP) * ((float)expLength / HS_MAX_EXP / 2.0f));

  if (idx >= expLength || idx < 0) return;

  f = expTable[idx];
  g = (static_cast<T>(1.0f) - static_cast<T>(code) - f) * (T)alpha;

  // axpy1

  for (int e = 0; e < vectorLength; e++) {
    neu1e[e] = g * syn1[e] + neu1e[e];
  }

  // axpy2
  if (!isInference) {
    for (int e = 0; e < vectorLength; e++) {
      syn1[e] = g * syn0[e] + syn1[e];
    }
  }
}

template <typename T>
void hSoftmax_(void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code,
               int expLength, bool isInference, cudaStream_t *stream) {
  hSoftmaxKernel<T>
      <<<1, 1, 128, *stream>>>(vsyn0, vsyn1, vexpTable, vneu1e, alpha, vectorLength, code, expLength, isInference);
  sd::DebugHelper::checkErrorCode(stream, "hSoftmaxKernel failed");

}
BUILD_SINGLE_TEMPLATE(void hSoftmax_, (void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference, cudaStream_t *stream), SD_FLOAT_TYPES);

template <typename T>
SD_KERNEL SD_INLINE void nSamplingKernel(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha,
                               int vectorLength, int code, int expLength, bool isInference) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);

  T dot = (T)0.0f;
  T g = (T)0.0f;

  for (int e = 0; e < vectorLength; e++) {
    dot += syn0[e] * syn1Neg[e];
  }

  if (dot > HS_MAX_EXP)
    g = (code - 1) * alpha;
  else if (dot < (T)-HS_MAX_EXP)
    g = (code - 0) * alpha;
  else {
    int idx = (int)((dot + (T)HS_MAX_EXP) * ((T)expLength / HS_MAX_EXP / 2.0));
    if (idx >= expLength) return;

    if (idx < 0) return;

    g = ((T)code - expTable[idx]) * alpha;
  }

  // axpy1
  for (int e = 0; e < vectorLength; e++) {
    neu1e[e] = g * syn1Neg[e] + neu1e[e];
  }

  // axpy2
  if (!isInference) {
    for (int e = 0; e < vectorLength; e++) {
      syn1Neg[e] = g * syn0[e] + syn1Neg[e];
    }
  }
}

template <typename T>
void nSampling_(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code,
                int expLength, bool isInference, cudaStream_t *stream) {
  nSamplingKernel<T>
      <<<1, 1, 128, *stream>>>(vsyn0, vsyn1Neg, vexpTable, vneu1e, alpha, vectorLength, code, expLength, isInference);
  sd::DebugHelper::checkErrorCode(stream, "nSamplingKernel failed");

}
BUILD_SINGLE_TEMPLATE(void nSampling_, (void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference, cudaStream_t *stream), SD_FLOAT_TYPES);

/*
 * binarySearch - find element in haystack buffer (haystack - sorted device memory)
 * */
int binarySearch(const int *haystack, const int needle, const int totalElements) {
  int firstIndex = 0;
  int lastIndex = totalElements - 1;
  int halfIndex = sd::math::sd_floor<float, int>((lastIndex + firstIndex) / (float)2);

  while (haystack[halfIndex] != needle && firstIndex < lastIndex) {
    if (needle < haystack[halfIndex]) {
      lastIndex = halfIndex - 1;
    } else if (needle > haystack[halfIndex]) {
      firstIndex = halfIndex + 1;
    }
    halfIndex = sd::math::sd_floor<float, int>((lastIndex + firstIndex) / (float)2);
  }

  return (haystack[halfIndex] == needle) ? halfIndex : -1;
}
template <typename T>
SD_KERNEL SD_INLINE void addInfVectorKernel(T *neu1, T *infVector, int vectorLength) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (auto i = start; i < vectorLength; i += step) {
    neu1[i] += infVector[i];
  }
}

template <typename T>
SD_KERNEL SD_INLINE void zeroVectorKernel(T *vector, int vectorLength) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start; i < vectorLength; i += step) {
    vector[i] = (T)0.0f;
  }
}

// A work vector - the error of an iteration, the average of a window - starts every use from zero. The kernels that fill
// it run on the stream, so the clearing is a kernel on that stream as well. Unlike cudaMemsetAsync it also reaches a
// vector in pinned host memory, where the buffer of an array lands when the device is out of memory.
template <typename T>
static void zeroVector_(T *vector, const int vectorLength, cudaStream_t *stream) {
  dim3 w2vDims = getLaunchDims("word2vec");
  zeroVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(vector, vectorLength);
  sd::DebugHelper::checkErrorCode(stream, "zeroVectorKernel failed");
}

// The rounds of one target against syn0row, their errors adding up in neu1e: hierarchic softmax first - the points and
// codes of one row of indices and codes, a code of -1 being a padding - negative sampling second. The indices, the
// codes and the negative table are read element by element on the host, so they can be arrays of any numeric type. The
// word to sample negatives for was checked before the first kernel ran.
template <typename T>
static void skipgramRounds_(T *syn0row, T *syn1, T *syn1Neg, T *expTable, NDArray &negTableV, T *neu1e,
                            NDArray &indices, NDArray &codes, const LongType indicesBase, const LongType codesBase,
                            const int hsRounds, const int nsRounds, const int ngStarter, LongType &randomValue,
                            const double alpha, const int vocabSize, const int vectorLength, const int expLength,
                            const int negLength, const bool isInference, cudaStream_t *stream) {
  // hierarchic softmax goes first (if enabled)
  for (int r = 0; r < hsRounds; r++) {
    const int irow = indices.e<int>(indicesBase + r);
    const int code = codes.e<int>(codesBase + r);

    // the code of a padding is -1
    if (code < 0 || irow < 0 || irow >= vocabSize) continue;

    hSoftmax_<T>(syn0row, syn1 + static_cast<LongType>(irow) * vectorLength, expTable, neu1e, alpha, vectorLength, code,
                 expLength, isInference, stream);
  }

  // negative sampling goes second (if enabled)
  if (nsRounds > 0) {
    int irow = ngStarter;
    for (int r = 0; r < nsRounds + 1; r++) {
      if (r == 0) {
        // target is known in advance
      } else {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = sd::math::sd_abs<LongType,LongType>((randomValue >> 16) % negLength);
        irow = idx >= negLength ? -1 : negTableV.e<int>(idx);

        if (irow < 0 || irow >= vocabSize) {
          irow = static_cast<int>(static_cast<unsigned long long>(randomValue) % (vocabSize - 1)) + 1;
        }
        if (irow == ngStarter) continue;
      }

      nSampling_<T>(syn0row, syn1Neg + static_cast<LongType>(irow) * vectorLength, expTable, neu1e, alpha, vectorLength,
                    r == 0 ? 1 : 0, expLength, isInference, stream);
    }
  }
}

// One skipgram round, `iterations` times over. Every iteration trains the vector that learns - the inference vector when
// there is one, the row of the target in syn0 otherwise - against a neu1e that starts from zero, and the learning rate
// decays after it: alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate. With an inference
// vector syn1 and syn1Neg are only read.
template <typename T>
void skipgram_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &expTableV, NDArray &negTableV, NDArray &infV,
               int target, int ngStarter, NDArray &indices, NDArray &codes, double alpha, LongType randomValue,
               const int hsRounds, const int nsRounds, const int iterations, const double minLearningRate) {
  const int vocabSize = s0.sizeAt(0);
  const int vectorLength = s0.sizeAt(1);
  const int expLength = expTableV.lengthOf();
  const int negLength = negTableV.lengthOf();
  auto stream = s0.getContext()->getCudaStream();

  // everything that can be wrong with the arguments is found before the first kernel runs
  if (infV.isEmpty() && (target < 0 || target >= vocabSize)) {
    THROW_EXCEPTION("SkipGram: the target is not a row of syn0");
  }

  if (nsRounds > 0 && (ngStarter < 0 || ngStarter >= vocabSize)) {
    THROW_EXCEPTION("SkipGram: the word to sample negatives for is not a row of syn1Neg");
  }

  if (hsRounds > indices.lengthOf()) {
    THROW_EXCEPTION("SkipGram: there are fewer indices than codes");
  }

  // Prepare device coherence: kernels read and write syn0/syn1/syn1Neg/infV, read expTableV.
  // neu1eArr is a local temporary, not an NDArray input — no prepare needed.
  NDArray::prepareSpecialUse({&s0, &s1, &s1n, &infV}, {&s0, &s1, &s1n, &infV, &expTableV});
  // indices/codes and negTableV are read on the host; bring them to primary.
  NDArray::preparePrimaryUse({}, {&indices, &codes, &negTableV});

  auto syn0 = reinterpret_cast<T *>(s0.specialBuffer());
  auto syn1 = reinterpret_cast<T *>(s1.specialBuffer());
  auto syn1Neg = reinterpret_cast<T *>(s1n.specialBuffer());
  auto expTable = reinterpret_cast<T *>(expTableV.specialBuffer());
  auto infVector = reinterpret_cast<T *>(infV.specialBuffer());

  std::vector<sd::LongType> neuShape = {vectorLength};
  NDArray neu1eArr('c', neuShape, DataTypeUtils::fromT<T>(), s0.getContext());
  T *neu1e = reinterpret_cast<T*>(neu1eArr.specialBuffer());

  // the vector that learns: the inference vector if there is one, the row of the target in syn0 otherwise
  auto syn0row = infVector != nullptr ? infVector : syn0 + static_cast<LongType>(target) * vectorLength;

  for (int iteration = 0; iteration < iterations; iteration++) {
    // neu1e is the error of one iteration: it starts from zero every time
    zeroVector_<T>(neu1e, vectorLength, stream);

    skipgramRounds_<T>(syn0row, syn1, syn1Neg, expTable, negTableV, neu1e, indices, codes, 0, 0, hsRounds, nsRounds,
                       ngStarter, randomValue, alpha, vocabSize, vectorLength, expLength, negLength,
                       infVector != nullptr, stream);

    {
      dim3 w2vDims = getLaunchDims("word2vec");
      addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(syn0row, neu1e, vectorLength);
      sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel failed");
    }

    alpha = ((alpha - minLearningRate) / (iterations - iteration)) + minLearningRate;
  }
  auto err = cudaStreamSynchronize(*stream);
  if (0 != err) {
    { std::string msg = "helpers::skipgram_: Cannot synchronize stream after addInfVectorKernel; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerPrimaryUse({}, {&indices, &codes, &negTableV});
  NDArray::registerSpecialUse({&s0, &s1, &s1n, &infV}, {&expTableV});
}
BUILD_SINGLE_TEMPLATE( void skipgram_,
                      (NDArray & syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                       NDArray &infVector, int target, int ngStarter, NDArray &indices, NDArray &codes, double alpha,
                       sd::LongType randomValue, const int hsRounds, const int nsRounds, const int iterations,
                       const double minLearningRate),
                      SD_FLOAT_TYPES);

// What can be wrong with the arguments of a batch is found before its first kernel runs: a failure after that would
// leave the tables half trained. With an inference vector the targets pull the vector, so no row of syn0 is read for them.
static void checkSkipgramBatch(NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                               NDArray &nextRandom, const int nsRounds, const int vocabSize, const bool isInference,
                               const LongType hsRounds) {
  const LongType numTargets = targets.lengthOf();

  if (lr.lengthOf() < numTargets) {
    THROW_EXCEPTION("SkipGram: every target needs a learning rate");
  }

  if (nextRandom.lengthOf() < numTargets) {
    THROW_EXCEPTION("SkipGram: every target needs a random value");
  }

  if (hsRounds > 0 &&
      (indices.sizeAt(0) < numTargets || codes.sizeAt(0) < numTargets || indices.sizeAt(1) < hsRounds)) {
    THROW_EXCEPTION(
        "SkipGram: the indices and codes of a batch need a row for every target, and the indices as many columns as the "
        "codes");
  }

  if (!isInference) {
    for (LongType t = 0; t < numTargets; t++) {
      const int target = targets.e<int>(t);
      if (target < 0 || target >= vocabSize) {
        THROW_EXCEPTION("SkipGram: the target is not a row of syn0");
      }
    }
  }

  if (nsRounds > 0) {
    if (negStarters.lengthOf() < numTargets) {
      THROW_EXCEPTION("SkipGram: negative sampling needs a word to sample for with every target");
    }

    for (LongType t = 0; t < numTargets; t++) {
      const int ngStarter = negStarters.e<int>(t);
      if (ngStarter < 0 || ngStarter >= vocabSize) {
        THROW_EXCEPTION("SkipGram: the word to sample negatives for is not a row of syn1Neg");
      }
    }
  }
}

/*
 * batched version of skipgram routine
 *
 * Training: every target trains its own row of syn0 (and syn1/syn1Neg), one target after the other.
 * Inference (an inference vector is given): the targets pull the inference vector, every one of them against the same
 * vector, and their errors add up once per iteration; syn1 and syn1Neg are only read, the learning rate of every
 * target decays after each iteration, and its random values go on from one iteration to the next, as they do on the CPU.
 * */
template <typename T>
void skipgramBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &expTableV, NDArray &negTableV, NDArray &infV,
                        NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                        NDArray &nextRandom, const int nsRounds, const bool preciseMode, const int numThreads,
                        const int iterations, const double minLearningRate) {
  auto stream = s0.getContext()->getCudaStream();
  const int vocabSize = s0.sizeAt(0);
  const int vectorLength = s0.sizeAt(1);
  const int expLength = expTableV.lengthOf();
  const int negLength = negTableV.lengthOf();

  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  const bool useHierarchicSoftmax = !codes.isEmpty() && !s1.isEmpty();
  if (useHierarchicSoftmax && (codes.rankOf() != 2 || indices.rankOf() != 2)) {
    THROW_EXCEPTION("SkipGram: the indices and codes of a batch are matrices");
  }

  const LongType hsRounds = useHierarchicSoftmax ? codes.sizeAt(1) : 0;
  const LongType indicesColumns = useHierarchicSoftmax ? indices.sizeAt(1) : 0;
  const LongType codesColumns = useHierarchicSoftmax ? codes.sizeAt(1) : 0;
  const auto numTargets = targets.lengthOf();

  // negTableV, targets, indices, codes, lr, nextRandom, negStarters are all read on the host.
  NDArray::preparePrimaryUse({}, {&negTableV, &targets, &indices, &codes, &lr, &nextRandom, &negStarters});
  checkSkipgramBatch(targets, negStarters, indices, codes, lr, nextRandom, nsRounds, vocabSize, !infV.isEmpty(),
                     hsRounds);

  // Device kernels read and write s0/s1/s1n/infV; read expTableV.
  NDArray::prepareSpecialUse({&s0, &s1, &s1n, &infV}, {&s0, &s1, &s1n, &infV, &expTableV});
  const auto syn0 = reinterpret_cast<T *>(s0.specialBuffer());
  const auto syn1 = reinterpret_cast<T *>(s1.specialBuffer());
  const auto syn1Neg = reinterpret_cast<T *>(s1n.specialBuffer());
  const auto expTable = reinterpret_cast<T *>(expTableV.specialBuffer());
  const auto infVector = reinterpret_cast<T *>(infV.specialBuffer());

  std::vector<sd::LongType> neuShape = {vectorLength};
  NDArray neu1eArr('c', neuShape, DataTypeUtils::fromT<T>(), s0.getContext());
  T *neu1e = reinterpret_cast<T*>(neu1eArr.specialBuffer());

  if (infVector == nullptr) {
    // regular mode provides 0 guarantees for reproducibility
    for (LongType t = 0; t < numTargets; t++) {
      const int target = targets.e<int>(t);

      // neu1e is the error of one target: it starts from zero every time
      zeroVector_<T>(neu1e, vectorLength, stream);

      auto alpha = lr.e<double>(t);
      LongType randomValue = nextRandom.e<LongType>(t);
      const int ngStarter = nsRounds > 0 ? negStarters.e<int>(t) : -1;

      auto syn0row = syn0 + static_cast<LongType>(target) * vectorLength;

      skipgramRounds_<T>(syn0row, syn1, syn1Neg, expTable, negTableV, neu1e, indices, codes, t * indicesColumns,
                         t * codesColumns, hsRounds, nsRounds, ngStarter, randomValue, alpha, vocabSize, vectorLength,
                         expLength, negLength, false, stream);

      {
        dim3 w2vDims = getLaunchDims("word2vec");
        addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(syn0row, neu1e, vectorLength);
        sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel failed");
      }
    }
  } else {
    std::vector<double> lrs(numTargets);
    std::vector<LongType> randoms(numTargets);
    for (LongType t = 0; t < numTargets; t++) {
      lrs[t] = lr.e<double>(t);
      randoms[t] = nextRandom.e<LongType>(t);
    }

    for (int curr = 0; curr < iterations; curr++) {
      // neu1e is the error of one iteration, all the targets together: it starts from zero every time
      zeroVector_<T>(neu1e, vectorLength, stream);

      for (LongType t = 0; t < numTargets; t++) {
        const int ngStarter = nsRounds > 0 ? negStarters.e<int>(t) : -1;

        skipgramRounds_<T>(infVector, syn1, syn1Neg, expTable, negTableV, neu1e, indices, codes, t * indicesColumns,
                           t * codesColumns, hsRounds, nsRounds, ngStarter, randoms[t], lrs[t], vocabSize, vectorLength,
                           expLength, negLength, true, stream);
      }

      {
        dim3 w2vDims = getLaunchDims("word2vec");
        addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(infVector, neu1e, vectorLength);
        sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel failed");
      }

      for (LongType t = 0; t < numTargets; t++) {
        lrs[t] = ((lrs[t] - minLearningRate) / (iterations - curr)) + minLearningRate;
      }
    }
  }

  auto err = cudaStreamSynchronize(*stream);
  if (0 != err) {
    { std::string msg = "helpers::skipgramBatchExec_: Cannot synchronize stream after addInfVectorKernel; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerPrimaryUse({}, {&negTableV, &targets, &indices, &codes, &lr, &nextRandom, &negStarters});
  NDArray::registerSpecialUse({&s0, &s1, &s1n, &infV}, {&expTableV});
}
BUILD_SINGLE_TEMPLATE( void skipgramBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, NDArray &expTable, NDArray &negTable, NDArray &infV,
                       NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                       NDArray &nextRandom, const int nsRounds, const bool preciseMode, const int numThreads,
                       const int iterations, const double minLearningRate),
                      SD_FLOAT_TYPES);


void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target,
              NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue,
              NDArray &inferenceVector, const bool preciseMode, const int numWorkers,const int iterations,double minLearningRate) {
  auto xType = syn0.dataType();
  // single round case
  if ((ngStarter.isScalar() && !ngStarter.isEmpty()) || (target.isScalar() && !target.isEmpty())) {
    // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
    auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();
    NDArray::preparePrimaryUse({}, {&target, &ngStarter, &alpha, &randomValue});
    auto targetV = target.isEmpty() ? -1 : target.e<int>(0);
    auto starterV = ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0);
    auto alphaV = alpha.e<double>(0);
    auto randomV = randomValue.e<LongType>(0);
    NDArray::registerPrimaryUse({}, {&target, &ngStarter, &alpha, &randomValue});
    BUILD_SINGLE_SELECTOR(xType, skipgram_,
                          (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, targetV, starterV, indices, codes,
                           alphaV, randomV, hsRounds, nsRounds, iterations, minLearningRate),
                          SD_FLOAT_TYPES);
  } else if (ngStarter.isVector() || target.isVector()) {
    BUILD_SINGLE_SELECTOR(xType, skipgramBatchExec_,
                          (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, target, ngStarter, indices, codes,
                           alpha, randomValue, nsRounds, preciseMode, numWorkers, iterations, minLearningRate),
                          SD_FLOAT_TYPES);
  } else
    THROW_EXCEPTION("SkipGram: target must have rank 0 or 1");
}


void skipgramInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                       int ngStarter, int nsRounds, NDArray &indices, NDArray &codes, double alpha,
                       LongType randomValue,
                       NDArray &inferenceVector, const bool preciseMode, const int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();
  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();


  /**
   * void skipgram_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &expTableV, NDArray &negTableV, NDArray &infV,
int target, int ngStarter, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
const int hsRounds, const int nsRounds, const int iterations, const double minLearningRate)
   */


  BUILD_SINGLE_SELECTOR(xType, skipgram_,
                        (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, target, ngStarter, indices, codes,
                         alpha, randomValue, hsRounds, nsRounds, iterations, minLearningRate),
                        SD_FLOAT_TYPES);
}





// The sum of the words of a window. The values that pad the window are negative. The words are rows of syn0: the window
// was checked on the host before the first kernel ran.
template <typename T>
static SD_KERNEL SD_INLINE void sumWindowKernel(int *context, T *syn0, T *neu1, int contextWidth, int vectorLength) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int c = start; c < contextWidth; c += step) {
    if (context[c] < 0) continue;

    T *syn0word = syn0 + static_cast<LongType>(context[c]) * vectorLength;

    for (int i = 0; i < vectorLength; i++) {
      neu1[i] += syn0word[i];
    }
  }
}

// The average of a window: neu1 holds the sum of its members, `members` of them.
template <typename T>
SD_KERNEL SD_INLINE void shiftKernel(T *neu1, int members, int vectorLength) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start; i < vectorLength; i += step) {
    neu1[i] /= members;
  }
}

template <typename T>
SD_KERNEL SD_INLINE void fillUpSynonymsKernel(int starter, int contextWidth, int vectorLength, int *lockedWords,
                                              int numLocked, int *context, T *neu1e, T *syn0) {
  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (int c = starter + start; c < contextWidth; c += step) {
    // the words that are locked do not learn; the values that pad the window are negative
    if (c < numLocked && lockedWords[c] == 1) continue;
    if (context[c] < 0) continue;

    T *syn0word = syn0 + static_cast<LongType>(context[c]) * vectorLength;

    for (int i = 0; i < vectorLength; i++) {
      syn0word[i] += neu1e[i];
    }
  }
}

// One cbow round on a single window, `iterations` times over. The indices and codes are read element by element on the
// host, so they can be arrays of any integer type; a code of -1 is a padding. Every iteration averages the window into
// neu1 - over its validContext words that are there (the values that pad the window are negative) and the inference
// vector - and collects the error in neu1e, both starting from zero; the error moves the words of the window - or the
// inference vector, the only thing that learns when there is one - and the learning rate decays after it:
//   alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate
// Of the first numLocked words of the window the ones that are locked do not learn.
template <typename T>
void cbow_(LaunchContext *lc, void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable,
           void *vinfVector, int target, int ngStarter, int *context, int *lockedWords, NDArray &indices,
           NDArray &codes, double alpha, LongType randomValue, const int contextWidth, const int validContext,
           const int numLocked, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength,
           const int expLength, const int negLength, const int numLabels, const bool trainWords, const int iterations,
           const double minLearningRate) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto negTable = reinterpret_cast<T *>(vnegTable);
  auto infVector = reinterpret_cast<T *>(vinfVector);
  auto stream = lc->getCudaStream();

  std::vector<sd::LongType> neuShape = {vectorLength};
  NDArray neu1Arr('c', neuShape, DataTypeUtils::fromT<T>(), lc);
  NDArray neu1eArr('c', neuShape, DataTypeUtils::fromT<T>(), lc);
  T *neu1  = reinterpret_cast<T*>(neu1Arr.specialBuffer());
  T *neu1e = reinterpret_cast<T*>(neu1eArr.specialBuffer());

  // the members of the average: the words of the window that are there, and the inference vector
  const int members = validContext + (infVector != nullptr ? 1 : 0);

  for (int iteration = 0; iteration < iterations; iteration++) {
    // neu1 and neu1e belong to one iteration: both start from zero
    zeroVector_<T>(neu1, vectorLength, stream);
    zeroVector_<T>(neu1e, vectorLength, stream);

    // building neu1 for current window
    sumWindowKernel<T><<<1, 1, 128, *stream>>>(context, syn0, neu1, contextWidth, vectorLength);
    sd::DebugHelper::checkErrorCode(stream, "sumWindowKernel failed");

    // for inference we add additional inference vector
    if (infVector != nullptr) {
      dim3 w2vDims = getLaunchDims("word2vec");
      addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(neu1, infVector, vectorLength);
      sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel failed");

    }

    // average neu1
    if (members > 1) {
      dim3 w2vDims = getLaunchDims("word2vec");
      shiftKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(neu1, members, vectorLength);
      sd::DebugHelper::checkErrorCode(stream, "shiftKernel failed");

    }

    // softmax round
    for (int i = 0; i < hsRounds; i++) {
      const int cIndex = indices.e<int>(i);
      const int cCode = codes.e<int>(i);

      // we're skipping padded values: the code of a padding is -1
      if (cIndex < 0 || cCode < 0) continue;

      T *syn1Shifted = syn1 + static_cast<LongType>(cIndex) * vectorLength;
      hSoftmax_<T>(neu1, syn1Shifted, expTable, neu1e, alpha, vectorLength, cCode, expLength, infVector != nullptr,
                   stream);
    }

    auto nsStarter = ngStarter;
    auto irow = nsStarter;
    if (nsRounds > 0) {
      for (int r = 0; r < nsRounds + 1; r++) {
        if (r == 0) {
          // target is known in advance
        } else {
          randomValue = randomValue * (unsigned long long)25214903917 + 11;
          auto idx = sd::math::sd_abs<LongType,LongType>((randomValue >> 16) % negLength);
          irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

          if (irow < 0 || irow >= vocabSize) {
            irow = static_cast<int>(static_cast<unsigned long long>(randomValue) % (vocabSize - 1)) + 1;
          }
          if (irow == nsStarter) continue;
        }

        nSampling_<T>(neu1, syn1Neg + static_cast<LongType>(irow) * vectorLength, expTable, neu1e, alpha, vectorLength,
                      r == 0 ? 1 : 0, expLength, infVector != nullptr, stream);
      }
    }

    // if we don't train words - we skip start of idxSyn0
    int starter = trainWords == 1 ? 0 : contextWidth - numLabels;
    if (starter < 0) starter = 0;

    // propagate neu1e -> syn0
    if (infVector == nullptr) {
      fillUpSynonymsKernel<T><<<1, 1, 128, *stream>>>(starter, contextWidth, vectorLength, lockedWords, numLocked,
                                                      context, neu1e, syn0);
      sd::DebugHelper::checkErrorCode(stream, "fillUpSynonymsKernel failed");

    } else {
      // infVector and neu1e are both device pointers — must use a kernel, not a host loop
      dim3 w2vDims = getLaunchDims("word2vec");
      addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(infVector, neu1e, vectorLength);
      sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel (cbow infVector) failed");
    }

    alpha = ((alpha - minLearningRate) / (iterations - iteration)) + minLearningRate;
  }
  auto err = cudaStreamSynchronize(*stream);
  if (0 != err) {
    { std::string msg = "helpers::cbow_: Cannot synchronize stream after kernel executing; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }
}
BUILD_SINGLE_TEMPLATE( void cbow_,
                      (LaunchContext * lc, void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable,
                       void *vinfVector, int target, int ngStarter, int *context, int *lockedWords, NDArray &indices,
                       NDArray &codes, double alpha, sd::LongType randomValue, const int contextWidth,
                       const int validContext, const int numLocked, const int hsRounds, const int nsRounds,
                       const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                       const int numLabels, const bool trainWords, const int iterations,
                       const double minLearningRate),
                      SD_FLOAT_TYPES);

// The single round of cbow and cbow_inference: the window is the whole of context, a vector of INT32 words.
static void cbowSingleRound(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                            int target, int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords,
                            NDArray &indices, NDArray &codes, double alpha, LongType randomValue, int numLabels,
                            NDArray &inferenceVector, const bool trainWords, const int iterations,
                            const double minLearningRate) {
  auto xType = syn0.dataType();
  auto lc = context.getContext();
  const int vocabSize = syn0.sizeAt(0);

  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();

  // Everything that can be wrong with the arguments is found before the first kernel runs. The indices, the codes, the
  // negative table and the window are read on the host.
  NDArray::preparePrimaryUse({}, {&indices, &codes, &negTable, &context});

  if (nsRounds > 0 && (ngStarter < 0 || ngStarter >= vocabSize)) {
    THROW_EXCEPTION("CBOW: the word to sample negatives for is not a row of syn1Neg");
  }

  if (hsRounds > indices.lengthOf()) {
    THROW_EXCEPTION("CBOW: there are fewer indices than codes");
  }

  for (LongType h = 0; h < hsRounds; h++) {
    // we're skipping padded values: the code of a padding is -1
    if (indices.e<int>(h) < 0 || codes.e<int>(h) < 0) continue;

    if (indices.e<int>(h) >= vocabSize) THROW_EXCEPTION("Bad context 5");
  }

  // the values that pad the window are negative: the average is taken over the words that are there
  int validContext = 0;
  for (LongType c = 0; c < context.lengthOf(); c++) {
    const int word = context.e<int>(c);
    if (word >= vocabSize) THROW_EXCEPTION("Bad context 4");

    if (word >= 0) validContext++;
  }

  // Device kernels read and write syn0/syn1/syn1Neg and - for inference - the inference vector; they read expTable,
  // context and lockedWords.
  NDArray::prepareSpecialUse({&syn0, &syn1, &syn1Neg, &inferenceVector},
                             {&syn0, &syn1, &syn1Neg, &inferenceVector, &expTable, &context, &lockedWords});

  BUILD_SINGLE_SELECTOR(
      xType, cbow_,
      (lc, syn0.specialBuffer(), syn1.specialBuffer(), syn1Neg.specialBuffer(), expTable.specialBuffer(),
       negTable.buffer(), inferenceVector.specialBuffer(), target, ngStarter,
       reinterpret_cast<int *>(context.specialBuffer()), reinterpret_cast<int *>(lockedWords.specialBuffer()), indices,
       codes, alpha, randomValue, (int)context.lengthOf(), validContext, (int)lockedWords.lengthOf(), hsRounds,
       nsRounds, vocabSize, (int)syn0.sizeAt(1), (int)expTable.lengthOf(), (int)negTable.lengthOf(), numLabels,
       trainWords, iterations, minLearningRate),
      SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({}, {&indices, &codes, &negTable, &context});
  NDArray::registerSpecialUse({&syn0, &syn1, &syn1Neg, &inferenceVector}, {&expTable, &context, &lockedWords});
}


void cbowInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                   int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
                   double alpha, LongType randomValue, int numLabels, NDArray &inferenceVector, const bool trainWords,
                   int numWorkers,int iterations,double minLearningRate) {
  cbowSingleRound(syn0, syn1, syn1Neg, expTable, negTable, target, ngStarter, nsRounds, context, lockedWords, indices,
                  codes, alpha, randomValue, numLabels, inferenceVector, trainWords, iterations, minLearningRate);
}

template <typename T>
static SD_KERNEL SD_INLINE void buildCurrentWindowKernel(int vocabSize, int contextWidth, int vectorLength, int *bContext,
                                               T *syn0, T *neu1, int *actualContext, int e) {
  // building neu1 for current window
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int c = start; c < contextWidth; c += step) {
    // getting next context word
    auto cContext = bContext[c + (e * contextWidth)];

    // skipping padded values
    if (cContext < 0) continue;



    T *syn0word = syn0 + static_cast<LongType>(cContext) * vectorLength;

    for (int i = 0; i < vectorLength; i++) neu1[i] += syn0word[i];

    atomicAdd(actualContext, 1);
  }
}

template <typename T>
SD_KERNEL SD_INLINE void arrangeNeuKernel(int vectorLength, T *neu1, T *infVector, int *actualContext) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start; i<vectorLength && * actualContext> 0; i += step)
    neu1[i] /= (*actualContext + int(infVector != nullptr));
}

template <typename T>
SD_KERNEL SD_INLINE void applyShiftKernel(int *bContext, int *bLocker, T *syn0, T *neu1e, int contextWidth, int vectorLength,
                                int e, int starter) {
  auto step = blockDim.x * gridDim.x;
  auto start = blockDim.x * blockIdx.x + threadIdx.x;

  for (int c = starter + start; c < contextWidth; c += step) {
    // getting context
    auto cContext = bContext[c + (e * contextWidth)];
    // no word is locked when no locked words are given
    const int cLock = bLocker != nullptr ? bLocker[c + (e * contextWidth)] : 0;

    // skipping padded values
    if (cContext < 0 || cLock == 1) continue;


    // one word from context
    T *syn0word = syn0 + static_cast<LongType>(cContext) * vectorLength;

    for (int i = 0; i < vectorLength; i++) syn0word[i] += neu1e[i];
  }
}

// What can be wrong with the arguments of a batch is found before its first kernel runs: a failure after that would
// leave the tables half trained, and a word that is not a row of a table is an illegal access on the device. A window is a
// row of context; the words of the windows, the points of the hierarchic softmax and the words to sample negatives for
// have to be rows of the tables.
static void checkCbowBatch(NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters,
                           NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels,
                           const int nsRounds, const int vocabSize, const LongType numIndices) {
  const LongType numWindows = targets.lengthOf();
  const LongType contextWidth = context.sizeAt(1);

  if (numWindows > context.sizeAt(0)) {
    THROW_EXCEPTION("CBOW: every target needs a window");
  }

  if (lr.lengthOf() < numWindows) {
    THROW_EXCEPTION("CBOW: every window needs a learning rate");
  }

  if (!nLabels.isEmpty() && nLabels.lengthOf() < numWindows) {
    THROW_EXCEPTION("CBOW: every window needs a number of labels");
  }

  if (!lockedWords.isEmpty() && lockedWords.lengthOf() < numWindows * contextWidth) {
    THROW_EXCEPTION("CBOW: the locked words need a status for every word of the windows");
  }

  for (LongType t = 0; t < numWindows; t++) {
    for (LongType c = 0; c < contextWidth; c++) {
      // the values that pad a window are negative
      if (context.e<int>(t, c) >= vocabSize) THROW_EXCEPTION("Bad context 4");
    }
  }

  if (numIndices > 0) {
    if (codes.rankOf() != 2 || indices.sizeAt(0) < numWindows || codes.sizeAt(0) < numWindows ||
        codes.sizeAt(1) < numIndices) {
      THROW_EXCEPTION(
          "CBOW: the indices and codes of a batch need a row for every window, and the codes as many columns as the "
          "indices");
    }

    for (LongType t = 0; t < numWindows; t++) {
      for (LongType i = 0; i < numIndices; i++) {
        // we're skipping padded values: the code of a padding is -1
        if (indices.e<int>(t, i) < 0 || codes.e<int>(t, i) < 0) continue;

        if (indices.e<int>(t, i) >= vocabSize) THROW_EXCEPTION("Index can't be > vocab size");
      }
    }
  }

  if (nsRounds > 0 && !negStarters.isEmpty()) {
    if (negStarters.lengthOf() < numWindows || nextRandom.lengthOf() < numWindows) {
      THROW_EXCEPTION("CBOW: negative sampling needs a word to sample for and a random value with every window");
    }

    for (LongType t = 0; t < numWindows; t++) {
      const int ngStarter = negStarters.e<int>(t);
      if (ngStarter < 0 || ngStarter >= vocabSize) {
        THROW_EXCEPTION("CBOW: the word to sample negatives for is not a row of syn1Neg");
      }
    }
  }
}

/*
 * batched version of cbow routine
 *
 * Training: every window trains the words of its row of context (and syn1/syn1Neg), one window after the other; the
 * learning rate is the one of the window, and `iterations` does not apply.
 * Inference (an inference vector is given): the windows are visited in order and every one of them moves the inference
 * vector, which is all that learns; the inference vector is one more member of every window, the learning rate of every
 * window decays after each iteration, and its random values go on from one iteration to the next, as they do on the CPU.
 * */
template <typename T>
void cbowBatchExec_(LaunchContext *lc, NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable,
                    void *vinfVector, NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters,
                    NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels,
                    const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                    const int negLength, const bool trainWords, const int numThreads, const int iterations,
                    const double minLearningRate) {
  auto stream = lc->getCudaStream();
  // targets, indices, codes, negStarters, context, locked words, lr, nLabels and nextRandom are read on the host.
  NDArray::preparePrimaryUse({}, {&targets, &indices, &codes, &negStarters, &context, &lockedWords, &lr, &nLabels,
                                  &nextRandom});

  // a window is a row of context, and there is one for every target
  const auto numTargets = targets.lengthOf();
  const int contextWidth = context.sizeAt(1);

  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  const LongType numIndices = (indices.isEmpty() || codes.isEmpty() || s1.isEmpty()) ? 0 : indices.sizeAt(1);
  checkCbowBatch(context, lockedWords, targets, negStarters, indices, codes, lr, nextRandom, nLabels, nsRounds,
                 vocabSize, numIndices);

  // Device kernels read and write s0/s1/s1n; read context and lockedWords (via device pointers).
  NDArray::prepareSpecialUse({&s0, &s1, &s1n}, {&s0, &s1, &s1n, &context, &lockedWords});

  const auto syn0 = reinterpret_cast<T *>(s0.specialBuffer());      // bufferAsT<T>();
  const auto syn1 = reinterpret_cast<T *>(s1.specialBuffer());      // bufferAsT<T>();
  const auto syn1Neg = reinterpret_cast<T *>(s1n.specialBuffer());  // bufferAsT<T>();

  const auto expTable = reinterpret_cast<T *>(vexpTable);
  // the negative table is read on the host
  const auto negTable = reinterpret_cast<T *>(vnegTable);
  const auto infVector = reinterpret_cast<T *>(vinfVector);

  const auto dContext = reinterpret_cast<int *>(context.specialBuffer());
  const auto dLocker = reinterpret_cast<int *>(lockedWords.specialBuffer());
  std::vector<sd::LongType> neuShape = {vectorLength};
  std::vector<sd::LongType> oneShape = {1};
  NDArray neu1Arr('c', neuShape, DataTypeUtils::fromT<T>(), lc);
  NDArray neu1eArr('c', neuShape, DataTypeUtils::fromT<T>(), lc);
  NDArray actualContextArr('c', oneShape, DataTypeUtils::fromT<int>(), lc);
  T   *neu1          = reinterpret_cast<T*>(neu1Arr.specialBuffer());
  T   *neu1e         = reinterpret_cast<T*>(neu1eArr.specialBuffer());
  int *actualContext = reinterpret_cast<int*>(actualContextArr.specialBuffer());

  // the rates and the random values of the windows are read once: the windows of an inference vector are visited in every
  // iteration, and their rates decay and their random values go on
  const bool negativeSampling = nsRounds > 0 && !negStarters.isEmpty();
  std::vector<double> lrs(numTargets);
  std::vector<int> starters(negativeSampling ? numTargets : 0);
  std::vector<unsigned long long> randoms(negativeSampling ? numTargets : 0);
  for (LongType e = 0; e < numTargets; e++) {
    lrs[e] = lr.e<double>(e);
    if (negativeSampling) {
      starters[e] = negStarters.e<int>(e);
      randoms[e] = static_cast<unsigned long long>(nextRandom.e<LongType>(e));
    }
  }

  // only inference repeats its round
  const int passes = infVector != nullptr ? iterations : 1;
  for (int iteration = 0; iteration < passes; iteration++) {
    for (int e = 0; e < numTargets; e++) {
      auto alpha = lrs[e];
      auto numLabels = nLabels.isEmpty() ? 0 : nLabels.e<LongType>(e);

      // neu1 and neu1e belong to one window: both start from zero
      zeroVector_<T>(neu1, vectorLength, stream);
      zeroVector_<T>(neu1e, vectorLength, stream);
      zeroVector_<int>(actualContext, 1, stream);

      buildCurrentWindowKernel<T>
          <<<1, 1, 128, *stream>>>(vocabSize, contextWidth, vectorLength, dContext, syn0, neu1, actualContext, e);
      sd::DebugHelper::checkErrorCode(stream, "buildCurrentWindowKernel failed");

      // for inference the inference vector is one more member of the window
      if (infVector != nullptr) {
        dim3 w2vDims = getLaunchDims("word2vec");
        addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(neu1, infVector, vectorLength);
        sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel failed");
      }

      arrangeNeuKernel<T><<<1, 1, 128, *stream>>>(vectorLength, neu1, infVector, actualContext);
      sd::DebugHelper::checkErrorCode(stream, "arrangeNeuKernel failed");

      // hierarchic softmax step
      for (LongType i = 0; i < numIndices; i++) {
        const int cIndex = indices.e<int>(e, i);
        const int cCode = codes.e<int>(e, i);

        // we're skipping padded values: the code of a padding is -1
        if (cIndex < 0 || cCode < 0) continue;

        hSoftmax_<T>(neu1, syn1 + static_cast<LongType>(cIndex) * vectorLength, expTable, neu1e, alpha, vectorLength,
                     cCode, expLength, infVector != nullptr, stream);
      }

      // negative sampling step
      if (negativeSampling) {
        int irow = starters[e];
        const int nsStarter = irow;
        unsigned long long &randomValue = randoms[e];

        for (int r = 0; r < nsRounds + 1; r++) {
          // the target is known in advance: we're skipping rng on 0 step
          if (r != 0) {
            randomValue = randomValue * (unsigned long long)25214903917 + 11;
            auto idx = sd::math::sd_abs<LongType,LongType>((randomValue >> 16) % negLength);
            irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

            if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
            if (irow == nsStarter) continue;
          }

          nSampling_<T>(neu1, syn1Neg + static_cast<LongType>(irow) * vectorLength, expTable, neu1e, alpha,
                        vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr, stream);
        }
      }

      if (infVector == nullptr) {
        // if we're skipping labels
        int starter = trainWords == 1 ? 0 : contextWidth - numLabels;
        if (starter < 0) starter = 0;

        // applying previously averaged results
        applyShiftKernel<T><<<1, 1, 128, *stream>>>(dContext, dLocker, syn0, neu1e, contextWidth, vectorLength, e, starter);

        sd::DebugHelper::checkErrorCode(stream, "applyShiftKernel failed");
      } else {
        // inference: the inference vector is all that learns
        dim3 w2vDims = getLaunchDims("word2vec");
        addInfVectorKernel<T><<<w2vDims.x, w2vDims.y, w2vDims.z, *stream>>>(infVector, neu1e, vectorLength);
        sd::DebugHelper::checkErrorCode(stream, "addInfVectorKernel (cbow infVector) failed");
      }

    }

    // the learning rate decays after every iteration
    for (int e = 0; e < numTargets; e++) {
      lrs[e] = ((lrs[e] - minLearningRate) / (passes - iteration)) + minLearningRate;
    }
  }
  auto cerr = cudaStreamSynchronize(*stream);
  if (cerr) {
    { std::string msg = "Cannot syncronize stream before memory deallocation; Error code: [" + std::to_string(cerr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerPrimaryUse({}, {&targets, &indices, &codes, &negStarters, &context, &lockedWords, &lr, &nLabels,
                                   &nextRandom});
  NDArray::registerSpecialUse({&s0, &s1, &s1n}, {&context, &lockedWords});
}
BUILD_SINGLE_TEMPLATE( void cbowBatchExec_,
                      (LaunchContext * lc, NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable,
                       void *vinfVector, NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters,
                       NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels,
                       const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                       const int negLength, const bool trainWords, const int numThreads, const int iterations,
                       const double minLearningRate),
                      SD_FLOAT_TYPES);

void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target,
          NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
          NDArray &alpha, NDArray &randomValue, NDArray &numLabels, NDArray &inferenceVector, const bool trainWords,
          int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();
  auto lc = context.getContext();
  // target, ngStarter, alpha, randomValue, numLabels are read on the host.
  NDArray::preparePrimaryUse({}, {&target, &ngStarter, &alpha, &randomValue, &numLabels});
  if ((context.rankOf() == 0 || context.rankOf() == 1) && (indices.rankOf() == 1 || indices.rankOf() == 0)) {
    // single round case
    cbowSingleRound(syn0, syn1, syn1Neg, expTable, negTable, target.isEmpty() ? -1 : target.e<int>(0),
                    ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0), nsRounds, context, lockedWords, indices, codes,
                    alpha.isEmpty() ? 0.025 : alpha.e<double>(0),
                    randomValue.isEmpty() ? -1 : randomValue.e<sd::LongType>(0),
                    numLabels.isEmpty() ? 0 : numLabels.e<int>(0), inferenceVector, trainWords, iterations,
                    minLearningRate);
  } else if (context.rankOf() == 2 && (indices.rankOf() == 2 || indices.isEmpty())) {
    // batch mode; a configuration without hierarchic softmax passes no indices and no codes
    // Device kernels read and write syn0/syn1/syn1Neg and - for inference - the inference vector; they read expTable,
    // context and lockedWords. The negative table is read on the host.
    NDArray::prepareSpecialUse({&syn0, &syn1, &syn1Neg, &inferenceVector},
                               {&syn0, &syn1, &syn1Neg, &inferenceVector, &expTable, &context, &lockedWords});
    NDArray::preparePrimaryUse({}, {&negTable});

    BUILD_SINGLE_SELECTOR(
        xType, cbowBatchExec_,
        (lc, syn0, syn1, syn1Neg, expTable.specialBuffer(), negTable.buffer(), inferenceVector.specialBuffer(), context,
         lockedWords, target, ngStarter, indices, codes, alpha, randomValue, numLabels, nsRounds, syn0.sizeAt(0),
         syn0.sizeAt(1), expTable.lengthOf(), negTable.isEmpty() ? 0 : negTable.lengthOf(), trainWords, numWorkers,
         iterations, minLearningRate),
        SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({}, {&negTable});
    NDArray::registerSpecialUse({&syn0, &syn1, &syn1Neg, &inferenceVector}, {&expTable, &context, &lockedWords});
  } else
    THROW_EXCEPTION("CBOW: context must have rank 0/1 or 2");

  NDArray::registerPrimaryUse({}, {&target, &ngStarter, &alpha, &randomValue, &numLabels});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
