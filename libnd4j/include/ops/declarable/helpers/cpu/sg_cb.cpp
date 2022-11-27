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
#include <execution/Threads.h>
#include <ops/declarable/helpers/sg_cb.h>
#include <ops/specials.h>
#include <math/templatemath.h>
#define HS_MAX_EXP 6.0f

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
void hSoftmax_(void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code,
               int expLength, bool isInference) {
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

  if(!isInference) {
    for (int e = 0; e < vectorLength; e++) {
      syn1[e] = g * syn0[e] + syn1[e];
      neu1e[e] = g * syn1[e] + neu1e[e];
    }

  } else {
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] = g * syn1[e] + neu1e[e];
    }
  }
}

template <typename T>
void hSoftmaxDot_(T dot,void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code,
               int expLength, bool isInference) {
  //sd_printf("In hsoftmax with params alpha %f vector length %d code %d expLength %d isInference %d\n",alpha,vectorLength,code,expLength,isInference);
  // gradient
  if (dot < (T)-HS_MAX_EXP || dot >= (T) HS_MAX_EXP) return;
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);

  T g(0.0f);
  T f(0.0f);

  int idx = static_cast<int>((dot + HS_MAX_EXP) * ((float)expLength / HS_MAX_EXP / 2.0f));
  if (idx >= expLength || idx < 0) return;

  f = expTable[idx];
  g = (static_cast<T>(1.0f) - static_cast<T>(code) - f) * (T)alpha;
  PRAGMA_OMP_SIMD
  for (int x = 0; x < vectorLength; x++) {
    neu1e[x] = g * syn1[x] + neu1e[x];
  }

  // axpy2
  if (!isInference) {
    PRAGMA_OMP_SIMD
    for (int x = 0; x < vectorLength; x++) {
      syn1[x] = g * syn0[x] + syn1[x];
    }
  }

  }


template <typename T>
void nSampling_(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code,
                int expLength, bool isInference) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);

  T dot = (T)0.0f;
  T g = (T)0.0f;

   PRAGMA_OMP_SIMD
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


  // axpy2
  if (!isInference) {
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] = g * syn1Neg[e] + neu1e[e];
      syn1Neg[e] = g * syn0[e] + syn1Neg[e];
    }
  } else {
    // axpy1
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] = g * syn1Neg[e] + neu1e[e];
    }
  }

}

template <typename T>
void cbow_(void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target,
           int ngStarter, int *context, int *lockedWords, int *indices, int8_t *codes, double alpha,
           sd::LongType randomValue, const int contextWidth, const int hsRounds, const int nsRounds,
           const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels,
           const bool trainWords,double minLearningRate,const int iterations) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto negTable = reinterpret_cast<T *>(vnegTable);
  auto infVector = reinterpret_cast<T *>(vinfVector);

  auto neu1 = new T[vectorLength];
  auto neu1e = new T[vectorLength];
  memset(neu1, 0, vectorLength * sizeof(T));
  memset(neu1e, 0, vectorLength * sizeof(T));

  // building neu1 for current window

  for (int c = 0; c < contextWidth; c++) {

    T *syn0word = syn0 + (context[c] * vectorLength);
     PRAGMA_OMP_SIMD
    for (int i = 0; i < vectorLength; i++) {
      neu1[i] += syn0word[i];
    }
  }

  // for inference we add additional inference vector
  if(infVector != nullptr  && contextWidth > 0) {
     PRAGMA_OMP_SIMD
    for (int i = 0; i < vectorLength; i++) {
      neu1[i] = (infVector[i] + neu1[i]) / (contextWidth + 1);
    }
  } else if(infVector == nullptr && contextWidth > 0) {
     PRAGMA_OMP_SIMD
    for (int i = 0; i < vectorLength; i++) {
      neu1[i] = (infVector[i] + neu1[i]) / (contextWidth);
    }
  }

  // softmax round
  if (hsRounds > 0) {
    for (int i = 0; i < hsRounds; i++) {
      hSoftmax_<T>(neu1, syn1 + (indices[i] * vectorLength), expTable, neu1e, alpha, vectorLength, codes[i], expLength,
                   infVector != nullptr);
    }
  }

  auto nsStarter = ngStarter;
  auto irow = nsStarter;
  if (nsRounds > 0) {
   
    for (int r = 0; r < nsRounds + 1; r++) {
      if (r == 0) {
        // target is known in advance
      } else {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = sd::math::sd_abs<sd::LongType>((randomValue >> 16) % negLength);
        irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

        if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
        if (irow == nsStarter) continue;
      }

      nSampling_<T>(neu1, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0,
                    expLength, infVector != nullptr);
    }
  }

  // if we don't train words - we skip start of idxSyn0
  int starter = trainWords == 1 ? 0 : contextWidth - numLabels;

  // propagate neu1e -> syn0
  if (infVector == nullptr) {
   
    for (int c = starter; c < contextWidth; c++) {
      if (lockedWords[c] == 1) continue;

      T *syn0word = syn0 + (context[c] * vectorLength);
       PRAGMA_OMP_SIMD
      for (int i = 0; i < vectorLength; i++) {
        syn0word[i] += neu1e[i];
      }
    }
  } else {
     PRAGMA_OMP_SIMD
    for (int i = 0; i < vectorLength; i++) {
      infVector[i] += neu1e[i];
    }
  }

  delete[] neu1;
  delete[] neu1e;
}
BUILD_SINGLE_TEMPLATE(template void cbow_,
                      (void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector,
                          int target, int ngStarter, int *context, int *lockedWords, int *indices, int8_t *codes,
                          double alpha, sd::LongType randomValue, const int contextWidth, const int hsRounds,
                          const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                          const int negLength, const int numLabels, const bool trainWords,double minLearningRate,const int iterations),
                      SD_FLOAT_TYPES);

template <typename T>
void skipgram_(void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target,
               int ngStarter, int *indices, int8_t *codes, double alpha, sd::LongType randomValue, const int hsRounds,
               const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
               const int negLength,double minLearningRate,const int iterations) {

  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto negTable = reinterpret_cast<T *>(vnegTable);
  auto infVector = reinterpret_cast<T *>(vinfVector);

  auto neu1e = new T[vectorLength];
  memset(neu1e, 0, vectorLength * sizeof(T));

  for(int i = 0; i < iterations; i++) {
    // hierarchic softmax goes first (if enabled)
    auto syn0row = infVector != nullptr ? infVector : syn0 + (target * vectorLength);
    auto irow = 0;
    if (hsRounds > 0) {

      for (int r = 0; r < hsRounds; r++) {
        irow = indices[r];
        if (irow < 0 || irow >= vocabSize) break;

        hSoftmax_<T>(syn0row, syn1 + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, codes[r], expLength,
                     infVector != nullptr);
      }

    }

    // negative sampling goes second (if enabled)
    auto nsStarter = ngStarter;
    irow = nsStarter;
    if (nsRounds > 0) {

      for (int r = 0; r < nsRounds + 1; r++) {
        if (r == 0) {
          // target is known in advance
        } else {
          randomValue = randomValue * (unsigned long long)25214903917 + 11;
          auto idx = sd::math::sd_abs<sd::LongType>((randomValue >> 16) % negLength);
          irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

          if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
          if (irow == nsStarter) continue;
        }
        nSampling_<T>(syn0row, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0,
                      expLength, infVector != nullptr);

      }
    }

    if (infVector == nullptr) {
      for (int e = 0; e < vectorLength; e++) {
        syn0row[e] += neu1e[e];
      }
    } else {
      for (int e = 0; e < vectorLength; e++) {
        infVector[e] += neu1e[e];
      }

    }


  }

  delete[] neu1e;
}
BUILD_SINGLE_TEMPLATE(template void skipgram_,
                      (void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector,
                          int target, int ngStarter, int *indices, int8_t *codes, double alpha, sd::LongType randomValue,
                          const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength,
                          const int expLength, const int negLength,double minLearningRate,const int iterations),
                      SD_FLOAT_TYPES);

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
static void do_update(const int target, const int rowIndex, const int count, T *syn0, T *neu1t,
                      const int vectorLength) {
  auto syn0row = syn0 + (target * vectorLength);
  auto neu1e = neu1t + (rowIndex * vectorLength);
  for (int e = 0; e < vectorLength; e++) syn0row[e] += neu1e[e] / count;
}

template <typename T>
static void do_positive(const int target, const int postive, T *syn0, T *syn1Neg, T *expTable, T *neu1e,
                        const double alpha, const int vectorLength, const int expLength) {
  nSampling_<T>(syn0, syn1Neg, expTable, neu1e, alpha, vectorLength, 1, expLength, false);
}

template <typename T>
static void do_negative(int target, int positive, T *syn0, T *syn1Neg, T *expTable, T *negTable, T *neu1e,
                        int *sStarters, const double alpha, const unsigned long long rv, const int vocabSize,
                        const int vectorLength, const int expLength, const int negLength, const int nsRounds,
                        const int numThreads, const int numTargets) {
  int irow = 0;
  unsigned long long randomValue = rv;
  for (int r = 0; r < nsRounds; r++) {
    randomValue = sd::math::sd_abs<sd::LongType>(randomValue * (unsigned long long)25214903917 + 11);
    auto idx = sd::math::sd_abs<sd::LongType>((randomValue >> 16) % negLength);
    irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

    if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;

    if (irow == positive) continue;

    // we shift irow here to guarantee independence

    int dim = irow % numThreads;
    if (dim != omp_get_thread_num()) {
      irow += (numThreads - dim + omp_get_thread_num());

      // roll back to nearest affilated word
      while (irow >= vocabSize) irow -= numThreads;

      // if this row was processed as first step somewhere - skip it
      if (binarySearch(sStarters, irow, numTargets) > 0) {
        r--;
        continue;
      }
    }

    nSampling_<T>(syn0, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, 0, expLength, false);
  }
}

template <typename T>
T _dot(T *x,T *y,int vectorLength) {
  T dot(0.0f);
  // dot
  PRAGMA_OMP_SIMD
  for (int e = 0; e < vectorLength; e++) {
    dot += x[e] * y[e];
  }
  return dot;
}

template <typename T>
void skipgramBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector,
                        NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                        NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                        const int expLength, const int negLength, const bool preciseMode, const int numThreads,const int iterations,double minLearningRate) {
  const auto expTable = reinterpret_cast<T *>(vexpTable);
  const auto negTable = reinterpret_cast<T *>(vnegTable);
  const auto infVector = reinterpret_cast<T *>(vinfVector);

  const auto idxShift = indices.isEmpty() ? 0 : indices.sizeAt(1);
  const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);

  // regular mode provides 0 guarantees for reproducibility
  auto numTargets = targets.lengthOf();
  auto bTarget = targets.bufferAsT<int>();
  auto bIndices = indices.bufferAsT<int>();
  auto bCodes = codes.bufferAsT<int8_t>();
  int iteration = 0;
  auto func = PRAGMA_THREADS_FOR {
    T sneu1e[600];
    for (auto t = start; t < stop; t++) {
      T *neu1e = vectorLength <= 600 ? sneu1e : new T[vectorLength];
      memset(neu1e, 0, vectorLength * sizeof(T));
      auto alpha = lr.e<double>(t);
      alpha = ((alpha - minLearningRate) / (iterations - iteration)) + minLearningRate;
      sd::LongType randomValue = nextRandom.e<sd::LongType>(t);
       auto target = bTarget[t];

       auto syn0row = reinterpret_cast<T *>(s0.bufferWithOffset(target * vectorLength));
       if (hsRounds > 0) {
         auto cShift = t * idxShift;
         for (sd::LongType e = 0; e < hsRounds; e++) {
           int currRow = bIndices[e + cShift];
           if (currRow > 0 && currRow < vocabSize) {
             // sd_printf("Curr row %d\n",currRow);
             signed int code = bCodes[e + cShift];
             // sd_printf("Code was %d\n",code);
             T *syn1row = (T *) s1.bufferWithOffset(currRow * vectorLength);
             T dot = _dot(syn0row, syn1row, vectorLength);
             hSoftmaxDot_<T>(dot, syn0row, syn1row, expTable, neu1e, alpha, vectorLength, code, expLength, vinfVector != nullptr);
           }
         }
       }

       if (nsRounds > 0) {
         int irow = negStarters.e<int>(t);
         int nsStarter = irow;
         for (int r = 0; r < nsRounds + 1; r++) {
           if (r == 0) {
             // target is known in advance
           } else {
             randomValue = randomValue * (unsigned long long)25214903917 + 11;
             auto idx = sd::math::sd_abs<sd::LongType>((randomValue >> 16) % negLength);
             irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

             if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;

             if (irow == nsStarter) continue;
           }

           nSampling_<T>(syn0row, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength,
                         r == 0 ? 1 : 0, expLength, infVector != nullptr);
         }
       }


       if (infVector == nullptr) {
         for (int e = 0; e < vectorLength; e++) {
           syn0row[e] += neu1e[e];
         }
       } else {
         for (int e = 0; e < vectorLength; e++) {
           infVector[e] += neu1e[e];
         }
     }

        // optionally release temp arrays
        if (vectorLength > 600) {
          delete[] neu1e;

        }
    }
    };

    for(int i = 0; i < iterations; i++) {
      iteration = i;
      samediff::Threads::parallel_tad(func, 0, numTargets, 1, numThreads);
    }

  }

BUILD_SINGLE_TEMPLATE(template void skipgramBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector,
                          NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                          NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                          const int expLength, const int negLength, const bool preciseMode, const int numThreads,const int iterations,double minLearningRate),
                      SD_FLOAT_TYPES);

template <typename T>
void cbowBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector,
                    NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices,
                    NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds,
                    const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                    const bool trainWords, const int numThreads,double minLearningRate,int iterations) {
  const auto syn0 = s0.bufferAsT<T>();
  const auto syn1 = s1.bufferAsT<T>();
  const auto syn1Neg = s1n.bufferAsT<T>();

  const auto expTable = reinterpret_cast<T *>(vexpTable);
  const auto negTable = reinterpret_cast<T *>(vnegTable);
  const auto infVector = reinterpret_cast<T *>(vinfVector);

  const auto idxShift = indices.isEmpty() ? 0 : indices.sizeAt(1);
  const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);
  const auto numTargets = context.sizeAt(0);
  const int contextWidth = context.sizeAt(1);

  const auto bContext = context.bufferAsT<int>();
  const auto bLocker = lockedWords.bufferAsT<int>();
  const auto bIndices = indices.bufferAsT<int>();
  const auto bCodes = codes.bufferAsT<int8_t>();
  const auto bStarters = negStarters.bufferAsT<int>();
  const auto numIndices = indices.isEmpty() ? 0 : indices.sizeAt(1);

  int iteration = 0;
  auto func = PRAGMA_THREADS_FOR {
    T sneu1[600];
    T sneu1e[600];

    for (auto e = start; e < stop; e++) {
      T *neu1 = vectorLength <= 600 ? sneu1 : new T[vectorLength];
      T *neu1e = vectorLength <= 600 ? sneu1e : new T[vectorLength];

      // optionally we nullify temp arrays after successful (and on first) cycle
      memset(neu1, 0, sizeof(T) * vectorLength);
      memset(neu1e, 0, sizeof(T) * vectorLength);

      auto alpha = lr.e<double>(e);
      alpha = ((alpha - minLearningRate) / (iterations - iteration)) + minLearningRate;

      auto numLabels = nLabels.isEmpty() ? 0 : nLabels.e<int>(e);

      int actualContext = 0;

      // building neu1 for current window
      for (int c = 0; c < contextWidth; c++) {
        // getting next context word
        auto cContext = bContext[c + (e * contextWidth)];

        // skipping padded values
        if (cContext < 0) continue;

        if (cContext >= vocabSize) throw std::runtime_error("ContextID can't be >= vocab size");

        T *syn0word = syn0 + (cContext * vectorLength);

        for (int i = 0; i < vectorLength; i++) neu1[i] += syn0word[i];

        actualContext++;
      }

      if (infVector != nullptr) actualContext++;

      if (actualContext > 1) {
        for (int i = 0; i < vectorLength; i++) neu1[i] /= actualContext;
      }

      // hierarchic softmax step
      if (!indices.isEmpty()) {
        for (sd::LongType i = 0; i < numIndices; i++) {
          const int cIndex = bIndices[(e * numIndices) + i];
          const int cCode = bCodes[(e * numIndices) + i];

          // we're skipping padded values
          if (cIndex < 0) continue;

          if (cIndex >= vocabSize) throw std::runtime_error("Index can't be > vocab size");

          hSoftmax_<T>(neu1, syn1 + (cIndex * vectorLength), expTable, neu1e, alpha, vectorLength, cCode, expLength,
                       false);
        }
      }

      // negative sampling step
      if (!negStarters.isEmpty() && nsRounds > 0) {
        int irow = bStarters[e];
        const int nsStarter = irow;
        unsigned long long randomValue = nextRandom.e<sd::LongType>(e);

        for (int r = 0; r < nsRounds + 1; r++) {
          // we're skipping rng on 0 step
          if (r != 0) {
            randomValue = randomValue * (unsigned long long)25214903917 + 11;
            auto idx = sd::math::sd_abs<sd::LongType>((randomValue >> 16) % negLength);
            irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

            if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
            if (irow == nsStarter) continue;

            nSampling_<T>(neu1, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength,
                          r == 0 ? 1 : 0, expLength, infVector != nullptr);
          } else {
            nSampling_<T>(neu1, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength,
                          r == 0 ? 1 : 0, expLength, infVector != nullptr);
          }

        }
      }

      // if we're skipping labels
      int starter = trainWords == 1 ? 0 : contextWidth - numLabels;

      // applying previously averaged results
      for (int c = starter; c < contextWidth; c++) {
        // getting context
        auto cContext = bContext[c + (e * contextWidth)];
        auto cLock = bLocker[c + (e * contextWidth)];

        // skipping padded values
        if (cContext < 0 || cLock == 1) continue;

        if (cContext >= vocabSize) throw std::runtime_error("ContextID can't be > vocab size");

        // one word from context
        T *syn0word = syn0 + (cContext * vectorLength);

        for (int i = 0; i < vectorLength; i++) syn0word[i] += neu1e[i];
      }

      // optionally release temp arrays
      if (vectorLength > 600) {
        delete[] neu1;
        delete[] neu1e;
      }
    }
  };

  for(int i = 0; i < iterations; i++) {
    iteration = i;
    samediff::Threads::parallel_tad(func, 0, numTargets, 1, numThreads);
  }

}
BUILD_SINGLE_TEMPLATE(template void cbowBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector,
                          NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices,
                          NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds,
                          const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                          const bool trainWords, const int numThreads,double minLearningRate,const int iterations),
                      SD_FLOAT_TYPES);



void skipgramInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
             int ngStarter, int nsRounds, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
              NDArray &inferenceVector, const bool preciseMode, const int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();
  auto hsRounds = codes.lengthOf();

  BUILD_SINGLE_SELECTOR(
      xType, skipgram_,
      (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
        target, ngStarter,
       reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t *>(codes.buffer()), alpha,
       randomValue, hsRounds, nsRounds, (int)syn0.sizeAt(0), (int)syn0.sizeAt(1),
       (int)expTable.lengthOf(), (int)negTable.lengthOf(),minLearningRate,iterations),
      SD_FLOAT_TYPES);
}


void cbowInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
          int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
          double alpha, sd::LongType randomValue, int numLabels, NDArray &inferenceVector, const bool trainWords,
          int numWorkers,int iterations,double minLearningRate) {
  auto xType = syn0.dataType();
  auto hsRounds = codes.lengthOf();

  BUILD_SINGLE_SELECTOR(
      xType, cbow_,
      (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
      target, ngStarter,
       reinterpret_cast<int *>(context.buffer()), reinterpret_cast<int *>(lockedWords.buffer()),
       reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t *>(codes.buffer()), alpha,
       randomValue, (int)context.lengthOf(), hsRounds, nsRounds, (int)syn0.sizeAt(0),
       (int)syn0.sizeAt(1), (int)expTable.lengthOf(), (int)negTable.lengthOf(),
       numLabels, trainWords,minLearningRate,iterations),
      SD_FLOAT_TYPES);
}

void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target,
              NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue,
              NDArray &inferenceVector, const bool preciseMode, const int numWorkers,const int iterations,double minLearningRate) {
  auto xType = syn0.dataType();

  // single round case
  if ((ngStarter.isScalar() && !ngStarter.isEmpty()) || (target.isScalar() && !target.isEmpty())) {
    auto hsRounds = codes.lengthOf();

    BUILD_SINGLE_SELECTOR(
        xType, skipgram_,
        (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
            target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0),
            reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t *>(codes.buffer()), alpha.e<double>(0),
            randomValue.e<sd::LongType>(0), hsRounds, nsRounds, (int)syn0.sizeAt(0), (int)syn0.sizeAt(1),
            (int)expTable.lengthOf(), (int)negTable.lengthOf(),minLearningRate,iterations),
        SD_FLOAT_TYPES);
  } else if (ngStarter.isVector() || target.isVector()) {
    // batch mode
    BUILD_SINGLE_SELECTOR(xType, skipgramBatchExec_,
                          (syn0, syn1, syn1Neg, expTable.buffer(), negTable.buffer(), inferenceVector.buffer(), target, ngStarter,
                              indices, codes, alpha, randomValue, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1),
                              expTable.lengthOf(), negTable.lengthOf(), preciseMode, numWorkers,iterations,minLearningRate),
                          SD_FLOAT_TYPES);
  } else
    throw std::runtime_error("SkipGram: target must have rank 0 or 1");
}

void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target,
          NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
          NDArray &alpha, NDArray &randomValue, NDArray &numLabels, NDArray &inferenceVector, const bool trainWords,
          int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();

  if ((context.rankOf() == 0 || context.rankOf() == 1) && (indices.rankOf() == 1 || indices.rankOf() == 0)) {
    auto hsRounds = codes.lengthOf();

    BUILD_SINGLE_SELECTOR(
        xType, cbow_,
        (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
            target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0),
            reinterpret_cast<int *>(context.buffer()), reinterpret_cast<int *>(lockedWords.buffer()),
            reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t *>(codes.buffer()), alpha.e<double>(0),
            randomValue.e<sd::LongType>(0), (int)context.lengthOf(), hsRounds, nsRounds, (int)syn0.sizeAt(0),
            (int)syn0.sizeAt(1), (int)expTable.lengthOf(), (int)negTable.lengthOf(),
            numLabels.isEmpty() ? 0 : numLabels.e<int>(0), trainWords,minLearningRate,iterations),
        SD_FLOAT_TYPES);
  } else if (context.rankOf() == 2 && indices.rankOf() == 2) {
    BUILD_SINGLE_SELECTOR(
        xType, cbowBatchExec_,
        (syn0, syn1, syn1Neg, expTable.buffer(), negTable.buffer(), nullptr, context, lockedWords, target, ngStarter,
            indices, codes, alpha, randomValue, numLabels, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1), expTable.lengthOf(),
            negTable.isEmpty() ? 0 : negTable.lengthOf(), trainWords, numWorkers,minLearningRate,iterations),
        SD_FLOAT_TYPES);
  } else
    throw std::runtime_error("CBOW: context must have rank 0/1 or 2");
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
