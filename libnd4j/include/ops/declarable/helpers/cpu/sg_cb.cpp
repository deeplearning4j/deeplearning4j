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
#include <math/templatemath.h>
#define HS_MAX_EXP 6.0f
#include <cstddef>
#include <cstdlib>
#include <new>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
void hSoftmax_(T *vsyn0, T *vsyn1, T *vexpTable, T *vneu1e, const double alpha, const int vectorLength, const int code,

               const int expLength, const bool isInference) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);

  T dot(0.0f);
  T g(0.0f);
  T f(0.0f);


  // dot
  PRAGMA_OMP_SIMD
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
    PRAGMA_OMP_SIMD
    for (int x = 0; x < vectorLength; x++) {
      neu1e[x] += g * syn1[x];
      syn1[x] += g * syn0[x];

    }
  } else {
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] = g * syn1[e] + neu1e[e];
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
      neu1e[e] += g * syn1Neg[e];
      syn1Neg[e] += g * syn0[e];
    }


  } else {
    // axpy1
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] += g * syn1Neg[e];
    }


  }

}

template <typename T>
void cbow_(NDArray &vsyn0, NDArray &vsyn1, NDArray &vsyn1Neg, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector, int target,
           int ngStarter, int *context, int *lockedWords, int *indices, int *codes, double alpha,
           sd::LongType randomValue, const int contextWidth, const int hsRounds, const int nsRounds,
           const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels,
           const bool trainWords,double minLearningRate,const int iterations) {
  auto syn0 = reinterpret_cast<T *>(vsyn0.bufferAsT<T>());
  auto syn1 = reinterpret_cast<T *>(vsyn1.bufferAsT<T>());
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg.bufferAsT<T>());
  auto expTable = reinterpret_cast<T *>(vexpTable.bufferAsT<T>());
  auto negTable = reinterpret_cast<T *>(vnegTable.bufferAsT<T>());
  auto infVector = reinterpret_cast<T *>(vinfVector.bufferAsT<T>());
  auto neu1 = new T[vectorLength];
  auto neu1e = new T[vectorLength];
  memset(neu1, 0, vectorLength * sizeof(T));
  memset(neu1e, 0, vectorLength * sizeof(T));

  for(int i = 0; i < iterations; i++) {
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

        auto syn1NegRow = syn1Neg + (irow * vectorLength);
        nSampling_<T>(neu1, syn1NegRow, expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0,
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

    alpha = ((alpha - static_cast<double>(minLearningRate)) / static_cast<double>((iterations - i))) + static_cast<double>(minLearningRate);


  }
}
BUILD_SINGLE_TEMPLATE(template void cbow_,
                      (NDArray &syn0, NDArray &syn1,NDArray &syn1Neg, NDArray &expTable, NDArray &vnegTable, NDArray &vinfVector,
                          int target, int ngStarter, int *context, int *lockedWords, int *indices, int *codes,
                          double alpha, sd::LongType randomValue, const int contextWidth, const int hsRounds,
                          const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                          const int negLength, const int numLabels, const bool trainWords,double minLearningRate,const int iterations),
                      SD_NATIVE_FLOAT_TYPES);

template <typename T>
void skipgram_(void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target,
               int ngStarter, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue, const int hsRounds,
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
  PRAGMA_OMP_SIMD
  for(int i = 0; i < iterations; i++) {
    // hierarchic softmax goes first (if enabled)
    auto syn0row = infVector != nullptr ? infVector : syn0 + (target * vectorLength);
    alpha = ((alpha - minLearningRate) / (iterations - i)) + minLearningRate;

    auto irow = 0;
    if (hsRounds > 0) {
      for (int r = 0; r < hsRounds; r++) {
        irow = indices.e<int>(r);
        hSoftmax_<T>(syn0row, syn1 + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, codes.e<int>(r), expLength,
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

    alpha = ((alpha - static_cast<double>(minLearningRate)) / static_cast<double>((iterations - i))) + static_cast<double>(minLearningRate);
  }

  delete[] neu1e;
}

BUILD_SINGLE_TEMPLATE(template void skipgram_,
                      (void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector,
                          int target, int ngStarter,NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
                          const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength,
                          const int expLength, const int negLength,double minLearningRate,const int iterations),
                      SD_NATIVE_FLOAT_TYPES);

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
void doSkipGramLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vinfVector, const NDArray &targets,
                     const NDArray &negStarters, const NDArray &indices, const NDArray &codes, const NDArray &lr,
                     const NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                     const int expLength, const int negLength, T *const expTable, const T *negTable,
                     const LongType hsRounds, int t);

template <typename T>
void doSkipGramInferenceLoop_(NDArray &s1, NDArray &s1n, T *syn0row, const NDArray &targets,
                              const NDArray &negStarters, const NDArray &indices, const NDArray &codes,
                              const double lr, const NDArray &nextRandom, const int nsRounds, const int vocabSize,
                              const int vectorLength, const int expLength, const int negLength, T *const expTable,
                              const T *negTable, const LongType hsRounds, int t, T *neu1e);

//used for lifecycle tracking in thread locals for error accumulation
template <typename T>
class BufferHolder {
 public:
  BufferHolder(const int vectorLength) {
    neu1e = new T[vectorLength];
  }
  T *neu1e;
  ~BufferHolder() {
    delete[] neu1e;
  }

};


#include <cstdlib>

template <typename T, std::size_t Alignment>
class AlignedAllocator
{
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  template <typename U>
  struct rebind { typedef AlignedAllocator<U, Alignment> other; };

  AlignedAllocator() {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }

  pointer allocate(size_type n, const void* = nullptr)
  {
#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
    void* ptr = nullptr;
    if(posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
      ptr = nullptr;
#endif
    if (!ptr)
      throw std::bad_alloc();
    return static_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type)
  {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }

  size_type max_size() const
  {
    return static_cast<size_type>(-1) / sizeof(T);
  }

  void construct(pointer p, const value_type& x)
  {
    ::new(p) value_type(x);
  }

  void destroy(pointer p)
  {
    p->~value_type();
  }
};


template <typename T>
void skipgramBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable,NDArray &vnegTable, NDArray &vinfVector,
                        NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                        NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                        const int expLength, const int negLength, const bool preciseMode, const int numThreads,const int iterations,double minLearningRate) {
  const auto expTable = reinterpret_cast<T *>(vexpTable.buffer());
  const auto negTable = reinterpret_cast<T *>(vnegTable.buffer());
  const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);
  //training
  if(vinfVector.isEmpty()) {
    const sd::LongType  targetsLen = targets.lengthOf();

    auto func = PRAGMA_THREADS_FOR {
      for (auto t = start; t < stop; t+= increment) {
        doSkipGramLoop_(s0, s1, s1n, vinfVector, targets, negStarters, indices, codes, lr, nextRandom, nsRounds,
                        vocabSize, vectorLength, expLength, negLength, expTable, negTable, hsRounds, t);
      }
    };


    int chunkSize = 1024;

    if(targetsLen < chunkSize) {
      samediff::Threads::parallel_tad(func,0,targetsLen,1);
    } else {
      int chunks = targetsLen / chunkSize;
      for(int i = 0; i < chunks; i++) {
        int start = i * chunkSize;
        int potentialEnd = start + chunkSize;
        int end = sd::math::sd_min<int>(targetsLen,potentialEnd);
        samediff::Threads::parallel_tad(func,start,end,1);
      }

    }



  } else { //inference
    auto numTargets = targets.lengthOf();
    auto vec = reinterpret_cast<T *>(vinfVector.buffer());

    T **neu1e = new T*[numTargets];
    for(int i = 0; i < numTargets; i++) {
      neu1e[i] = new T[vectorLength];
    }

    for(int curr = 0; curr < iterations; curr++) {
      std::vector<double> lrs(numTargets);
      for(int t = 0; t < numTargets; t++) {
        lrs[t] = ((lr.e<double>(t) - static_cast<double>(minLearningRate)) / (static_cast<double>(iterations - curr))) + static_cast<double>(minLearningRate);
      }

#pragma omp parallel for num_threads(numThreads) schedule(guided)
      for(int t = 0; t < numTargets; t++) {
        auto currNeu1e = neu1e[t];
        std::fill_n(currNeu1e, vectorLength, T(0));
        double currentLr = lrs[t];

        doSkipGramInferenceLoop_(s1,s1n,vec,
                                 targets,negStarters,
                                 indices,
                                 codes,
                                 currentLr,
                                 nextRandom,
                                 nsRounds,
                                 vocabSize,
                                 vectorLength,
                                 expLength,
                                 negLength,
                                 expTable,
                                 negTable,
                                 hsRounds,
                                 t,
                                 currNeu1e);
      }

      std::vector<std::vector<T> > buffer(numThreads, std::vector<T>(vectorLength));

#pragma omp parallel num_threads(numThreads)
      {
        int threadId = omp_get_thread_num();
        auto& vec_local = buffer[threadId];

#pragma omp for schedule(dynamic)
        for(int i = 0; i < numTargets; i++) {
          for(int j = 0; j < vectorLength; j++) {
            vec_local[j] += neu1e[i][j];
          }
        }
      }

      for(int j = 0; j < vectorLength; j++) {
        for(const auto& vec_local : buffer) {
          vec[j] += vec_local[j];
        }
      }
    }

    for(int i = 0; i < numTargets; i++) {
      delete[] neu1e[i];
    }
    delete[] neu1e;

  }// end else
}




template <typename T>
void doSkipGramInferenceLoop_(NDArray &s1, NDArray &s1n, T *syn0row, const NDArray &targets,
                              const NDArray &negStarters, const NDArray &indices, const NDArray &codes,
                              const double alpha, const NDArray &nextRandom, const int nsRounds, const int vocabSize,
                              const int vectorLength, const int expLength, const int negLength, T *const expTable,
                              const T *negTable, const LongType hsRounds, int t, T *neu1e) {

  LongType randomValue = nextRandom.e<LongType>(t);
  auto target = targets.e<int>(t);

  std::vector<int> currRows(hsRounds);
  std::vector<int> codes_vals(hsRounds);
  std::vector<int> irows(nsRounds+1, negStarters.e<int>(t));

#pragma omp parallel
  {
#pragma omp for nowait
    for (LongType e = 0; e < hsRounds; e++) {
      currRows[e] = indices.e<int>(t,e);
      codes_vals[e] = codes.e<int>(t,e);
    }

#pragma omp for nowait
    for (int r = 1; r < nsRounds + 1; r++) {
      randomValue = randomValue * (unsigned long long)25214903917 + 11;
      auto idx = math::sd_abs<LongType>((randomValue >> 16) % negLength);
      irows[r] = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

      if (irows[r] < 0 || irows[r] >= vocabSize) irows[r] = randomValue % (vocabSize - 1) + 1;
    }
  }

#pragma omp parallel for
  for (LongType e = 0; e < hsRounds; e++) {
    if(codes_vals[e] < 0) {
      continue;
    }

    T *syn1row = (T *) s1.bufferWithOffset(currRows[e] * vectorLength);
    hSoftmax_<T>(syn0row,syn1row,expTable,neu1e,alpha,vectorLength,codes_vals[e],expLength,true);
  }

  int nsStarter = irows[0];
  for (int r = 0; r < nsRounds + 1; r++) {
    if (r != 0 && irows[r] == nsStarter) continue;

    nSampling_<T>(syn0row, s1n.bufferWithOffset(irows[r] * vectorLength), expTable, neu1e, alpha, vectorLength,
                  r == 0 ? 1 : 0, expLength, true);
  }
}


template <typename T>
void doSkipGramLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vinfVector, const NDArray &targets,
                     const NDArray &negStarters, const NDArray &indices, const NDArray &codes, const NDArray &lr,
                     const NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                     const int expLength, const int negLength, T *const expTable, const T *negTable,
                     const LongType hsRounds, int t) {
  T *neu1e = new T[vectorLength];
  memset(neu1e, 0, vectorLength * sizeof(T));

  auto alpha = lr.e<double>(t);

  LongType randomValue = nextRandom.e<LongType>(t);
  auto target = targets.e<int>(t);
  auto syn0row = vinfVector.isEmpty() ?  reinterpret_cast<T *>(s0.bufferWithOffset(target * vectorLength)) : reinterpret_cast<T *>(vinfVector.buffer());
  if(hsRounds > 0) {
    for (LongType e = 0; e < hsRounds; e++) {
      int currRow = indices.e<int>(t,e);
      int code = codes.e<int>(t,e);
      //codes are only 0 and 1, -1 are placeholders for invalid codes
      //the codes matrix is padded with extra values at time of allocation
      //this is due to the code rows effectively being a ragged matrix (rows have different shapes)
      if(code < 0)  {
        continue;
      }

      T *syn1row = (T *) s1.bufferWithOffset(currRow * vectorLength);
      hSoftmax_<T>(syn0row,syn1row,expTable,neu1e,lr.e<double>(t),vectorLength,code,expLength,!vinfVector.isEmpty());

    }
  }

  if(nsRounds > 0) {
    int irow = negStarters.e<int>(t);
    int nsStarter = irow;
    for (int r = 0; r < nsRounds + 1; r++) {
      if (r == 0) {
        // target is known in advance
      } else {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = math::sd_abs<LongType>((randomValue >> 16) % negLength);
        irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

        if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;

        if (irow == nsStarter) continue;
      }

      nSampling_<T>(syn0row, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength,
                    r == 0 ? 1 : 0, expLength, !vinfVector.isEmpty());
    }
  }
      PRAGMA_OMP_SIMD
  for (int e = 0; e < vectorLength; e++) {
    syn0row[e] += neu1e[e];
  }
  delete[] neu1e;
}

BUILD_SINGLE_TEMPLATE(template void skipgramBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector,
                          NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                          NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                          const int expLength, const int negLength, const bool preciseMode, const int numThreads,const int iterations,double minLearningRate),
                      SD_NATIVE_FLOAT_TYPES);

template <typename T>
void doCbowLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, const NDArray &negStarters, const NDArray &indices,
                 const NDArray &codes, const NDArray &lr, const NDArray &nextRandom, const NDArray &nLabels,
                 const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                 const int negLength, const bool trainWords, T *const expTable, const T *negTable, const T *infVector,
                 const int contextWidth, const int *bContext, const int *bLocker, const int *bStarters,
                 const LongType numIndices, int t);
template <typename T>
void cbowBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector,
                    NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices,
                    NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds,
                    const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                    const bool trainWords, const int numThreads,double minLearningRate,int iterations) {

  const auto syn1Neg = s1n.bufferAsT<T>();

  const auto expTable = vexpTable.bufferAsT<T>();
  const auto negTable = vnegTable.bufferAsT<T>();
  const auto infVector = vinfVector.bufferAsT<T>();

  const auto idxShift = indices.isEmpty() ? 0 : indices.sizeAt(1);
  const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);
  const auto numTargets = context.sizeAt(0);
  const int contextWidth = context.sizeAt(1);

  const auto bContext = context.bufferAsT<int>();
  const auto bLocker = lockedWords.bufferAsT<int>();
  const auto bStarters = negStarters.bufferAsT<int>();
  const auto numIndices = indices.isEmpty() ? 0 : indices.sizeAt(1);
  if(vinfVector.isEmpty()) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto t = start; t < stop; t+= increment) {
        doCbowLoop_(s0, s1, s1n, negStarters, indices, codes, lr, nextRandom, nLabels, nsRounds, vocabSize,
                    vectorLength, expLength, negLength, trainWords, expTable, negTable, infVector, contextWidth,
                    bContext, bLocker, bStarters, numIndices, t);

      }
    };


    int targetsLen = targets.lengthOf();
    int chunkSize = 1024;
    if(targetsLen < chunkSize) {
      samediff::Threads::parallel_tad(func,0,targetsLen,1);
    } else {
      int chunks = targetsLen / chunkSize;
      for(int i = 0; i < chunks; i++) {
        int start = i * chunkSize;
        int potentialEnd = start + chunkSize;
        int end = sd::math::sd_min<int>(targetsLen,potentialEnd);
        samediff::Threads::parallel_tad(func,start,end,1);
      }
    }




  } else {
    // regular mode provides 0 guarantees for reproducibility
    auto numTargets = targets.lengthOf();
    for(int iteration = 0; iteration < iterations; iteration++) {
      for (auto t = 0; t < numTargets; t++) {
        doCbowLoop_(s0, s1, s1n, negStarters, indices, codes, lr, nextRandom, nLabels, nsRounds, vocabSize,
                    vectorLength, expLength, negLength, trainWords, expTable, negTable, infVector, contextWidth,
                    bContext, bLocker, bStarters, numIndices, t);
      }
    }
  }


}
template <typename T>
void doCbowLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, const NDArray &negStarters, const NDArray &indices,
                 const NDArray &codes, const NDArray &lr, const NDArray &nextRandom, const NDArray &nLabels,
                 const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                 const int negLength, const bool trainWords, T *const expTable, const T *negTable, const T *infVector,
                 const int contextWidth, const int *bContext, const int *bLocker, const int *bStarters,
                 const LongType numIndices, int t) {
  T *neu1 =  new T[vectorLength];
  T *neu1e =  new T[vectorLength];

  // optionally we nullify temp arrays after successful (and on first) cycle
  memset(neu1, 0, sizeof(T) * vectorLength);
  memset(neu1e, 0, sizeof(T) * vectorLength);

  auto alpha = lr.e<double>(t);

  auto numLabels = nLabels.isEmpty() ? 0 : nLabels.e<int>(t);

  int actualContext = 0;

  // building neu1 for current window
  for (int c = 0; c < contextWidth; c++) {
    // getting next context word
    auto cContext = bContext[c + (t * contextWidth)];

    // skipping padded values
    if (cContext < 0) continue;

    if (cContext >= vocabSize) THROW_EXCEPTION("ContextID can't be >= vocab size");

    T *syn0word = (T *) s0.bufferWithOffset(cContext * vectorLength);

    for (int i = 0; i < vectorLength; i++) neu1[i] += syn0word[i];

    actualContext++;
  }

  if (infVector != nullptr) actualContext++;

  if (actualContext > 1) {
    for (int i = 0; i < vectorLength; i++) neu1[i] /= actualContext;
  }

  // hierarchic softmax step
  if (!indices.isEmpty()) {
    for (LongType i = 0; i < numIndices; i++) {
      const int cIndex = indices.e<int>(t,i);
      const int cCode = codes.e<int>(t,i);

      // we're skipping padded values
      if (cIndex < 0) continue;

      if (cIndex >= vocabSize) THROW_EXCEPTION("Index can't be > vocab size");

      hSoftmax_<T>(neu1, s1.bufferasTWithOffset<T>(cIndex * vectorLength), expTable, neu1e, alpha, vectorLength, cCode, expLength,
                   false);
    }
  }

  // negative sampling step
  if (!negStarters.isEmpty() && nsRounds > 0) {
    int irow = bStarters[t];
    const int nsStarter = irow;
    unsigned long long randomValue = nextRandom.e<LongType>(t);

    for (int r = 0; r < nsRounds + 1; r++) {
      // we're skipping rng on 0 step
      if (r != 0) {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = math::sd_abs<LongType>((randomValue >> 16) % negLength);
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
    auto cContext = bContext[c + (t * contextWidth)];
    auto cLock = bLocker[c + (t * contextWidth)];

    // skipping padded values
    if (cContext < 0 || cLock == 1) continue;

    if (cContext >= vocabSize) THROW_EXCEPTION("ContextID can't be > vocab size");

    // one word from context
    T *syn0word = (T *) s0.bufferWithOffset(cContext * vectorLength);
    PRAGMA_OMP_SIMD
    for (int i = 0; i < vectorLength; i++) syn0word[i] += neu1e[i];
  }

  // optionally release temp arrays
  if (vectorLength > 600) {
    delete[] neu1;
    delete[] neu1e;
  }
}
BUILD_SINGLE_TEMPLATE(template void cbowBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector,
                          NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices,
                          NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds,
                          const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                          const bool trainWords, const int numThreads,double minLearningRate,const int iterations),
                      SD_NATIVE_FLOAT_TYPES);



void skipgramInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                       int ngStarter, int nsRounds, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
                       NDArray &inferenceVector, const bool preciseMode, const int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();
  auto hsRounds = codes.lengthOf();
  BUILD_SINGLE_SELECTOR(
      xType, skipgram_,
      (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
          target, ngStarter,
          indices, codes, alpha,
          randomValue, hsRounds, nsRounds, (int)syn0.sizeAt(0), (int)syn0.sizeAt(1),
          (int)expTable.lengthOf(), (int)negTable.lengthOf(),minLearningRate,iterations),
      SD_NATIVE_FLOAT_TYPES);
}


void cbowInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                   int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
                   double alpha, sd::LongType randomValue, int numLabels, NDArray &inferenceVector, const bool trainWords,
                   int numWorkers,int iterations,double minLearningRate) {
  auto xType = syn0.dataType();
  auto hsRounds = codes.lengthOf();
  BUILD_SINGLE_SELECTOR(
      xType, cbow_,
      (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector,
          target, ngStarter,
          context.bufferAsT<int>(), lockedWords.bufferAsT<int>(),
          indices.bufferAsT<int>(), codes.bufferAsT<int>(), alpha,
          randomValue, (int)context.lengthOf(), hsRounds, nsRounds, (int)syn0.sizeAt(0),
          (int)syn0.sizeAt(1), (int)expTable.lengthOf(), (int)negTable.lengthOf(),
          numLabels, trainWords,minLearningRate,iterations),
      SD_NATIVE_FLOAT_TYPES);
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
            indices, codes, alpha.e<double>(0),
            randomValue.e<sd::LongType>(0), hsRounds, nsRounds, (int)syn0.sizeAt(0), (int)syn0.sizeAt(1),
            (int)expTable.lengthOf(), (int)negTable.lengthOf(),minLearningRate,iterations),
        SD_NATIVE_FLOAT_TYPES);
  } else if (ngStarter.isVector() || target.isVector()) {
    // batch mode
    BUILD_SINGLE_SELECTOR(xType, skipgramBatchExec_,
                          (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, target, ngStarter,
                              indices, codes, alpha, randomValue, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1),
                              expTable.lengthOf(), negTable.lengthOf(), preciseMode, numWorkers,iterations,minLearningRate),
                          SD_NATIVE_FLOAT_TYPES);
  } else
    THROW_EXCEPTION("SkipGram: target must have rank 0 or 1");
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
        (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector,
            target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0),
            context.bufferAsT<int>(), lockedWords.bufferAsT<int>(),
            indices.bufferAsT<int>(), codes.bufferAsT<int>(), alpha.e<double>(0),
            randomValue.e<sd::LongType>(0), (int)context.lengthOf(), hsRounds, nsRounds, (int)syn0.sizeAt(0),
            (int)syn0.sizeAt(1), (int)expTable.lengthOf(), (int)negTable.lengthOf(),
            numLabels.isEmpty() ? 0 : numLabels.e<int>(0), trainWords,minLearningRate,iterations),
        SD_NATIVE_FLOAT_TYPES);
  } else if (context.rankOf() == 2 && indices.rankOf() == 2) {
    BUILD_SINGLE_SELECTOR(
        xType, cbowBatchExec_,
        (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, context, lockedWords, target, ngStarter,
            indices, codes, alpha, randomValue, numLabels, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1), expTable.lengthOf(),
            negTable.isEmpty() ? 0 : negTable.lengthOf(), trainWords, numWorkers,minLearningRate,iterations),
        SD_NATIVE_FLOAT_TYPES);
  } else
    THROW_EXCEPTION("CBOW: context must have rank 0/1 or 2");
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
